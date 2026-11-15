"""Flix v5: cut every clip again so audio, lips and label cover the same window.

Why (reports/flix_v4/decision_log.md, edge check 15 Nov 2026): the v4 clip wavs were trimmed to
the labelled words and cover only ~77 % of the mouth-crop window (median), so the lips move over
silence that no label describes, and words at the cut are often half-heard. Padding the TED test
audio by 0.32 s per side lowered the audio-only twin's edge-word error from 22 % to 19 %.

v5 window = the v4 window widened by PAD frames per side (fewer on the left if the source starts
sooner). For that window the mouth crops (same Auto-AVSR crop as v4) and the 16 kHz audio come from
the same source video, and the label is every Whisper word with at least half its duration inside
the window, read from a context cut with CTX seconds either side.

  jobs      (Alienware) flix_avsr/jobs_{ted,film}.csv, only clips in data/flix_v4c -> flix_v5/jobs_*.csv
  ted       (Alienware) per talk: download the MP4 again, crop + cut audio + context, delete it
  films     (laptop)    the same from the local films (scripts.mine_flix titles)
  relabel   (Alienware) Whisper large-v3-turbo with word timestamps on every context cut; resumable,
                        runs alongside the cutting and finishes after flix_v5/.ted_done and .films_done
  manifests (Alienware) data/flix_v4c/*.csv -> data/flix_v5/*.csv (new paths and labels, v4 filters)

Outputs: flix_v5/clips/<stem>.{npz,wav}, flix_v5/ctx/<stem>.wav, flix_v5/whisper_words.jsonl.
  PYTHONPATH=asr_pkgs:. .venv/bin/python -m scripts.flix_v5_recut ted --workers 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

FPS, SR, SPF = 25, 16000, 640
PAD = 8            # frames per side (0.32 s): the edge check's plateau
CTX = 1.5          # seconds of context either side for Whisper
OUT = Path("flix_v5")
SETS = ("train_mix", "film_val", "flix_test", "ted_val", "ted_test")


def window(start_s: float, n_frames: int) -> tuple[float, int, int]:
    """v4 window -> (v5 start, v5 frames, left pad frames)."""
    left = min(PAD, int(start_s * FPS))
    return start_s - left / FPS, n_frames + left + PAD, left


# ------------------------------------------------------------------ jobs
def cmd_jobs(a) -> None:
    keep = set()
    for s in SETS:
        for r in csv.DictReader(open(f"data/flix_v4c/{s}.csv")):
            keep.add(Path(r["mouths_path"]).stem)
    OUT.mkdir(exist_ok=True)
    for kind in ("ted", "film"):
        rows = [r for r in csv.DictReader(open(f"flix_avsr/jobs_{kind}.csv")) if r["stem"] in keep]
        out = []
        for r in rows:
            s, n, left = window(float(r["start_s"]), int(r["n_frames"]))
            out.append({"stem": r["stem"], "title": r["title"], "start_s": f"{s:.3f}", "n_frames": n,
                        "left_pad": left, "v4_start_s": r["start_s"], "v4_frames": r["n_frames"]})
        with open(OUT / f"jobs_{kind}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)
        print(f"{kind}: {len(out)} clips")


# ------------------------------------------------------------------ one source (talk or film)
def cut_audio(ff: str, src: str, jobs: list[dict], out: Path) -> None:
    """Clip wav (exactly n_frames * 640 samples) and context wav for each job, from one source file.

    The whole source is decoded once: input seeking (-ss before -i) skips the AAC priming
    compensation and shifts the audio by ~49 ms against the video (smoke test 15 Nov).
    """
    import soundfile as sf
    todo = [j for j in jobs if not ((out / "clips" / f"{j['stem']}.wav").exists()
                                    and (out / "ctx" / f"{j['stem']}.json").exists())]
    if not todo:
        return
    raw = subprocess.run([ff, "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    full = np.frombuffer(raw, np.int16)
    for j in todo:
        s, n = float(j["start_s"]), int(j["n_frames"])
        c0 = max(0.0, s - CTX)
        a0, b0 = int(round(c0 * SR)), int(round((s + n / FPS + CTX) * SR))
        w = full[a0:b0].astype(np.float32) / 32768.0
        off = int(round(s * SR)) - a0
        seg = w[off: off + n * SPF]
        seg = np.pad(seg, (0, n * SPF - len(seg)))
        sf.write(out / "ctx" / f"{j['stem']}.wav", w, SR, subtype="FLOAT")
        sf.write(out / "clips" / f"{j['stem']}.wav", seg, SR, subtype="FLOAT")
        (out / "ctx" / f"{j['stem']}.json").write_text(json.dumps({"ctx_start": c0, "start": s, "dur": n / FPS}))


def _crop_jobs(src: str, jobs: list[dict], out: Path) -> list[tuple]:
    return [(src, j["stem"], float(j["start_s"]), int(j["n_frames"]), str(out)) for j in jobs
            if not (out / "clips" / f"{j['stem']}.npz").exists()]


def _log(out: Path, res: list[dict]) -> None:
    with open(out / "crop_stats.jsonl", "a") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")


def cmd_ted(a) -> None:
    from scripts.flix_recrop import _pool, crop_clip, fetch_video
    from scripts.ted_harvest import ffmpeg_exe
    for d in ("clips", "ctx"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    url_of = {}
    for line in open("flix_ted/talks_done.tsv"):
        u = line.split("\t")[0].strip()
        if u.startswith("http"):
            url_of[u.rstrip("/").rsplit("/", 1)[-1][:60]] = u
    by_slug: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(OUT / "jobs_ted.csv")):
        by_slug.setdefault(r["title"].split("/", 1)[1], []).append(r)
    done_f = OUT / "talks_done.tsv"
    done = {l.split("\t")[0] for l in open(done_f)} if done_f.exists() else set()
    todo = [s for s in by_slug if s not in done][: a.limit_talks or None]
    print(f"{len(by_slug)} talks, {len(todo)} to do", flush=True)
    ff = ffmpeg_exe()
    q: queue.Queue = queue.Queue(maxsize=a.downloaders + 1)
    it, lock = iter(todo), threading.Lock()

    def downloader():
        while True:
            with lock:
                s = next(it, None)
            if s is None:
                q.put(None); return
            q.put(fetch_video(url_of[s], tmp, ff) if s in url_of else {"slug": s, "skip": "no url"})

    for _ in range(a.downloaders):
        threading.Thread(target=downloader, daemon=True).start()
    finished, n_ok = 0, 0
    with _pool(a.workers) as ex:
        while finished < a.downloaders:
            r = q.get()
            if r is None:
                finished += 1; continue
            s = r["slug"]
            if "skip" in r:
                status = r["skip"]
            else:
                try:
                    cut_audio(ff, str(r["video"]), by_slug[s], OUT)
                    res = list(ex.map(crop_clip, _crop_jobs(str(r["video"]), by_slug[s], OUT), chunksize=4))
                    _log(OUT, res)
                    ok = sum(x["status"] == "ok" for x in res); n_ok += ok
                    status = f"ok {ok}/{len(res)}"
                except Exception as e:                       # one bad talk must not stop the run
                    status = f"error {type(e).__name__}: {e}"[:200]
                shutil.rmtree(r["dir"], ignore_errors=True)
            with open(done_f, "a") as f:
                f.write(f"{s}\t{status}\n")
            done.add(s)
            if len(done) % 25 == 0 or a.limit_talks:
                print(f"{time.strftime('%H:%M')} {len(done)}/{len(by_slug)} talks ({s}: {status})", flush=True)
    if not a.limit_talks:
        (OUT / ".ted_done").touch()
    print(f"DONE: {len(done)} talks, {n_ok} clips cropped this run", flush=True)


def cmd_films(a) -> None:
    import scripts.mine_flix as mf
    from scripts.flix_recrop import _pool, crop_clip
    out = Path(a.out)
    for d in ("clips", "ctx"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if a.extra_titles and Path(a.extra_titles).exists():
        mf.EXTRA.extend(tuple((l.split("|", 2) + [""])[:3]) for l in Path(a.extra_titles).read_text().splitlines() if l.strip())
    video_of = {k: str(v) for k, v, _ in mf._titles()}
    rows = list(csv.DictReader(open(a.jobs)))
    missing = sorted({r["title"] for r in rows if r["title"] not in video_of or not Path(video_of[r["title"]]).exists()})
    if missing:
        print(f"no video for {len(missing)} titles: {missing}", flush=True)
    by_title: dict[str, list[dict]] = {}
    for r in rows:
        if r["title"] not in missing:
            by_title.setdefault(r["title"], []).append(r)
    ff = shutil.which("ffmpeg") or "ffmpeg"
    n = 0
    with _pool(a.workers) as ex:
        for t, js in sorted(by_title.items()):
            js.sort(key=lambda r: float(r["start_s"]))
            cut_audio(ff, video_of[t], js, out)
            for res in ex.map(crop_clip, _crop_jobs(video_of[t], js, out), chunksize=8):
                _log(out, [res])
            n += len(js)
            print(f"{time.strftime('%H:%M')} {t}: {len(js)} clips ({n}/{len(rows)})", flush=True)
    print(f"DONE: {n} film clips", flush=True)


# ------------------------------------------------------------------ relabel
def norm(text: str) -> str | None:
    t = text.lower().replace("’", "'").replace("'", "")
    t = re.sub(r"[-—–/]", " ", t)
    t = re.sub(r"[.,!?;:\"()\[\]“”…]", " ", t)
    if re.search(r"[^a-z ]", t):
        return None
    return " ".join(t.split()) or None


def cmd_relabel(a) -> None:
    import soundfile as sf
    import torch
    import whisper
    cache_f = OUT / "whisper_words.jsonl"
    seen = set()
    if cache_f.exists():
        for ln in cache_f.read_text().splitlines():
            if ln.strip():
                seen.add(json.loads(ln)["stem"])
    model = whisper.load_model("large-v3-turbo", device="cuda")
    stems = [r["stem"] for k in ("ted", "film") for r in csv.DictReader(open(OUT / f"jobs_{k}.csv"))]
    with open(cache_f, "a") as cf:
        while True:
            ready = [s for s in stems if s not in seen and (OUT / "ctx" / f"{s}.json").exists()
                     and (OUT / "ctx" / f"{s}.wav").exists()]
            for i, s in enumerate(ready):
                meta = json.loads((OUT / "ctx" / f"{s}.json").read_text())
                w, _ = sf.read(OUT / "ctx" / f"{s}.wav", dtype="float32")
                res = model.transcribe(w, language="en", fp16=True, temperature=0.0, word_timestamps=True,
                                       condition_on_previous_text=False)
                words = [{"w": x["word"], "s": round(x["start"], 3), "e": round(x["end"], 3),
                          "p": round(x.get("probability", 0.0), 3)}
                         for seg in res.get("segments", []) for x in seg.get("words", [])]
                segs = res.get("segments") or [{}]
                cf.write(json.dumps({"stem": s, **meta, "words": words,
                                     "no_speech": max(x.get("no_speech_prob", 0.0) for x in segs),
                                     "compression": max(x.get("compression_ratio", 0.0) for x in segs),
                                     "avg_logprob": min(x.get("avg_logprob", 0.0) for x in segs)}) + "\n")
                seen.add(s)
                if i % 500 == 0:
                    cf.flush(); print(f"{time.strftime('%H:%M')} relabel {len(seen)}/{len(stems)}", flush=True)
            cf.flush()
            if (OUT / ".ted_done").exists() and (OUT / ".films_done").exists():
                left = [s for s in stems if s not in seen and (OUT / "ctx" / f"{s}.json").exists()]
                if not left:
                    break
            elif not ready:
                time.sleep(120)
    print(f"DONE: relabelled {len(seen)}/{len(stems)}", flush=True)
    del model; torch.cuda.empty_cache()


def label_of(d: dict) -> str:
    """Words with at least half their duration inside the clip window."""
    lo = d["start"] - d["ctx_start"]; hi = lo + d["dur"]
    keep = []
    for x in d["words"]:
        s, e = x["s"], max(x["e"], x["s"] + 1e-3)
        if min(e, hi) - max(s, lo) >= 0.5 * (e - s):
            keep.append(x["w"])
    return " ".join(keep)


def cmd_manifests(a) -> None:
    from lipscribe.data.dataset import ctc_min_frames
    from lipscribe.data.tokenizer import encode
    src, dst = Path("data/flix_v4c"), Path("data/flix_v5")
    dst.mkdir(parents=True, exist_ok=True)
    lab = {}
    for ln in open(OUT / "whisper_words.jsonl"):
        d = json.loads(ln); lab[d["stem"]] = d
    clips = OUT.resolve() / "clips"
    stats = {}
    for f in src.glob("*.csv"):
        rows = list(csv.DictReader(open(f)))
        if f.stem not in SETS:
            shutil.copy(f, dst / f.name); continue
        keep, drop = [], {"not_cut": 0, "no_speech": 0, "unspellable": 0, "hallucination": 0, "ctc_len": 0}
        for r in rows:
            stem = Path(r["mouths_path"]).stem
            npz, wav, d = clips / f"{stem}.npz", clips / f"{stem}.wav", lab.get(stem)
            if not (npz.exists() and wav.exists() and d):
                drop["not_cut"] += 1; continue
            n_frames = int(round(d["dur"] * FPS))
            raw = label_of(d)
            t = norm(raw)
            if not raw.strip() or (d["no_speech"] > 0.6 and not d["words"]):
                drop["no_speech"] += 1; continue
            if t is None:
                drop["unspellable"] += 1; continue
            if d["compression"] > 2.4 or d["avg_logprob"] < -1.0 or len(t) > 22 * n_frames / FPS + 10:
                drop["hallucination"] += 1; continue
            if ctc_min_frames(encode(t)) > n_frames:
                drop["ctc_len"] += 1; continue
            keep.append({**r, "transcript": t, "mouths_path": str(npz), "audio_path": str(wav),
                         "speech_start_s": "", "valid_path": ""})
        with open(dst / f.name, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(keep)
        stats[f.stem] = {"before": len(rows), "kept": len(keep), "dropped": drop}
        print(f.stem, stats[f.stem], flush=True)
    (dst / "recut_stats.json").write_text(json.dumps(stats, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["jobs", "ted", "films", "relabel", "manifests"])
    ap.add_argument("--out", default=str(OUT), help="films: output folder (laptop)")
    ap.add_argument("--jobs", default=str(OUT / "jobs_film.csv"))
    ap.add_argument("--extra-titles", default="scripts/flix_asr_films.txt")
    ap.add_argument("--tmp", default="ted/tmp_v5")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--downloaders", type=int, default=4)
    ap.add_argument("--limit-talks", type=int, default=0)
    a = ap.parse_args()
    {"jobs": cmd_jobs, "ted": cmd_ted, "films": cmd_films, "relabel": cmd_relabel, "manifests": cmd_manifests}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Flix v4: re-crop every clip the Auto-AVSR way (lipscribe.preprocessing.avsr_crop).

Only the mouth crops change: clip ids, time windows, frame sampling (scripts.mine_flix.mine_cue),
audio, Whisper labels and splits stay as they are. Output: <out>/clips/<old stem>.npz with the
same keys and length as before ({crops [T,96,96] uint8, valid [T]}).

  jobs       (Alienware) data/flix_v4 manifests + flix_v3/manifest_local.csv -> <out>/jobs_{ted,film}.csv
  ted        (Alienware) per talk: download the same 640x480 MP4 again, crop its clips, delete it
  films      (laptop)    crop film clips from ~/Videos/entertainment (scripts.mine_flix titles)
  manifests  (Alienware) data/flix_v4/*.csv -> data/flix_v4c/*.csv with mouths_path -> new crops

Resumable: clips whose npz exists are skipped; finished talks are listed in <out>/talks_done.tsv.
  PYTHONPATH=asr_pkgs .venv/bin/python -m scripts.flix_recrop ted --out flix_avsr --workers 12
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SETS = ("train_mix", "film_val", "flix_test", "ted_val", "ted_test")
JOB_FIELDS = ["stem", "title", "start_s", "dur_s", "n_frames"]


# ------------------------------------------------------------------ one clip
def read_frames(video: str, a: float, n: int, fps: float = 25.0) -> list:
    """Exactly scripts.mine_flix.mine_cue's frame sampling (so video stays in sync with the wav)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, a - 0.2) * 1000)
    targets = [a + k / fps for k in range(n)]
    frames, ti = [], 0
    while ti < n:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t + 0.02 < targets[ti]:
            continue
        while ti < n and t + 0.02 >= targets[ti]:
            frames.append(fr); ti += 1
    cap.release()
    return frames


def crop_clip(job) -> dict:
    """(video, stem, start_s, n_frames, out_dir) -> stats dict; writes <out_dir>/clips/<stem>.npz."""
    import cv2
    import scripts.mine_flix as mf
    from lipscribe.preprocessing import avsr_crop as ac
    video, stem, a, n, out_dir = job
    try:
        frames = read_frames(video, a, n)
        if len(frames) < n * 0.95:
            return {"stem": stem, "status": "frames"}
        frames = frames[:n] + [frames[-1]] * (n - len(frames))
        h0, w0 = frames[0].shape[:2]
        sc = 640.0 / w0 if w0 > 640 else 1.0
        rgbs = [cv2.cvtColor(cv2.resize(f, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB) for f in frames]
        h, w = rgbs[0].shape[:2]
        lms, prev = [], None
        for rgb in rgbs:
            p = ac.pick_face(mf._faces(rgb), w, h, prev)
            lms.append(p); prev = p if p is not None else prev
        found = sum(l is not None for l in lms)
        full = ac.interpolate(lms)
        if full is None:
            return {"stem": stem, "status": "no face"}
        crops, over = ac.crop_sequence(rgbs, full)
        valid = np.array([l is not None for l in lms])
        T = max(mf.T_GRID, n)                        # same length as the old crop file
        if crops.shape[0] < T:
            pad = T - crops.shape[0]
            crops = np.concatenate([crops, np.repeat(crops[-1:], pad, 0)])
            valid = np.concatenate([valid, np.zeros(pad, bool)])
        dst = Path(out_dir) / "clips" / f"{stem}.npz"
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, crops=crops[:T], valid=valid[:T])
        os.replace(tmp, dst)
        return {"stem": stem, "status": "ok", "face_frac": round(found / n, 3), "overflow": over}
    except Exception as e:                           # never kill the pool on one bad clip
        return {"stem": stem, "status": f"error {type(e).__name__}: {e}"[:160]}


# ------------------------------------------------------------------ jobs
def cmd_jobs(a) -> None:
    meta = {r["utt_id"]: r for r in csv.DictReader(open(a.meta))}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), {"ted": [], "film": []}
    for s in SETS:
        for r in csv.DictReader(open(Path(a.data) / f"{s}.csv")):
            stem = Path(r["mouths_path"]).stem
            if stem in seen:
                continue
            seen.add(stem); m = meta[stem]
            rows["ted" if stem.startswith("ted") else "film"].append(
                {"stem": stem, "title": m["title"], "start_s": m["start_s"], "dur_s": m["dur_s"], "n_frames": m["n_frames"]})
    for k, rs in rows.items():
        with open(out / f"jobs_{k}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, JOB_FIELDS); wr.writeheader(); wr.writerows(rs)
        print(f"{k}: {len(rs)} clips, {len({r['title'] for r in rs})} sources -> {out}/jobs_{k}.csv")


def _pending(out: Path, rows: list[dict]) -> list[dict]:
    return [r for r in rows if not (out / "clips" / f"{r['stem']}.npz").exists()]


def _log(out: Path, results: list[dict]) -> None:
    with open(out / "recrop_stats.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver"))


# ------------------------------------------------------------------ TED (Alienware)
def fetch_video(url: str, tmp: Path, ff: str) -> dict:
    """scripts.ted_harvest.fetch without subtitles/audio: the same progressive <=1600k MP4."""
    import yt_dlp
    slug = url.rstrip("/").rsplit("/", 1)[-1][:60]
    base = dict(quiet=True, no_warnings=True, noplaylist=True, retries=3, socket_timeout=30,
                ffmpeg_location=ff, noprogress=True, allowed_extractors=["TedTalk"])
    d = tmp / slug
    try:
        with yt_dlp.YoutubeDL(base) as y:
            info = y.extract_info(url, download=False)
        prog = [f for f in info.get("formats") or [] if str(f.get("format_id", "")).startswith("h264")
                and (f.get("tbr") or 0) <= 1600]
        if not prog:
            return {"slug": slug, "skip": "no progressive <=1600k mp4"}
        v = max(prog, key=lambda f: f.get("tbr") or 0)
        d.mkdir(parents=True, exist_ok=True)
        with yt_dlp.YoutubeDL({**base, "format": v["format_id"], "outtmpl": str(d / "v.%(ext)s")}) as y:
            if y.download([url]) != 0:
                raise RuntimeError("yt-dlp failed")
        video = next(p for p in d.iterdir() if p.name.startswith("v.") and not p.name.endswith(".part"))
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        return {"slug": slug, "skip": f"download: {type(e).__name__}: {e}"[:200]}
    return {"slug": slug, "video": video, "dir": d}


def cmd_ted(a) -> None:
    from scripts.ted_harvest import ffmpeg_exe
    out = Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True)
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    url_of = {}
    for line in open(a.talks_done):
        u, status = line.rstrip("\n").split("\t")[:2]
        if status == "ok":
            url_of[u.rstrip("/").rsplit("/", 1)[-1][:60]] = u
    by_slug: dict[str, list[dict]] = {}
    only = {Path(r["mouths_path"]).stem for r in csv.DictReader(open(a.stems_from))} if a.stems_from else None
    for r in csv.DictReader(open(out / "jobs_ted.csv")):
        if only is not None and r["stem"] not in only:
            continue
        by_slug.setdefault(r["title"].split("/", 1)[1], []).append(r)
    done_f = out / "talks_done.tsv"
    done = {l.split("\t")[0] for l in open(done_f)} if done_f.exists() else set()
    todo = [s for s in by_slug if s not in done]
    if a.limit_talks:
        todo = todo[:a.limit_talks]
    print(f"{len(by_slug)} talks, {len(todo)} to do, {sum(len(by_slug[s]) for s in todo)} clips", flush=True)
    ff = ffmpeg_exe()
    q: queue.Queue = queue.Queue(maxsize=a.downloaders + 1)
    it = iter(todo); lock = threading.Lock()

    def downloader():
        while True:
            with lock:
                s = next(it, None)
            if s is None:
                q.put(None); return
            q.put(fetch_video(url_of[s], tmp, ff) if s in url_of else {"slug": s, "skip": "no url"})

    for _ in range(a.downloaders):
        threading.Thread(target=downloader, daemon=True).start()
    ex, finished, n_ok, n_clips = _pool(a.workers), 0, 0, 0
    with ex:
        while finished < a.downloaders:
            r = q.get()
            if r is None:
                finished += 1; continue
            s = r["slug"]
            if "skip" in r:
                status = r["skip"]
            else:
                jobs = [(str(r["video"]), j["stem"], float(j["start_s"]), int(j["n_frames"]), str(out))
                        for j in _pending(out, by_slug[s])]
                res = list(ex.map(crop_clip, jobs, chunksize=4))
                _log(out, res); shutil.rmtree(r["dir"], ignore_errors=True)
                ok = sum(x["status"] == "ok" for x in res); n_ok += ok; n_clips += len(res)
                status = f"ok {ok}/{len(res)}"
            with open(done_f, "a") as f:
                f.write(f"{s}\t{status}\n")
            done.add(s)
            if len(done) % 25 == 0 or a.limit_talks:
                print(f"{len(done)}/{len(by_slug)} talks, {n_ok}/{n_clips} clips ok this run ({s}: {status})", flush=True)
    print(f"DONE: {len(done)} talks; {n_ok}/{n_clips} clips ok this run", flush=True)


# ------------------------------------------------------------------ films (laptop)
def cmd_films(a) -> None:
    import scripts.mine_flix as mf
    out = Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True)
    if a.extra_titles and Path(a.extra_titles).exists():
        mf.EXTRA.extend(tuple((l.split("|", 2) + [""])[:3]) for l in Path(a.extra_titles).read_text().splitlines() if l.strip())
    video_of = {k: str(v) for k, v, _ in mf._titles()}
    rows = _pending(out, list(csv.DictReader(open(a.jobs))))
    missing = sorted({r["title"] for r in rows if r["title"] not in video_of or not Path(video_of[r["title"]]).exists()})
    if missing:
        print(f"no video for {len(missing)} titles: {missing}", flush=True)
    rows = [r for r in rows if r["title"] not in missing]
    rows.sort(key=lambda r: (r["title"], float(r["start_s"])))          # sequential seeks per film
    print(f"{len(rows)} film clips to crop", flush=True)
    jobs = [(video_of[r["title"]], r["stem"], float(r["start_s"]), int(r["n_frames"]), str(out)) for r in rows]
    n_ok = 0
    with _pool(a.workers) as ex:
        for i, res in enumerate(ex.map(crop_clip, jobs, chunksize=8), 1):
            _log(out, [res]); n_ok += res["status"] == "ok"
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} ({n_ok} ok)", flush=True)
    print(f"DONE: {n_ok}/{len(jobs)} film clips ok", flush=True)


# ------------------------------------------------------------------ manifests
def cmd_manifests(a) -> None:
    src, dst, clips = Path(a.data), Path(a.dst), Path(a.out).resolve() / "clips"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.csv"):
        rows = list(csv.DictReader(open(f)))
        if not rows or "mouths_path" not in rows[0] or f.stem not in SETS:
            shutil.copy(f, dst / f.name); continue
        keep = []
        for r in rows:
            p = clips / Path(r["mouths_path"]).name
            if p.exists():
                keep.append({**r, "mouths_path": str(p)})
        with open(dst / f.name, "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(rows[0])); wr.writeheader(); wr.writerows(keep)
        print(f"{f.stem}: {len(keep)}/{len(rows)} clips with new crops")
    for extra in ("relabel_stats.json", "whisper_cache.jsonl"):
        if (src / extra).exists() and not (dst / extra).exists():
            os.symlink((src / extra).resolve(), dst / extra)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["jobs", "ted", "films", "manifests"])
    ap.add_argument("--out", default="flix_avsr")
    ap.add_argument("--data", default="data/flix_v4")
    ap.add_argument("--dst", default="data/flix_v4c")
    ap.add_argument("--meta", default="flix_v3/manifest_local.csv")
    ap.add_argument("--talks-done", default="flix_ted/talks_done.tsv")
    ap.add_argument("--tmp", default="ted/tmp_recrop")
    ap.add_argument("--jobs", default="flix_avsr/jobs_film.csv")
    ap.add_argument("--extra-titles", default="scripts/flix_asr_films.txt")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--downloaders", type=int, default=4)
    ap.add_argument("--stems-from", default=None, help="ted: only clips listed in this manifest (smoke test)")
    ap.add_argument("--limit-talks", type=int, default=0, help="first N talks only (smoke test)")
    a = ap.parse_args()
    {"jobs": cmd_jobs, "ted": cmd_ted, "films": cmd_films, "manifests": cmd_manifests}[a.cmd](a)


if __name__ == "__main__":
    main()

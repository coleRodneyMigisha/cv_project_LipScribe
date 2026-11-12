"""X6 'Flix' TED: download TED talks (480p), mine mouth clips, delete the downloads (Alienware).

For each talk URL (shuffled; ted/talks_playlists.txt = the 3,040 main-stage talks in TED's 772 playlists): yt-dlp fetches the metadata and
keeps the talk only if its audio track is English, English subtitles exist and it is
<= --max-min minutes. Then it downloads the 640x480 video, the English VTT subtitles and the
audio track; the audio becomes 16 kHz mono FLAC (bundled imageio-ffmpeg). Every subtitle cue
of 0.8 to --max-dur s is mined with the exact Flix/GRID pipeline (scripts/mine_flix.mine_cue:
MediaPipe face mesh -> 96x96 grey mouth crops, same filters). When a talk's cues are done its
video, audio and subtitles are deleted; only clips (npz + wav) and manifest rows stay.

~3% of talks (by a hash of the URL) are held out as partition "ted_test" (scored on its own);
the rest are "train". Film val/test are untouched. Resumable via <out>/talks_done.tsv.

  PYTHONPATH=asr_pkgs .venv/bin/python -m scripts.ted_harvest --out flix_ted --hours 400
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import multiprocessing as mp
import random
import re
import shutil
import subprocess
import sys
import threading
import queue
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import scripts.mine_flix as mf

FIELDS = ["utt_id", "speaker", "title", "transcript", "start_s", "dur_s", "n_frames", "mouth_valid_frac", "partition"]


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def fetch(url: str, tmp: Path, max_min: float, ff: str) -> dict:
    """Download one talk -> {'slug','video','flac','vtt'} or {'skip': reason}."""
    import yt_dlp
    slug = url.rstrip("/").rsplit("/", 1)[-1][:60]
    # TED-hosted talks only: never follow a talk page to YouTube (bot checks we must not bypass)
    base = dict(quiet=True, no_warnings=True, noplaylist=True, retries=3, socket_timeout=30,
                ffmpeg_location=ff, noprogress=True, allowed_extractors=["TedTalk"])
    try:
        with yt_dlp.YoutubeDL(base) as y:
            info = y.extract_info(url, download=False)
    except Exception as e:
        return {"slug": slug, "skip": f"info: {type(e).__name__}"}
    if not info.get("subtitles", {}).get("en"):
        return {"slug": slug, "skip": "no English subtitles"}
    if (info.get("duration") or 0) > max_min * 60 or (info.get("duration") or 0) < 120:
        return {"slug": slug, "skip": "duration"}
    fmts = info.get("formats") or []
    aud = [f for f in fmts if f.get("vcodec") == "none" and f.get("acodec") != "none"]
    if aud and not all((f.get("language") or "en").startswith("en") for f in aud):
        return {"slug": slug, "skip": "audio not English"}
    # TED's progressive MP4s ("h264-<kbps>") hold 640x480 video WITH the audio in one file and
    # download fast; the HLS streams are split (audio-only HLS fails) and slow.
    prog = [f for f in fmts if str(f.get("format_id", "")).startswith("h264") and (f.get("tbr") or 0) <= 1600]
    if not prog:
        return {"slug": slug, "skip": "no progressive <=1600k mp4"}
    v = max(prog, key=lambda f: f.get("tbr") or 0)
    d = tmp / slug; d.mkdir(parents=True, exist_ok=True)
    try:
        # a fresh download (re-using the processed info dict gave HTTP 404 on TED's CDN)
        with yt_dlp.YoutubeDL({**base, "format": v["format_id"], "outtmpl": str(d / "v.%(ext)s"),
                               "writesubtitles": True, "subtitleslangs": ["en"], "subtitlesformat": "vtt"}) as y:
            if y.download([url]) != 0:
                raise RuntimeError("yt-dlp failed")
        video = next(p for p in d.iterdir() if p.name.startswith("v.") and not p.name.endswith(".vtt"))
        vtt = next(p for p in d.iterdir() if p.name.endswith(".vtt"))
        flac = d / "audio.flac"
        subprocess.run([ff, "-nostdin", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "flac", str(flac)], check=True)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        return {"slug": slug, "skip": f"download: {type(e).__name__}"}
    return {"slug": slug, "video": video, "flac": flac, "vtt": vtt, "dir": d,
            "minutes": (info.get("duration") or 0) / 60}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--talks", type=Path, default=Path("ted/talks_playlists.txt"))
    ap.add_argument("--out", type=Path, default=Path("flix_ted"))
    ap.add_argument("--tmp", type=Path, default=Path("ted/tmp"))
    ap.add_argument("--hours", type=float, default=400, help="stop after this many hours of kept clips")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--downloaders", type=int, default=6)
    ap.add_argument("--max-dur", type=float, default=5.0)
    ap.add_argument("--max-min", type=float, default=45)
    ap.add_argument("--test-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    (a.out / "clips").mkdir(parents=True, exist_ok=True); a.tmp.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    urls = [u.strip() for u in a.talks.read_text().splitlines() if u.strip()]
    random.Random(a.seed).shuffle(urls)
    log = a.out / "talks_done.tsv"
    done = set()
    if log.exists():
        done = {l.split("\t", 1)[0] for l in log.read_text().splitlines()}
    urls = [u for u in urls if u not in done]
    mf_path = a.out / "manifest_local.csv"
    if mf_path.exists() and mf_path.stat().st_size and not mf_path.read_text().startswith("utt_id,"):
        mf_path.write_text(",".join(FIELDS) + "\n" + mf_path.read_text())      # header was missing
    new_file = not mf_path.exists() or mf_path.stat().st_size == 0
    mfh = open(mf_path, "a", newline="")
    wr = csv.DictWriter(mfh, fieldnames=FIELDS)
    if new_file:
        wr.writeheader()
    kept_s = 0.0
    if not new_file:
        with open(mf_path, newline="") as fh:
            kept_s = sum(float(r["dur_s"]) for r in csv.DictReader(fh))
    print(f"{len(urls)} talks to try, {kept_s / 3600:.2f} h kept so far", flush=True)

    ready: queue.Queue = queue.Queue(maxsize=a.downloaders + 1)
    url_q: queue.Queue = queue.Queue()
    for u in urls:
        url_q.put(u)
    stop = threading.Event()

    def downloader():
        while not stop.is_set():
            try:
                u = url_q.get_nowait()
            except queue.Empty:
                return
            r = fetch(u, a.tmp, a.max_min, ff); r["url"] = u
            while not stop.is_set():
                try:
                    ready.put(r, timeout=5); break
                except queue.Full:
                    continue
    threads = [threading.Thread(target=downloader, daemon=True) for _ in range(a.downloaders)]
    for t in threads:
        t.start()

    # worker recycling only on Python >= 3.13: ProcessPoolExecutor(max_tasks_per_child=...) can deadlock
    # when it replaces workers on 3.11/3.12 (seen on the Alienware, Python 3.11)
    _recycle = {"max_tasks_per_child": 40} if sys.version_info >= (3, 13) else {}
    ex = ProcessPoolExecutor(a.workers, mp_context=mp.get_context("forkserver"), **_recycle)
    talks: dict[str, dict] = {}          # slug -> {'left': n, 'rows': [], 'talk': r}
    pending: dict = {}
    t0 = dt.datetime.now(); n_talks = n_skip = 0

    def finish(slug):
        nonlocal n_talks
        st = talks.pop(slug)
        wr.writerows(st["rows"]); mfh.flush()
        shutil.rmtree(st["talk"]["dir"], ignore_errors=True)           # delete video, audio, subtitles
        with open(log, "a") as fh:
            fh.write(f"{st['talk']['url']}\tok\t{st['talk']['minutes']:.1f}\t{len(st['rows'])}\n")
        n_talks += 1

    while True:
        # keep the mining pool fed from downloaded talks
        while len(pending) < 4 * a.workers:
            try:
                r = ready.get(timeout=1 if pending else 30)
            except queue.Empty:
                if not any(t.is_alive() for t in threads) and ready.empty():
                    break
                if pending:
                    break
                continue
            if "skip" in r:
                n_skip += 1
                with open(log, "a") as fh:
                    fh.write(f"{r['url']}\tskip\t{r['skip']}\n")
                continue
            part = "ted_test" if int(hashlib.md5(r["url"].encode()).hexdigest(), 16) % 1000 < a.test_frac * 1000 else "train"
            key = f"{'ted_test' if part == 'ted_test' else 'ted'}/{r['slug']}"
            jobs = []
            for s0, s1, lines in mf.parse_srt(r["vtt"].read_text(errors="ignore")):
                t = mf.clean_text(lines)
                if t and 0.8 <= s1 - s0 <= a.max_dur:
                    uid = f"{re.sub(r'[^a-z0-9]+', '_', key.lower())[:50]}_t{int(round(s0 * 1000)):08d}"
                    jobs.append((key, str(r["video"]), s0, s1, t, a.out, uid, str(r["flac"])))
            talks[r["slug"]] = {"left": len(jobs), "rows": [], "talk": r, "part": part}
            if not jobs:
                finish(r["slug"]); continue
            for j in jobs:
                pending[ex.submit(mf.mine_cue, j)] = r["slug"]
        if not pending:
            break
        fin, _ = wait(list(pending), timeout=900, return_when=FIRST_COMPLETED)
        if not fin:
            print("no cue finished in 15 min, stopping", flush=True); break
        for f in fin:
            slug = pending.pop(f)
            try:
                row = f.result()
            except Exception:
                row = None
            st = talks[slug]
            if row:
                row["speaker"] = st["part"] if st["part"] == "ted_test" else "ted"
                row["partition"] = st["part"]
                st["rows"].append(row); kept_s += float(row["dur_s"])
            st["left"] -= 1
            if st["left"] == 0:
                finish(slug)
                if n_talks % 10 == 0:
                    hrs = (dt.datetime.now() - t0).total_seconds() / 3600
                    print(f"{dt.datetime.now():%H:%M} {n_talks} talks mined, {n_skip} skipped, "
                          f"{kept_s / 3600:.2f} h of clips kept ({hrs:.1f} h running)", flush=True)
        if kept_s / 3600 >= a.hours:
            print(f"target of {a.hours} h reached", flush=True); break
    stop.set()
    ex.shutdown(wait=False, cancel_futures=True)
    mfh.close()
    print(f"DONE: {n_talks} talks, {kept_s / 3600:.2f} h of clips", flush=True)


if __name__ == "__main__":
    main()

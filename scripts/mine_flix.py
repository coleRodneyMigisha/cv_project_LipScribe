"""X6 'Flix': mine single-speaker, camera-facing speaking clips from local films/series.

For every subtitle cue of 0.8-3.0 s: decode its frames at 25 fps, keep it only if
exactly one large, roughly frontal face is visible in >= 90% of frames and the lips
move, then crop the mouth with the GRID pipeline (lipscribe.preprocessing.crop)
and cut the matching 16 kHz mono audio. Crops are padded to T=75 (valid=False on
padding) so the clips drop straight into GridDataset.

Output (out_dir): clips/<utt_id>.npz (crops, valid), clips/<utt_id>.wav,
manifest_local.csv, splits.csv (title -> train/val/test).

Usage: python -m scripts.mine_flix --out <dir> --deadline 03:17 --workers 4 [--videos <root>]
       (all cues: --per-movie 100000 --per-episode 100000)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import multiprocessing as mp
import os
import random
import re
import subprocess
from pathlib import Path

import numpy as np

V = Path.home() / "Videos" / "entertainment"
T_GRID = 75
FPS = 25.0

# (title, video, subtitle) -- subtitle is an .srt path or ("embedded", stream_hint)
def _titles():
    m = V / "movies"; b = V / "b & m"; s = V / "series"
    t = [
        ("i_robot", b / "I Robot (2004) [1080p]/I.Robot.2004.1080p.BrRip.x264.YIFY.mp4", b / "I Robot (2004) [1080p]/I.Robot.2004.1080.BrRip.x264.YIFY.srt"),
        ("casino_royale", b / "James Bond Casino Royale (2006)/Casino.Royale.2006.720p.BRrip.x264.YIFY.mp4", b / "James Bond Casino Royale (2006)/Casino.Royale.2006.720p.BRrip.x264.YIFY.srt"),
        ("quantum", b / "James Bond Quantum of Solace (2008) [1080p]/James.Bond.Quantum.of.Solace.2008.1080p.BRrip.x264.YIFY.mp4", b / "James Bond Quantum of Solace (2008) [1080p]/James.Bond.Quantum.of.Solace.2008.1080p.BRrip.x264.YIFY.srt"),
        ("skyfall", b / "Skyfall (2012) [1080p]/Skyfall.2012.1080p.BrRip.x264.YIFY.mp4", b / "Skyfall (2012) [1080p]/Skyfall.2012.1080p.BrRip.x264.YIFY.srt"),
    ]
    for d in sorted(m.iterdir()):
        if not d.is_dir():
            continue
        vids = list(d.glob("*.mp4"))
        eng = d / "Subs" / "English.srt"
        main = [p for p in d.glob("*.srt")]
        sub = eng if eng.exists() else (main[0] if main else None)
        if not sub:
            cand = [p for p in (d / "Subs").glob("*eng*.srt") if "HI" not in p.name] if (d / "Subs").exists() else []
            sub = cand[0] if cand else None
        if vids and sub:
            key = re.sub(r"[^a-z0-9]+", "_", d.name.split("(")[0].strip().lower()).strip("_")
            t.append((key, vids[0], sub))
    t.append(("little_women", m / "tfpdl.com-littleWomen1872wd.mkv", ("embedded", 0)))
    for p in sorted((b / "theChosen").glob("sn*/*.mkv")):
        t.append((f"the_chosen/{p.stem}", p, ("embedded", 0)))
    for p in sorted((s / "IT - Welcome to Derry - Season 1").glob("*.mkv")):
        t.append((f"it_derry/{p.stem[:30]}", p, ("embedded", "eng")))
    for p in sorted((s / "vision_quest").glob("*.mkv")):
        t.append((f"vision_quest/{p.stem[:20]}", p, ("embedded", "eng")))
    return t

# split by title (speaker-independent analogue)
VAL = {"the_devil_all_the_time", "vision_quest"}
TEST = {"challengers", "the_drama", "little_women"}

def group_of(key: str) -> str:
    return key.split("/")[0]

def partition_of(key: str) -> str:
    g = group_of(key)
    return "test" if g in TEST else "val" if g in VAL else "train"


# ------------------------------------------------------------------ subtitles
TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")

def parse_srt(text: str):
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        for i, l in enumerate(lines):
            m = TS.search(l)
            if m:
                g = list(map(int, m.groups()))
                a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                out.append((a, b, lines[i + 1:]))
                break
    return out

def clean_text(lines):
    if sum(1 for l in lines if l.strip().startswith("-")) >= 2:
        return None                      # two speakers in one cue
    t = " ".join(lines)
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>|\{[^}]*\}", " ", t)
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", t)
    t = re.sub(r"^\s*-\s*", " ", t)
    t = re.sub(r"\b[A-Z][A-Z .']+:\s*", " ", t)   # SPEAKER:
    if re.search(r"[0-9♪#]", t):
        return None
    t = t.lower().replace("&", " and ")
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z]+", " ", t).strip()
    t = re.sub(r"\s+", " ", t)
    if len(t.split()) < 2 or len(t) > 60:
        return None
    return t

def load_cues(video: Path, sub, tmp: Path):
    if isinstance(sub, tuple):
        hint = sub[1]
        if hint == "eng":
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
                                    "stream_tags=language,title", "-of", "csv=p=0", str(video)],
                                   capture_output=True, text=True).stdout.strip().split("\n")
            idx = 0
            for i, row in enumerate(probe):
                if row.startswith("eng") and "SDH" not in row and "Forced" not in row:
                    idx = i; break
        else:
            idx = int(hint)
        out = tmp / (re.sub(r"\W+", "_", video.stem) + ".srt")
        if not out.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", f"0:s:{idx}", str(out)], check=False)
        text = out.read_text(errors="ignore") if out.exists() else ""
    else:
        text = Path(sub).read_text(errors="ignore")
    return parse_srt(text)


# ------------------------------------------------------------------ worker
_LM = None

def _landmarker():
    global _LM
    if _LM is None:
        import mediapipe as mp_
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(Path("models/face_landmarker.task").resolve())),
            running_mode=vision.RunningMode.IMAGE, num_faces=2,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
        _LM = (mp_, vision.FaceLandmarker.create_from_options(opts))
    return _LM

def _faces(rgb):
    mp_, lm = _landmarker()
    res = lm.detect(mp_.Image(image_format=mp_.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    return res.face_landmarks or []

def _frontal_ok(f, w, h):
    xs = np.array([p.x for p in f]) * w
    width = xs.max() - xs.min()
    nose, l, r = f[1].x * w, f[234].x * w, f[454].x * w   # nose tip, cheek extremes
    lo, hi = min(l, r), max(l, r)
    if hi - lo < 1:
        return False, width
    ratio = (nose - lo) / (hi - lo)
    return 0.3 <= ratio <= 0.7, width

def _aperture(f):
    return abs(f[13].y - f[14].y) / (abs(f[152].y - f[10].y) + 1e-6)

def mine_cue(job):
    import cv2
    from lipscribe.preprocessing.crop import detect_lip_box, get_lip_indices, process_frames
    key, video, a, b, text, out_dir, uid = job
    try:
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, a - 0.2) * 1000)
        n = int(round((b - a) * FPS))
        targets = [a + k / FPS for k in range(n)]
        frames, ti, last = [], 0, None
        while ti < n:
            ok, fr = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t + 0.02 < targets[ti]:
                last = fr; continue
            while ti < n and t + 0.02 >= targets[ti]:
                frames.append(fr); ti += 1
        cap.release()
        if len(frames) < n * 0.95 or n < 20:
            return None
        h0, w0 = frames[0].shape[:2]
        sc = 640.0 / w0 if w0 > 640 else 1.0
        rgbs = [cv2.cvtColor(cv2.resize(f, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for f in frames]
        h, w = rgbs[0].shape[:2]
        small = [cv2.resize(cv2.cvtColor(r, cv2.COLOR_RGB2GRAY), (64, 36)).astype(np.float32) for r in rgbs]
        if max(float(np.mean(np.abs(x - y))) for x, y in zip(small, small[1:])) > 25:
            return None                  # camera cut inside the cue
        mid = _faces(rgbs[len(rgbs) // 2])
        if len(mid) != 1:
            return None
        ok, width = _frontal_ok(mid[0], w, h)
        if not ok or width < 0.12 * w:
            return None
        lips = get_lip_indices()
        boxes, aps, multi = [], [], 0
        for rgb in rgbs:
            fs = _faces(rgb)
            if len(fs) > 1:
                multi += 1
            if len(fs) >= 1:
                boxes.append(detect_lip_box(fs[0], w, h, lips)); aps.append(_aperture(fs[0]))
            else:
                boxes.append(None)
        valid_frac = sum(bx is not None for bx in boxes) / len(boxes)
        if valid_frac < 0.9 or multi > 0.1 * len(boxes) or len(aps) < 5 or np.std(aps) < 0.01:
            return None
        res = process_frames(rgbs, boxes, target_frame_count=n)
        crops, valid = res.crops, res.valid
        if crops.shape[0] < T_GRID:
            pad = T_GRID - crops.shape[0]
            crops = np.concatenate([crops, np.repeat(crops[-1:], pad, 0)])
            valid = np.concatenate([valid, np.zeros(pad, bool)])
        crops, valid = crops[:T_GRID], valid[:T_GRID]
        np.savez_compressed(out_dir / "clips" / f"{uid}.npz", crops=crops.astype(np.uint8), valid=valid)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(video),
                        "-vn", "-ac", "1", "-ar", "16000", str(out_dir / "clips" / f"{uid}.wav")], check=True)
        return {"utt_id": uid, "speaker": group_of(key), "title": key, "transcript": text,
                "start_s": f"{a:.3f}", "dur_s": f"{b - a:.3f}", "n_frames": n,
                "mouth_valid_frac": f"{valid_frac:.3f}", "partition": partition_of(key)}
    except Exception as e:  # never kill the pool on one bad cue
        return None


def main():
    global V
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deadline", default="03:17", help="local HH:MM to stop mining")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-movie", type=int, default=220)
    ap.add_argument("--per-episode", type=int, default=45)
    ap.add_argument("--videos", type=Path, default=V,
                    help="root holding movies/, series/, 'b & m/' (default ~/Videos/entertainment)")
    ap.add_argument("--stall-min", type=float, default=15,
                    help="stop if no cue finishes for this long (a killed worker hangs multiprocessing.Pool)")
    args = ap.parse_args()
    V = args.videos
    out = args.out; (out / "clips").mkdir(parents=True, exist_ok=True); (out / "subs").mkdir(exist_ok=True)
    now = dt.datetime.now()
    hh, mm = map(int, args.deadline.split(":"))
    deadline = now.replace(hour=hh, minute=mm, second=0)
    if deadline < now:
        deadline += dt.timedelta(days=1)

    rng = random.Random(2026)
    per_title = []
    for key, video, sub in _titles():
        if not Path(video).exists():
            continue
        cues = load_cues(Path(video), sub, out / "subs")
        cand = []
        for a, b, lines in cues:
            if 0.8 <= b - a <= 3.0:
                t = clean_text(lines)
                if t:
                    cand.append((a, b, t))
        rng.shuffle(cand)
        cap = args.per_episode if "/" in key else args.per_movie
        per_title.append([(key, str(video), a, b, t, out, f"{re.sub(r'[^a-z0-9]+', '_', key)}_{i:04d}")
                          for i, (a, b, t) in enumerate(cand[:cap])])
        print(f"{key}: {len(cues)} cues, {len(cand)} candidates, using {min(cap, len(cand))}", flush=True)
    # interleave titles so an early stop still covers every title
    jobs = [j for grp in zip(*[g + [None] * (max(map(len, per_title)) - len(g)) for g in per_title]) for j in grp if j]
    print(f"{len(jobs)} cue jobs, deadline {deadline:%H:%M}", flush=True)

    fields = ["utt_id", "speaker", "title", "transcript", "start_s", "dur_s", "n_frames", "mouth_valid_frac", "partition"]
    mf = out / "manifest_local.csv"
    # resume: keep rows already in the manifest (whose clips exist) and skip those cues
    prev = []
    if mf.exists():
        with open(mf, newline="") as fh:
            prev = [r for r in csv.DictReader(fh) if (out / "clips" / f"{r['utt_id']}.npz").exists()]
    have = {r["utt_id"] for r in prev}
    jobs = [j for j in jobs if j[-1] not in have]
    if prev:
        print(f"resuming: {len(prev)} clips already kept, {len(jobs)} cues left", flush=True)
    kept = len(prev); done = 0
    # maxtasksperchild: recycle workers so decoder memory cannot build up (OOM on 7 GB laptop)
    with open(mf, "w", newline="") as fh, mp.Pool(args.workers, maxtasksperchild=40) as pool:
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        wr.writerows(prev); fh.flush()
        it = pool.imap_unordered(mine_cue, jobs, chunksize=2)
        while True:
            try:
                r = it.next(timeout=args.stall_min * 60)
            except StopIteration:
                break
            except mp.TimeoutError:
                print(f"no cue finished in {args.stall_min:.0f} min (worker died?), stopping", flush=True)
                pool.terminate(); break
            done += 1
            if r:
                kept += 1; wr.writerow(r); fh.flush()
            if done % 100 == 0:
                print(f"{dt.datetime.now():%H:%M:%S} {done}/{len(jobs)} processed, {kept} kept", flush=True)
            if dt.datetime.now() >= deadline:
                print("deadline reached, stopping", flush=True)
                pool.terminate(); break
    with open(out / "splits.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["speaker", "partition"])
        for g in sorted({group_of(j[0]) for grp in per_title for j in grp}):
            w.writerow([g, partition_of(g)])
    print(f"DONE {kept} clips kept of {done} processed", flush=True)


if __name__ == "__main__":
    main()

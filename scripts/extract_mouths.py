"""Cache mouth crops for every GRID utterance (card W2-02).

One command, run from the repository root::

    python -m scripts.extract_mouths
    python -m scripts.extract_mouths --data-root Data/grip_corpus/grid-corpus-dataset-for-training-lipnet/data --workers 8

It finds every video with the same index as ``verify_dataset.py`` and writes, per
utterance::

    data/processed/mouths/s<N>/<utt_id>.npy        uint8 [T, 96, 96] grey crops
    data/processed/mouth_valid/s<N>/<utt_id>.npy   bool  [T] Face Mesh success per frame

Utterances run in parallel across CPU cores. The run is resumable: every
finished utterance is appended to ``extraction_log.csv`` straight away, and a
re-run skips anything already logged as done whose files exist. Stop it with
Ctrl+C at any point and run the same command again. Failed utterances
(``error``) are retried on the next run.

At the end it writes ``reports/face_mesh_report.csv`` (``speaker, utterance_id,
success_rate``) for ``verify_dataset.py --face-mesh-report``, and
``extraction_meta.json`` with the MediaPipe version and crop parameters for the
manifest. Statuses in the log:

    ok         crops saved, Face Mesh succeeded on >= 90 % of frames
    excluded   crops saved, but > 10 % of frames failed; the data pipeline drops it
    no_video   audio/alignment exist but there is no video (e.g. all of s21)
    error      the video could not be read or cropped (message in ``error``)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from scripts.verify_dataset import DatasetIndex, _default_data_root

DEFAULT_OUT = Path("data/processed/mouths")
DEFAULT_VALID_OUT = Path("data/processed/mouth_valid")
DEFAULT_MODEL = Path("models/face_landmarker.task")
DEFAULT_REPORT = Path("reports/face_mesh_report.csv")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
GRID_FRAMES = 75  # 3 s at 25 fps; crops are padded/truncated to this many frames

LOG_FIELDS = ("speaker", "utterance_id", "status", "frames", "success_rate",
              "seconds", "video_path", "error")
FINAL_STATUSES = {"ok", "excluded", "no_video"}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _init_worker() -> None:
    # One OpenCV thread per process; the parallelism comes from the pool.
    import cv2
    cv2.setNumThreads(1)


def _save_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)  # a killed run never leaves a half-written .npy


def default_crop_fn(video_path, model_path, target_frame_count):
    from lipscribe.preprocessing.crop import crop_video
    return crop_video(video_path, model_path, target_frame_count=target_frame_count)


def process_one(task: dict) -> dict:
    """Crop one utterance and save its files. Never raises: errors go in the log row."""
    row = {"speaker": task["speaker"], "utterance_id": task["utt"],
           "video_path": task["video"], "frames": "", "success_rate": "", "error": ""}
    start = time.time()
    try:
        result = task["crop_fn"](task["video"], task["model"], task["frames"])
        crops = np.asarray(result.crops, dtype=np.uint8)
        valid = np.asarray(result.valid, dtype=bool)
        if crops.ndim != 3 or len(crops) != len(valid) or len(crops) == 0:
            raise ValueError(f"unexpected crop output: crops {crops.shape}, valid {valid.shape}")
        _save_atomic(Path(task["crop_path"]), crops)
        _save_atomic(Path(task["valid_path"]), valid)
        row.update(status="excluded" if result.exclude_utterance else "ok",
                   frames=len(crops), success_rate=f"{valid.mean():.4f}")
    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop 33,000
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    row["seconds"] = f"{time.time() - start:.2f}"
    return row


# ---------------------------------------------------------------------------
# Planning, logging and reports
# ---------------------------------------------------------------------------

def read_log(path: Path) -> dict[tuple[str, str], dict]:
    """Last log row per utterance (later runs override earlier ones)."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["speaker"], r["utterance_id"]): r for r in csv.DictReader(f)}


def plan_tasks(index: DatasetIndex, out_dir: Path, valid_dir: Path, done: dict, *,
               model: Path, frames: int, crop_fn: Callable,
               speakers: set[str] | None = None) -> tuple[list[dict], list[dict], int]:
    """-> (tasks to run, ``no_video`` log rows, number skipped as already done)."""
    tasks, no_video, skipped = [], [], 0
    for speaker, utt in index.utterances:
        if speakers and speaker not in speakers:
            continue
        crop_path = out_dir / speaker / f"{utt}.npy"
        valid_path = valid_dir / speaker / f"{utt}.npy"
        previous = done.get((speaker, utt), {}).get("status")
        video = index.files["video"].get((speaker, utt))
        if video is None:
            if previous != "no_video":
                no_video.append({"speaker": speaker, "utterance_id": utt, "status": "no_video",
                                 "frames": "", "success_rate": "", "seconds": "0",
                                 "video_path": "", "error": "no video file found"})
            else:
                skipped += 1
            continue
        if previous in ("ok", "excluded") and crop_path.exists() and valid_path.exists():
            skipped += 1
            continue
        tasks.append({"speaker": speaker, "utt": utt, "video": str(video.path),
                      "crop_path": str(crop_path), "valid_path": str(valid_path),
                      "model": str(model), "frames": frames, "crop_fn": crop_fn})
    return tasks, no_video, skipped


def write_face_mesh_report(log: dict, path: Path) -> int:
    """CSV in the format ``verify_dataset.py --face-mesh-report`` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in log.values() if r["status"] in ("ok", "excluded")]
    rows.sort(key=lambda r: (int(r["speaker"][1:]), r["utterance_id"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "utterance_id", "success_rate", "excluded"])
        for r in rows:
            writer.writerow([r["speaker"], r["utterance_id"], r["success_rate"],
                             r["status"] == "excluded"])
    return len(rows)


def write_meta(path: Path, model: Path, frames: int) -> None:
    from lipscribe.preprocessing import crop
    try:
        import mediapipe
        mp_version = mediapipe.__version__
    except ImportError:
        mp_version = "not installed"
    meta = {
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "mediapipe_version": mp_version,
        "face_landmarker_model": str(model),
        "face_landmarker_sha256": _sha256(model) if model.exists() else None,
        "output_size": list(crop.OUTPUT_SIZE),
        "smoothing_radius": crop.SMOOTHING_RADIUS,
        "lip_box_margin": 1.25,  # hard-coded in crop.detect_lip_box
        "max_invalid_fraction": crop.MAX_INVALID_FRACTION,
        "target_frames": frames,
        "color": "grey",
        "dtype": "uint8",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def disk_usage(*dirs: Path) -> int:
    return sum(p.stat().st_size for d in dirs if d.exists() for p in d.rglob("*.npy"))


def summarise(log: dict) -> str:
    counts: dict[str, int] = {}
    for r in log.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return ", ".join(f"{k}={counts[k]}" for k in sorted(counts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_mouths(data_root: Path, out_dir: Path = DEFAULT_OUT, valid_dir: Path = DEFAULT_VALID_OUT,
                   *, model: Path = DEFAULT_MODEL, report: Path = DEFAULT_REPORT,
                   workers: int | None = None, frames: int = GRID_FRAMES,
                   speakers: set[str] | None = None, limit: int | None = None,
                   crop_fn: Callable = default_crop_fn, index: DatasetIndex | None = None) -> dict:
    """Run (or resume) extraction; return the final log, one row per utterance."""
    log_path = out_dir / "extraction_log.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    if index is None:
        print(f"Indexing {data_root} ...")
        index = DatasetIndex(Path(data_root).resolve())
    for err in index.scan_errors:
        print(f"  warning: {err}", file=sys.stderr)

    tasks, no_video, skipped = plan_tasks(index, out_dir, valid_dir, read_log(log_path),
                                          model=model, frames=frames, crop_fn=crop_fn,
                                          speakers=speakers)
    if limit is not None:
        tasks = tasks[:limit]
    workers = max(1, workers or (os.cpu_count() or 2) - 1)
    print(f"{len(index.utterances)} utterances indexed: {skipped} already done, "
          f"{len(no_video)} without video, {len(tasks)} to process on {workers} worker(s).")

    new_log = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_log:
            writer.writeheader()
        writer.writerows(no_video)
        f.flush()

        t0 = time.time()
        results = _run(tasks, workers)
        try:
            for n, row in enumerate(results, 1):
                writer.writerow(row)
                f.flush()  # each finished utterance survives a crash
                if row["status"] == "error":
                    print(f"  error {row['speaker']}/{row['utterance_id']}: {row['error']}",
                          file=sys.stderr)
                if n % 100 == 0 or n == len(tasks):
                    rate = n / max(1e-9, time.time() - t0)
                    eta = (len(tasks) - n) / max(1e-9, rate)
                    print(f"  {n}/{len(tasks)}  {rate:.1f} utt/s  ETA {eta / 60:.0f} min")
        except KeyboardInterrupt:
            print("\nInterrupted; progress is saved. Run the same command to resume.")
            raise

    log = read_log(log_path)
    n_report = write_face_mesh_report(log, report)
    write_meta(out_dir / "extraction_meta.json", model, frames)
    print(f"Done: {summarise(log)}")
    print(f"Face Mesh report ({n_report} rows): {report}")
    print(f"Disk use: {disk_usage(out_dir, valid_dir) / 1e9:.2f} GB in {out_dir} and {valid_dir}")
    return log


def _run(tasks: list[dict], workers: int):
    if workers == 1 or len(tasks) <= 1:
        _init_worker()
        return map(process_one, tasks)
    # "spawn" everywhere: MediaPipe is not fork-safe, and it matches Windows.
    pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker)

    def results():
        try:
            yield from pool.imap_unordered(process_one, tasks, chunksize=4)
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
    return results()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=_default_data_root())
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="crop .npy root")
    p.add_argument("--valid-out", type=Path, default=DEFAULT_VALID_OUT, help="per-frame mask .npy root")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MediaPipe face_landmarker.task")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--workers", type=int, help="parallel processes (default: CPU cores - 1)")
    p.add_argument("--frames", type=int, default=GRID_FRAMES)
    p.add_argument("--speakers", nargs="+", help="only these speakers, e.g. s1 s2")
    p.add_argument("--limit", type=int, help="process at most N utterances (smoke test)")
    args = p.parse_args(argv)

    if not args.model.exists():
        print(f"Face Landmarker model not found: {args.model}\n"
              f"Download it with:\n  curl -L -o {args.model} {MODEL_URL}", file=sys.stderr)
        return 2
    if not args.data_root.is_dir():
        print(f"Data root not found: {args.data_root} (pass --data-root)", file=sys.stderr)
        return 2
    try:
        log = extract_mouths(args.data_root, args.out, args.valid_out, model=args.model,
                             report=args.report, workers=args.workers, frames=args.frames,
                             speakers={s.lower() for s in args.speakers} if args.speakers else None,
                             limit=args.limit)
    except KeyboardInterrupt:
        return 130
    return 1 if any(r["status"] == "error" for r in log.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

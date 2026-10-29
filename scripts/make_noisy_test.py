"""Write the fixed noisy validation and test sets (W2-06).

Every usable val/test utterance is mixed once per SNR (10, 5, 0, -5 dB) with a
seed fixed per (utterance, SNR), using only the val or test MUSAN noise pool, so
every model is evaluated on identical audio. "clean" is the 16 kHz audio itself
(data/processed/audio16k), which the dataset reads directly.

Output: data/processed/noisy_test/<snr>/<utt_id>.wav  (16 kHz, 16-bit, same length as the clean clip)
Log:    data/processed/noisy_test/mix_log.csv  (noise file, offset, alpha, measured SNR per mixture)

Resumable: existing files are skipped. Run on the Alienware from the repo root,
after the manifest has been built:

    python -m scripts.make_noisy_test --workers 8
    python -m scripts.make_noisy_test --limit 5          # quick smoke run
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from lipscribe.data.dataset import (
    SAMPLES_PER_FRAME, fit_length, load_wav, read_manifest, read_noise_pool, read_splits, snr_label,
)
from lipscribe.preprocessing.noise import NoisePool

DEFAULT_SNRS = (10.0, 5.0, 0.0, -5.0)
LOG_FIELDS = ["utt_id", "partition", "snr_db", "noise_file", "offset", "alpha",
              "measured_snr_db", "peak_scale", "path"]

_POOLS: dict[str, NoisePool] = {}  # per worker process


def mixture_seed(seed: int, utt_id: str, snr_db: float) -> int:
    """Stable seed per (utterance, SNR): same mixture on every machine and every run."""
    return zlib.crc32(f"{seed}|{utt_id}|{snr_db:g}".encode())


def _init_worker(pools: dict[str, list[str]]) -> None:
    global _POOLS
    _POOLS = {part: NoisePool(files) for part, files in pools.items()}


def process_one(task: dict) -> list[dict]:
    """Mix one utterance at every SNR still missing. Returns log rows."""
    pool = _POOLS[task["partition"]]
    n = int(task["n_frames"]) * SAMPLES_PER_FRAME
    clean = fit_length(load_wav(task["audio_path"]), n)
    rows = []
    for snr in task["snrs"]:
        out = Path(task["out"]) / snr_label(snr) / f"{task['utt_id']}.wav"
        rng = random.Random(mixture_seed(task["seed"], task["utt_id"], snr))
        y, info = pool.mix(clean, snr, rng)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.wav")
        sf.write(tmp, y, 16000, subtype="PCM_16")
        tmp.replace(out)  # atomic: a killed run never leaves a half-written file
        rows.append({"utt_id": task["utt_id"], "partition": task["partition"], "snr_db": snr,
                     "noise_file": info["noise_file"], "offset": info["offset"],
                     "alpha": f"{info['alpha']:.6g}", "measured_snr_db": f"{info['measured_snr_db']:.4f}",
                     "peak_scale": f"{info['peak_scale']:.6g}", "path": str(out)})
    return rows


def plan(rows, speaker_part, partitions, snrs, out: Path, seed: int, limit: int | None):
    tasks, already = [], 0
    for r in rows:
        part = speaker_part.get(r["speaker"].strip().lower())
        if part not in partitions or str(r.get("usable")).lower() != "true":
            continue
        todo = [s for s in snrs if not (out / snr_label(s) / f"{r['utt_id']}.wav").exists()]
        already += len(snrs) - len(todo)
        if todo:
            tasks.append({"utt_id": r["utt_id"], "partition": part, "audio_path": r["audio_path"],
                          "n_frames": r["n_frames"], "snrs": todo, "out": str(out), "seed": seed})
        if limit and len(tasks) >= limit:
            break
    return tasks, already


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    p.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    p.add_argument("--noise-splits", type=Path, default=None,
                   help="MUSAN pool CSV from make_noise_splits.py (default: Data/Noise_splits/splits.csv)")
    p.add_argument("--out", type=Path, default=Path("data/processed/noisy_test"))
    p.add_argument("--snrs", type=float, nargs="+", default=list(DEFAULT_SNRS))
    p.add_argument("--partitions", nargs="+", default=["val", "test"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="only the first N utterances (smoke run)")
    args = p.parse_args(argv)

    if not args.manifest.exists():
        print(f"ERROR: {args.manifest} not found; run python -m scripts.build_manifest first", file=sys.stderr)
        return 1
    pools = {}
    for part in args.partitions:
        files = read_noise_pool(args.noise_splits, part)
        if not files:
            print(f"ERROR: no '{part}' noise files found (run scripts/make_noise_splits.py, "
                  f"or pass --noise-splits)", file=sys.stderr)
            return 1
        missing = [f for f in files if not Path(f).exists()]
        if missing:
            print(f"ERROR: {len(missing)} {part} noise files missing, e.g. {missing[0]}", file=sys.stderr)
            return 1
        pools[part] = [str(f) for f in files]
    print("Noise pools:", {k: len(v) for k, v in pools.items()})

    tasks, already = plan(read_manifest(args.manifest), read_splits(args.splits), set(args.partitions),
                          args.snrs, args.out, args.seed, args.limit)
    print(f"{len(tasks)} utterances to mix ({already} files already done), "
          f"SNRs {[snr_label(s) for s in args.snrs]}, {args.workers} workers")

    log_path = args.out / "mix_log.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    stats, failed = defaultdict(list), []
    with open(log_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new_log:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(pools,)) as ex:
            futures = {ex.submit(process_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    for row in fut.result():
                        w.writerow(row)
                        stats[row["snr_db"]].append(float(row["measured_snr_db"]))
                except Exception as exc:  # keep going; report at the end
                    failed.append((futures[fut]["utt_id"], repr(exc)))
                if i % 500 == 0:
                    fh.flush()
                    print(f"  {i}/{len(tasks)}", flush=True)

    for snr, vals in sorted(stats.items(), reverse=True):
        print(f"  {snr_label(snr):>3} dB: {len(vals)} files, measured SNR "
              f"{np.min(vals):.3f} .. {np.max(vals):.3f} dB")
    if stats:
        print("Listen to one example per SNR (-5 dB should be very noisy, speech still present):")
        example = next(iter(tasks))["utt_id"] if tasks else None
        for snr in args.snrs:
            print(f"  {args.out / snr_label(snr) / f'{example}.wav'}")
    print(f"Log: {log_path}")
    if failed:
        print(f"FAILED {len(failed)} utterances, e.g. {failed[:3]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

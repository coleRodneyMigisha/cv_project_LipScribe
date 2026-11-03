"""Mean/std from the TRAINING partition only.

    python scripts/compute_norm_stats.py --noise-splits Data/Noise_splits/splits.csv
"""
import argparse
from pathlib import Path

import numpy as np

from lipscribe.data.dataset import GridDataset
from lipscribe.preprocessing.features import N_MELS, logmel
from lipscribe.preprocessing.norm_stats import STATS_PATH, RunningMoments, split_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--noise-splits", default=None)
    ap.add_argument("--n-utts", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=STATS_PATH)
    args = ap.parse_args()

    # Training partition only: GridDataset picks the SNR and noise exactly as training does.
    ds = GridDataset(args.manifest, args.splits, "train",
                     noise_splits=args.noise_splits, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(ds), size=min(args.n_utts, len(ds)), replace=False)

    audio, pixels = RunningMoments(N_MELS), RunningMoments(None)
    for k, i in enumerate(chosen):
        item = ds[int(i)]
        crops = item["video"]
        T = crops.shape[0]
        audio.update(logmel(item["audio"], n_frames=T))

        x = crops.astype(np.float32)
        if crops.dtype == np.uint8:
            x /= 255.0
        valid = item.get("video_valid")
        if valid is not None and len(valid) == T:
            x = x[np.asarray(valid, bool)]   # skip frames where no mouth was found
        pixels.update(x)

        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(chosen)}")

    assert 0.0 < pixels.mean < 1.0, f"pixel mean {pixels.mean} is not on a 0-1 scale"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        audio_mean=audio.mean.astype(np.float32),
        audio_std=audio.std.astype(np.float32),
        pixel_mean=np.float32(pixels.mean),
        pixel_std=np.float32(pixels.std),
        n_utts=len(chosen), n_frames=audio.n, seed=args.seed,
        train_split_hash=split_hash(r["utt_id"] for r in ds.rows),
    )
    print(f"Saved {args.out}: {len(chosen)} utterances, {audio.n} audio frames")


if __name__ == "__main__":
    main()
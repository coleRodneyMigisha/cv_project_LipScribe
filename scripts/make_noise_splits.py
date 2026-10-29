import csv
import random
from pathlib import Path

NOISE_DIR = Path("/home/Lipscribe-team/Desktop/cv_project_LipScribe/Data/musan/noise")
Output_DIR = Path("/home/Lipscribe-team/Desktop/cv_project_LipScribe/Data/Noise_splits")

SEED = 42
RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

def main():
    files = sorted(NOISE_DIR.glob("**/*.wav"))
    if not files:
        raise ValueError(f"No .wav files found in {NOISE_DIR}")

    rng = random.Random(SEED)
    rng.shuffle(files)

    n=len(files)
    n_train = int(n * RATIOS["train"])
    n_val = int(n * RATIOS["val"])
    
    rows =[]
    for i, f in enumerate(files):
        if i < n_train:
            pool = "train"
        elif i < n_train + n_val:
            pool = "val"
        else:
            pool = "test"

        try:
            rel = f.relative_to(NOISE_DIR.parent.parent)
        except ValueError:
            rel =f 
        rows.append({"file": str(rel), "pool": pool})

    with open(Output_DIR / "splits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "pool"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"{n} files -> {dict(Counter(r['pool'] for r in rows))}")
    print(f"Written to {Output_DIR / 'splits.csv'}")

if __name__ == "__main__":
    main()


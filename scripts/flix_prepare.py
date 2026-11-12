"""X6 'Flix': turn mined film clips into GridDataset manifests (run on the Alienware).

Inputs:  <flix>/manifest_local.csv + <flix>/clips/*.npz|*.wav   (scripts/mine_flix.py)
         <grid-root>/data/manifest.csv + splits.csv               (the main GRID data)
Outputs (<out>/):
    splits.csv       speaker -> partition for Flix titles and GRID speakers
    train_mix.csv    Flix train/val (train rows repeated --flix-repeat times)
                     + a GRID train/val subset, so fine-tuning does not forget GRID
    flix_test.csv    Flix test titles only
    grid_test.csv    a fixed GRID test subset (forgetting check)
    stats.json       clip counts, duration, vocabulary size and GRID overlap

Usage: python -m scripts.flix_prepare --flix flix --grid-root ../cv_project_LipScribe --out data/flix
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from lipscribe.data.dataset import ctc_min_frames
from lipscribe.data.tokenizer import encode

T_GRID = 75
FIELDS = ["utt_id", "speaker", "partition", "transcript", "usable", "mouths_path",
          "valid_path", "audio_path", "speech_start_s", "source"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fields=FIELDS) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flix", type=Path, default=Path("flix"))
    ap.add_argument("--grid-root", type=Path, default=Path("../cv_project_LipScribe"))
    ap.add_argument("--out", type=Path, default=Path("data/flix"))
    ap.add_argument("--grid-train", type=int, default=1000)
    ap.add_argument("--grid-val", type=int, default=60)
    ap.add_argument("--grid-test", type=int, default=600)
    ap.add_argument("--flix-repeat", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--extra-val", nargs="*", default=[],
                    help="train titles to move into val (a bigger val set for tuning); test is never touched")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    clips = args.flix.resolve() / "clips"

    flix, dropped = [], Counter()
    for r in read_csv(args.flix / "manifest_local.csv"):
        t = (r["transcript"] or "").strip()
        npz, wav = clips / f"{r['utt_id']}.npz", clips / f"{r['utt_id']}.wav"
        if len(t) < 2:
            dropped["empty"] += 1; continue
        if not (npz.exists() and wav.exists()):
            dropped["missing file"] += 1; continue
        if ctc_min_frames(encode(t)) > T_GRID:
            dropped["too many letters for 3 s"] += 1; continue
        part = r["partition"]
        if r["speaker"] in args.extra_val:
            if part == "test":
                raise SystemExit(f"--extra-val {r['speaker']} is a test title")
            part = "val"
        flix.append({"utt_id": f"flix_{r['utt_id']}", "speaker": r["speaker"],
                     "partition": part, "transcript": t, "usable": "True",
                     "mouths_path": str(npz), "valid_path": "", "audio_path": str(wav),
                     "speech_start_s": "", "source": "flix", "dur_s": r.get("dur_s", "")})

    grid_root = args.grid_root.resolve()
    grid_split = {r["speaker"]: r["partition"] for r in read_csv(grid_root / "data/splits.csv")}
    grid = {"train": [], "val": [], "test": []}
    for r in read_csv(grid_root / "data/manifest.csv"):
        part = grid_split.get(r["speaker"])
        if part not in grid or str(r.get("usable")).lower() != "true":
            continue
        row = {k: r.get(k, "") for k in FIELDS}
        for k in ("mouths_path", "valid_path", "audio_path"):
            if row[k] and not Path(row[k]).is_absolute():
                row[k] = str(grid_root / row[k])
        row["partition"], row["source"] = part, "grid"
        grid[part].append(row)
    for part in grid:
        rng.shuffle(grid[part])

    splits = {r["speaker"]: r["partition"] for r in flix}
    clash = set(splits) & set(grid_split)
    if clash:
        raise SystemExit(f"Flix titles clash with GRID speaker ids: {clash}")
    splits.update(grid_split)
    write_csv(args.out / "splits.csv", [{"speaker": s, "partition": p} for s, p in sorted(splits.items())],
              fields=["speaker", "partition"])

    f_train = [r for r in flix if r["partition"] == "train"]
    f_val = [r for r in flix if r["partition"] == "val"]
    f_test = [r for r in flix if r["partition"] == "test"]
    train_mix = (f_train * args.flix_repeat + f_val + grid["train"][:args.grid_train]
                 + grid["val"][:args.grid_val])          # --grid-train 0 --grid-val 0 -> Flix only
    write_csv(args.out / "train_mix.csv", train_mix)
    write_csv(args.out / "flix_test.csv", f_test)
    write_csv(args.out / "grid_test.csv", grid["test"][:args.grid_test])

    grid_words = {w for r in grid["train"] for w in r["transcript"].split()}
    words = Counter(w for r in flix for w in r["transcript"].split())
    train_words = {w for r in f_train for w in r["transcript"].split()}
    test_words = Counter(w for r in f_test for w in r["transcript"].split())
    stats = {
        "flix_clips": len(flix), "dropped": dict(dropped),
        "by_partition": {p: sum(r["partition"] == p for r in flix) for p in ("train", "val", "test")},
        "by_title": dict(Counter(r["speaker"] for r in flix).most_common()),
        "titles_by_partition": {p: sorted({r["speaker"] for r in flix if r["partition"] == p})
                                for p in ("train", "val", "test")},
        "minutes": round(sum(float(r["dur_s"] or 0) for r in flix) / 60, 1),
        "distinct_words": len(words), "word_tokens": sum(words.values()),
        "distinct_words_in_grid": len(set(words) & grid_words), "grid_vocab": len(grid_words),
        "grid_words": sorted(grid_words),
        "test_tokens_seen_in_flix_train": sum(c for w, c in test_words.items() if w in train_words),
        "test_tokens": sum(test_words.values()),
        "top_words": words.most_common(25),
        "train_mix_rows": len(train_mix), "grid_test_rows": len(grid["test"][:args.grid_test]),
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("by_title", "top_words", "grid_words")}, indent=2))
    if not f_train or not f_test:
        raise SystemExit("Need Flix clips in both train and test titles")


if __name__ == "__main__":
    main()

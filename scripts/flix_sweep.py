"""X6 'Flix': retrain Model P on Flix only (the P4 recipe) with a hyperparameter sweep.

Stage 1  random search over the space below (+ 2 anchors), each run selected on
         Flix validation CER (WER sits near 100% on film speech and barely moves).
Stage 2  final runs: the best configuration with seeds 1-3 for --final-epochs (150)
         and no early stopping (seed spread).
Stage 3  the three best-config runs scored ONCE on the untouched Flix test titles
         (lips + sound, lips only, sound only) and on the GRID test subset.

Anchors: "p4_recipe" = exactly how P4 was trained (B1 audio + B2-v2 visual encoders,
fresh fusion/CTC head, lr 3e-4) but on Flix; "from_p4" = start from the whole P4 copy.
Resumable: finished trials are read back from <out>/sweep.csv and skipped.

Usage: python -m scripts.flix_sweep --trials 24 [--max-epochs 60]
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BASE_CFG = REPO / "configs/p_init_b1_b2_v2.yaml"
P4 = "checkpoints/flix_base_p4.pt"
SPACE = {
    "init": ["p4_recipe", "from_p4"],
    "lr": [1e-4, 3e-4, 1e-3],
    "freeze": ["none", "visual_encoder"],
    "dropout": [0.3, 0.5],
    "weight_decay": [1e-2, 5e-2],
    "audio_dropout": [0.25, 0.5],
    "batch_size": [8, 16],
    "snrs": ["all", "mild"],
}
ANCHORS = {
    "p4_recipe": dict(init="p4_recipe", lr=3e-4, freeze="none", dropout=0.3, weight_decay=1e-2,
                      audio_dropout=0.25, batch_size=16, snrs="all"),
    "from_p4": dict(init="from_p4", lr=1e-4, freeze="none", dropout=0.3, weight_decay=1e-2,
                    audio_dropout=0.25, batch_size=16, snrs="all"),
}
FIELDS = ["trial", "seed", *SPACE, "best_epoch", "epochs", "val_cer", "val_wer", "run_dir"]


def make_cfg(h: dict, seed: int, data: Path, max_epochs: int, patience: int = 15) -> dict:
    cfg = yaml.safe_load(BASE_CFG.read_text())
    cfg = copy.deepcopy(cfg)
    cfg["experiment"] = "flix_p"
    cfg["seed"] = seed
    d = cfg["data"]
    d.update(manifest=str(data / "train_mix.csv"), splits=str(data / "splits.csv"),
             eval_snrs=["clean", "0"],
             train_snrs=["clean", 10, 5, 0, -5] if h["snrs"] == "all" else ["clean", "clean", 10, 5])
    d.pop("noisy_dir", None)
    m = cfg["model"]
    if h["init"] == "from_p4":
        m.pop("init_audio_from", None); m.pop("init_visual_from", None)
        m["init_from"] = {k: P4 for k in ("audio_encoder", "visual_encoder", "fusion", "ctc_head")}
    m["dropout"] = h["dropout"]
    m["modality_dropout"] = {"audio": h["audio_dropout"], "visual": 0.15}
    if h["freeze"] != "none":
        m["freeze"] = [h["freeze"]]
    t = cfg["train"]
    t.update(lr=h["lr"], weight_decay=h["weight_decay"], batch_size=h["batch_size"],
             max_epochs=max_epochs, patience=patience, select_metric="cer")
    return cfg


def best_from_history(run_dir: Path) -> dict:
    with open(run_dir / "history.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    best = min(rows, key=lambda r: float(r["val_cer"]))
    return {"best_epoch": best["epoch"], "epochs": len(rows),
            "val_cer": f"{float(best['val_cer']):.4f}", "val_wer": f"{float(best['val_wer']):.4f}"}


def run_trial(name: str, h: dict, seed: int, a, out: Path, max_epochs: int | None = None,
              patience: int | None = None) -> dict:
    cfg_dir = REPO / "configs/flix_sweep"; cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{name}_s{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(make_cfg(h, seed, a.data, max_epochs or a.max_epochs, patience or a.patience),
                                       sort_keys=False))
    print(f"\n=== {name} seed {seed}: {h}", flush=True)
    proc = subprocess.run([sys.executable, "-m", "scripts.train", "--config", str(cfg_path),
                           "--no-results", "--runs-root", str(REPO / "runs/flix_sweep")],
                          cwd=REPO, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "logs" / f"{name}_s{seed}.log").write_text(log)
    m = re.search(r"Run directory: (\S+)", log)
    if proc.returncode != 0 or not m:
        print(f"    FAILED (exit {proc.returncode}); see logs/{name}_s{seed}.log", flush=True)
        return {"trial": name, "seed": seed, **h, "val_cer": "nan", "val_wer": "nan", "run_dir": ""}
    run_dir = Path(m.group(1))
    row = {"trial": name, "seed": seed, **h, **best_from_history(run_dir), "run_dir": str(run_dir)}
    print(f"    val CER {row['val_cer']}  WER {row['val_wer']}  (best epoch {row['best_epoch']}/{row['epochs']})",
          flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/flix_only"))
    ap.add_argument("--out", type=Path, default=Path("reports/flix_sweep"))
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--patience", type=int, default=40,
                    help="tuning runs; a fresh CTC head (p4_recipe) emits only blanks for its first epochs")
    ap.add_argument("--final-epochs", type=int, default=150,
                    help="final runs: best config, seeds 1-3, no early stopping")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sweep_csv = a.out / "sweep.csv"
    done = []
    if sweep_csv.exists():
        with open(sweep_csv, newline="") as fh:
            done = list(csv.DictReader(fh))
    have = {(r["trial"], str(r["seed"])) for r in done}

    def record(row):
        new = not sweep_csv.exists()
        with open(sweep_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        done.append({k: str(v) for k, v in row.items()})

    # stage 1: anchors + random configurations (fixed RNG, so a restart picks the same ones)
    grid = [dict(zip(SPACE, v)) for v in itertools.product(*SPACE.values())]
    random.Random(a.seed).shuffle(grid)
    anchors = list(ANCHORS.values())
    picks = [g for g in grid if g not in anchors][: a.trials]
    trials = list(ANCHORS.items()) + [(f"t{i:02d}", g) for i, g in enumerate(picks)]
    for name, h in trials:
        if (name, "1") not in have:
            record(run_trial(name, h, 1, a, a.out))

    # stage 2: final runs = best configuration, seeds 1-3, --final-epochs with no early
    # stopping (best.pt is still the epoch with the lowest val CER)
    stage1 = [r for r in done if r["seed"] == "1" and r["val_cer"] != "nan" and not r["trial"].startswith("final")]
    best = min(stage1, key=lambda r: float(r["val_cer"]))
    h_best = {k: type(SPACE[k][0])(best[k]) for k in SPACE}
    final = f"final_{best['trial']}"
    print(f"\nBest stage-1 trial: {best['trial']} val CER {best['val_cer']} {h_best}", flush=True)
    for seed in (1, 2, 3):
        if (final, str(seed)) not in have:
            record(run_trial(final, h_best, seed, a, a.out, max_epochs=a.final_epochs,
                             patience=a.final_epochs))
    (a.out / "best.json").write_text(json.dumps({"trial": final, "from_trial": best["trial"],
                                                 "final_epochs": a.final_epochs, "hparams": h_best}, indent=2))

    # stage 3: test once
    finals = [r for r in done if r["trial"] == final and r["val_cer"] != "nan"]
    for r in sorted(finals, key=lambda r: r["seed"]):
        ck = Path(r["run_dir"]) / "best.pt"
        tag = f"sweep_s{r['seed']}"
        for suffix, extra in (("flix", []), ("flix_lips_only", ["--drop-modality", "audio"]),
                              ("flix_sound_only", ["--drop-modality", "visual"])):
            subprocess.run([sys.executable, "-m", "scripts.flix_eval", "--checkpoint", str(ck),
                            "--manifest", str(a.data / "flix_test.csv"), "--splits", str(a.data / "splits.csv"),
                            "--tag", f"{tag}_{suffix}", "--out", str(a.out), *extra], cwd=REPO, check=True)
        subprocess.run([sys.executable, "-m", "scripts.flix_eval", "--checkpoint", str(ck),
                        "--manifest", "data/flix/grid_test.csv", "--splits", "data/flix/splits.csv",
                        "--tag", f"{tag}_grid", "--out", str(a.out)], cwd=REPO, check=True)
    print("sweep finished", flush=True)


if __name__ == "__main__":
    main()

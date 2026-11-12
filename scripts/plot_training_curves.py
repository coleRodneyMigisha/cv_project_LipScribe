"""Plot training/validation curves and evaluation metrics for LipScribe runs.

Usage::

    # Plot a specific run's history
    python scripts/plot_training_curves.py --run-dir runs/b2_visual_only_20261104-120000

    # Plot directly from an epoch history CSV
    python scripts/plot_training_curves.py --csv runs/b2_visual_only_20261104-120000/history.csv

    # Plot model comparison curves (WER vs SNR) from experiments/results.csv
    python scripts/plot_training_curves.py --compare --results-csv experiments/results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_from_history_csv(csv_path: Path, out_path: Path | None = None) -> Path:
    """Generate multi-panel training & validation curves from history.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"History CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"History CSV is empty: {csv_path}")

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    val_wer = [float(r["val_wer"]) for r in rows]
    val_cer = [float(r["val_cer"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows] if "lr" in rows[0] else None

    # Detect per-SNR WER columns (wer_clean, wer_10, etc.)
    snr_cols = [c for c in rows[0].keys() if c.startswith("wer_")]

    fig, axes = plt.subplots(1, 3 if snr_cols else 2, figsize=(15 if snr_cols else 11, 4.5), constrained_layout=True)

    # 1. Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=1.8, markersize=4)
    ax1.plot(epochs, val_loss, "s-", label="Val Loss", color="#ff7f0e", linewidth=1.8, markersize=4)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CTC Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    # 2. Mean WER & CER
    ax2 = axes[1]
    ax2.plot(epochs, [w * 100 for w in val_wer], "o-", label="Val WER (%)", color="#d62728", linewidth=1.8, markersize=4)
    ax2.plot(epochs, [c * 100 for c in val_cer], "^-", label="Val CER (%)", color="#2ca02c", linewidth=1.8, markersize=4)
    best_idx = int(np.argmin(val_wer))
    best_ep = epochs[best_idx]
    best_w = val_wer[best_idx] * 100
    ax2.axvline(best_ep, color="gray", linestyle=":", linewidth=1.5, label=f"Best Ep {best_ep} ({best_w:.1f}%)")
    ax2.scatter([best_ep], [best_w], color="#d62728", s=90, zorder=5, edgecolor="black")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Error Rate (%)", fontsize=11)
    ax2.set_title(f"Validation Error Rates (Best WER: {best_w:.2f}%)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True)

    # 3. SNR breakdown (if available)
    if snr_cols:
        ax3 = axes[2]
        snr_order = ["wer_clean", "wer_10", "wer_5", "wer_0", "wer_-5"]
        ordered_cols = [c for c in snr_order if c in snr_cols] or snr_cols
        for c in ordered_cols:
            label = c.replace("wer_", "").replace("clean", "Clean") + (" dB" if c != "wer_clean" else "")
            snr_values = [float(r[c]) * 100 for r in rows]
            ax3.plot(epochs, snr_values, "o-", label=label, linewidth=1.5, markersize=3)
        ax3.set_xlabel("Epoch", fontsize=11)
        ax3.set_ylabel("Validation WER (%)", fontsize=11)
        ax3.set_title("WER across SNR Levels", fontsize=12, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(frameon=True, fontsize=9)

    if out_path is None:
        out_path = csv_path.parent / "learning_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved learning curves to: {out_path}")
    return out_path


def plot_comparison_snr_curves(results_csv: Path, out_path: Path | None = None) -> Path:
    """Plot comparative WER vs SNR curves across experiments from experiments/results.csv."""
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Results CSV is empty: {results_csv}")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    desired_snrs = ["clean", "10", "5", "0", "-5"]

    markers = ["o", "s", "^", "D", "v"]
    for i, row in enumerate(rows):
        exp_name = row.get("experiment", f"run_{i}")
        val_snr_str = row.get("val_wer_by_snr", "")
        if not val_snr_str:
            continue
        pairs = dict(item.split("=") for item in val_snr_str.split(";") if "=" in item)
        x_labels = [s for s in desired_snrs if s in pairs]
        if not x_labels:
            continue
        y_vals = [float(pairs[s]) * 100 for s in x_labels]
        display_x = [f"{s} dB" if s != "clean" else "Clean" for s in x_labels]
        marker = markers[i % len(markers)]
        ax.plot(display_x, y_vals, marker=marker, linewidth=2, markersize=7, label=exp_name)

    ax.set_xlabel("Noise Condition (SNR)", fontsize=11)
    ax.set_ylabel("Validation WER (%)", fontsize=11)
    ax.set_title("Speech Recognition Robustness Across Noise Conditions", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    if out_path is None:
        out_path = results_csv.parent / "snr_comparison_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved SNR comparison plot to: {out_path}")
    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, help="Run directory containing history.csv")
    parser.add_argument("--csv", type=Path, help="Path to history.csv directly")
    parser.add_argument("--out", type=Path, help="Output PNG path (optional)")
    parser.add_argument("--compare", action="store_true", help="Plot comparison across experiments from results.csv")
    parser.add_argument("--results-csv", type=Path, default=Path("experiments/results.csv"),
                        help="Path to experiments/results.csv (used with --compare)")

    args = parser.parse_args(argv)

    if args.compare:
        plot_comparison_snr_curves(args.results_csv, args.out)
    elif args.csv:
        plot_from_history_csv(args.csv, args.out)
    elif args.run_dir:
        csv_path = args.run_dir / "history.csv"
        plot_from_history_csv(csv_path, args.out)
    else:
        # Default behavior: if experiments/results.csv exists, plot comparison
        if args.results_csv.exists():
            plot_comparison_snr_curves(args.results_csv, args.out)
        else:
            parser.error("Specify --run-dir, --csv, or --compare")


if __name__ == "__main__":
    main()

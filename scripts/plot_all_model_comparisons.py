"""Generate comprehensive comparison visualizations across B1, B2, B3, B2 v2, B3 v2, and all 4 Model P variants."""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
ARTIFACT_DIR = Path("/home/Lipscribe-team/.gemini/antigravity-cli/brain/01da24eb-8728-4682-baa4-d05709b76504")


def plot_model_p_4way(out_path: Path) -> None:
    """3-panel figure comparing the 4 Model P initialization variants on the full GRID dataset (30-min budget)."""
    snr_labels = ["Clean", "10 dB", "5 dB", "0 dB", "-5 dB"]

    p_full = {
        "P1: Scratch (conv3d, 11 ep)": {
            "epochs":     list(range(1, 12)),
            "train_loss": [2.0792, 0.7119, 0.4115, 0.3197, 0.2539, 0.2070, 0.1681, 0.1370, 0.1182, 0.1032, 0.0973],
            "val_loss":   [1.2160, 1.2997, 0.9512, 0.8895, 0.7745, 0.6271, 0.5515, 0.5814, 0.5595, 0.5254, 0.5364],
            "val_wer":    [65.38,  58.94,  44.85,  40.08,  36.41,  33.26,  29.34,  30.33,  28.68,  28.13,  28.35],
            "test_greedy":  [12.58, 13.05, 13.55, 14.49, 16.28],
            "test_grammar": [9.40,  9.84,  10.30, 11.32, 13.16],
            "color": "#7f7f7f",
            "marker": "o",
        },
        "P2: Init B2 v1 Only (conv3d, 11 ep)": {
            "epochs":     list(range(1, 12)),
            "train_loss": [1.6007, 0.2850, 0.2082, 0.1764, 0.1549, 0.1283, 0.1085, 0.0915, 0.0810, 0.0712, 0.0664],
            "val_loss":   [1.4302, 1.4742, 1.3575, 1.9847, 1.8474, 1.3220, 1.9005, 1.6178, 1.5391, 1.5723, 1.6004],
            "val_wer":    [43.08,  39.06,  38.01,  39.77,  40.55,  36.21,  38.10,  35.64,  34.93,  35.26,  35.26],
            "test_greedy":  [18.19, 18.20, 18.53, 18.93, 19.85],
            "test_grammar": [15.38, 15.48, 15.67, 16.16, 17.01],
            "color": "#ff7f0e",
            "marker": "s",
        },
        "P3: Init B1 + B2 v1 (conv3d, 10 ep)": {
            "epochs":     list(range(1, 11)),
            "train_loss": [1.3908, 0.0803, 0.0517, 0.0443, 0.0352, 0.0281, 0.0239, 0.0199, 0.0171, 0.0149],
            "val_loss":   [0.5006, 0.5805, 0.5239, 0.3934, 0.4946, 0.7206, 0.3947, 0.5223, 0.5510, 0.5379],
            "val_wer":    [21.15,  21.02,  19.65,  15.64,  16.48,  19.38,  14.17,  16.98,  16.93,  16.77],
            "test_greedy":  [2.13, 2.67, 3.27, 4.33, 7.30],
            "test_grammar": [1.25, 1.62, 2.08, 2.88, 5.20],
            "color": "#1f77b4",
            "marker": "^",
        },
        "P4: Init B1 + B2 v2 (resnet18, 6 ep)": {
            "epochs":     list(range(1, 7)),
            "train_loss": [1.4172, 0.1232, 0.0872, 0.0688, 0.0554, 0.0494],
            "val_loss":   [0.3788, 0.3180, 0.2410, 0.2577, 0.2224, 0.2255],
            "val_wer":    [15.93,  13.43,  10.02,  10.11,  8.96,   8.82],
            "test_greedy":  [1.02, 1.53, 1.87, 2.79, 5.34],
            "test_grammar": [0.71, 1.10, 1.42, 2.06, 3.95],
            "color": "#2ca02c",
            "marker": "D",
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.3), constrained_layout=True)

    # Panel 1: Validation & Training CTC Loss across epochs (full dataset, 30-min budget)
    ax1 = axes[0]
    for label, d in p_full.items():
        ax1.plot(d["epochs"], d["val_loss"], marker=d["marker"], color=d["color"],
                 linewidth=2.2, markersize=5.5, label=f"{label} (val)")
        ax1.plot(d["epochs"], d["train_loss"], linestyle="--", color=d["color"],
                 linewidth=1.3, alpha=0.65)
    ax1.set_title("Model P (Full Dataset): CTC Loss (Solid=Val, Dashed=Train)", fontsize=11.0, fontweight="bold")
    ax1.set_xlabel("Epoch (Full GRID Training Split, 30-min Budget)", fontsize=10.0)
    ax1.set_ylabel("CTC Loss", fontsize=10.0)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(frameon=True, fontsize=8.3)

    # Panel 2: Validation WER Trajectory across epochs
    ax2 = axes[1]
    for label, d in p_full.items():
        ax2.plot(d["epochs"], d["val_wer"], marker=d["marker"], color=d["color"],
                 linewidth=2.2, markersize=5.5, label=f"{label} (best {min(d['val_wer']):.2f}%)")
    ax2.set_title("Model P (Full Dataset): Validation WER (%) by Epoch", fontsize=11.0, fontweight="bold")
    ax2.set_xlabel("Epoch (Full GRID Training Split, 30-min Budget)", fontsize=10.0)
    ax2.set_ylabel("Mean Validation WER (%)", fontsize=10.0)
    ax2.set_ylim(0, 70)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(frameon=True, fontsize=8.3)

    # Panel 3: Held-Out Test WER (%) across SNR (Greedy & Grammar)
    ax3 = axes[2]
    for label, d in p_full.items():
        short_label = label.split(" (")[0]
        ax3.plot(snr_labels, d["test_greedy"], marker=d["marker"], color=d["color"],
                 linestyle="-", linewidth=2.2, markersize=5.5, label=f"{short_label} (Greedy)")
        ax3.plot(snr_labels, d["test_grammar"], marker=d["marker"], color=d["color"],
                 linestyle="--", linewidth=1.7, markersize=4.5, alpha=0.85, label=f"{short_label} (Grammar)")
    ax3.set_title("Model P: Held-Out Test WER (%) Across SNR", fontsize=11.0, fontweight="bold")
    ax3.set_xlabel("Acoustic SNR Condition", fontsize=10.0)
    ax3.set_ylabel("Held-Out Test WER (%)", fontsize=10.0)
    ax3.set_ylim(0, 22)
    ax3.grid(True, linestyle="--", alpha=0.45)
    ax3.legend(frameon=True, fontsize=7.6, ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_all_models_comparison(out_path: Path) -> None:
    """3-panel figure comparing B1, B2 v1, B2 v2, B3 v1, B3 v2, and all 4 full-dataset Model P variants."""
    snr_labels = ["Clean", "10 dB", "5 dB", "0 dB", "-5 dB"]

    # Held-Out Test WER (%) by SNR (Greedy & Top Grammar)
    test_snr_models = [
        ("P4: Init B1+B2v2 (Grammar)",   [0.71, 1.10, 1.42, 2.06, 3.95],   "#2ca02c", "*", "--"),
        ("B3 Concat v2 (Grammar)",       [0.77, 1.13, 1.38, 1.99, 3.83],   "#d62728", "D", "--"),
        ("P4: Init B1+B2v2 (Greedy)",    [1.02, 1.53, 1.87, 2.79, 5.34],   "#2ca02c", "D", "-"),
        ("B3 Concat v2 (Greedy)",        [1.16, 1.66, 1.98, 2.81, 5.21],   "#d62728", "D", "-"),
        ("P3: Init B1+B2v1 (Greedy)",    [2.13, 2.67, 3.27, 4.33, 7.30],   "#17becf", "P", "-"),
        ("B3 Concat v1 (Greedy)",        [2.41, 2.88, 3.36, 4.29, 7.35],   "#9467bd", "^", "-"),
        ("B1 Audio-Only (Greedy)",       [1.96, 3.02, 3.79, 5.44, 9.45],   "#1f77b4", "o", "-"),
        ("P1: Scratch (Greedy)",         [12.58, 13.05, 13.55, 14.49, 16.28], "#7f7f7f", "o", "-."),
        ("P2: Init B2v1 (Greedy)",       [18.19, 18.20, 18.53, 18.93, 19.85], "#8c564b", "v", "-."),
        ("B2 Visual-Only v1 (Greedy)",   [23.04, 23.04, 23.04, 23.04, 23.04], "#ff7f0e", "s", "-"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8), constrained_layout=True)

    # Panel 1: Official Held-Out Test WER (%) across SNR (All Full Models)
    ax1 = axes[0]
    for label, vals, color, marker, ls in test_snr_models:
        ax1.plot(snr_labels, vals, marker=marker, color=color, linestyle=ls,
                 linewidth=2.0, markersize=5.5, label=label)
    ax1.set_title("Held-Out Test WER (%) Across SNR (All Models)", fontsize=11.2, fontweight="bold")
    ax1.set_xlabel("Noise Condition (SNR)", fontsize=10.2)
    ax1.set_ylabel("Test WER (%)", fontsize=10.2)
    ax1.set_ylim(0, 25)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(frameon=True, fontsize=7.6, loc="upper left")

    # Panel 2: Missing-Modality Robustness at Test Time (Drop Audio vs Drop Visual)
    ax2 = axes[1]
    ablation_models = [
        "P4 (B1+B2v2)",
        "P3 (B1+B2v1)",
        "P2 (B2v1)",
        "P1 (Scratch)",
        "B3 v2 (Concat)",
    ]
    bimodal_clean = [1.02, 2.13, 18.19, 12.58, 1.16]
    no_visual_clean = [1.88, 2.72, 28.82, 32.41, 25.18]
    no_audio_clean = [31.57, 25.65, 23.78, 25.03, 75.31]

    x_ab = np.arange(len(ablation_models))
    w_ab = 0.26
    ax2.bar(x_ab - w_ab, bimodal_clean, w_ab, label="Bimodal (Clean)", color="#2ca02c", edgecolor="black", linewidth=0.6)
    ax2.bar(x_ab, no_visual_clean, w_ab, label="No-Visual (-V, Clean)", color="#1f77b4", edgecolor="black", linewidth=0.6)
    ax2.bar(x_ab + w_ab, no_audio_clean, w_ab, label="No-Audio (-A, All SNR)", color="#ff7f0e", edgecolor="black", linewidth=0.6)
    ax2.set_title("Missing-Modality Robustness (e_miss vs Concat)", fontsize=11.2, fontweight="bold")
    ax2.set_xticks(x_ab)
    ax2.set_xticklabels(ablation_models, fontsize=8.5)
    ax2.set_ylabel("Held-Out Test WER (%)", fontsize=10.2)
    ax2.set_ylim(0, 85)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax2.legend(frameon=True, fontsize=8.5)

    # Panel 3: Summary Bar Chart of Mean Test WER (Greedy & Grammar) across all 9 full models
    ax3 = axes[2]
    names = [
        "P4\nB1+B2v2",
        "B3 v2\nConcat",
        "P3\nB1+B2v1",
        "B3 v1\nConcat",
        "B1\nAudio",
        "P1\nScratch",
        "P2\nInit B2",
        "B2 v1\nVisual",
        "B2 v2\nVisual",
    ]
    mean_test_greedy = [2.51, 2.56, 3.94, 4.06, 4.73, 13.99, 18.74, 23.04, 32.20]
    mean_test_grammar = [1.85, 1.82, 2.61, 3.22, 4.26, 10.80, 15.94, 21.03, 28.26]

    x = np.arange(len(names))
    width = 0.38
    bars1 = ax3.bar(x - width / 2, mean_test_greedy, width, label="Mean Test WER - Greedy (%)", color="#1f77b4", edgecolor="black", linewidth=0.6)
    bars2 = ax3.bar(x + width / 2, mean_test_grammar, width, label="Mean Test WER - Grammar (%)", color="#2ca02c", edgecolor="black", linewidth=0.6)

    for b in bars1:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    ax3.set_title("Mean Test WER Across All 5 SNRs (All 9 Full Models)", fontsize=11.2, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=8.0)
    ax3.set_ylabel("Mean Test WER (%)", fontsize=10.2)
    ax3.set_ylim(0, 38)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax3.legend(frameon=True, fontsize=8.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_week3_snr_figure(out_path: Path) -> None:
    """Clean log-scale SNR curve for the Week 3 report (documentation/5_submissions/week3_report/figures/snr_comparison_curves.png)."""
    snr_labels = ["Clean", "10 dB", "5 dB", "0 dB", "\u22125 dB"]
    x = np.arange(len(snr_labels))

    series = [
        ("B2 visual-only (v1)",          [23.04, 23.04, 23.04, 23.04, 23.04], "#e05a33", "-",  2.4),
        ("B1 audio-only",                [1.96,  3.02,  3.79,  5.44,  9.45],  "#2b6fd6", "-",  2.4),
        ("B3 concat (v1)",               [2.41,  2.88,  3.36,  4.29,  7.35],  "#16a36e", "-",  2.4),
        ("B3 concat (v2)",               [1.16,  1.66,  1.98,  2.81,  5.21],  "#e69900", "-",  2.4),
        ("Model P4 (B1+B2v2, greedy)",   [1.02,  1.53,  1.87,  2.79,  5.34],  "#7b2cbf", "-",  2.4),
        ("Model P4 + grammar decoding",  [0.71,  1.10,  1.42,  2.06,  3.95],  "#7b2cbf", "--", 2.2),
        ("B3 v2 + grammar decoding",     [0.77,  1.13,  1.38,  1.99,  3.83],  "#e69900", "--", 2.2),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for label, vals, color, ls, lw in series:
        ax.plot(x, vals, marker="o", markersize=5.5, markeredgecolor="white",
                markeredgewidth=1.0, color=color, linestyle=ls, linewidth=lw, label=label)

    # Annotate right-hand endpoint values cleanly
    annotations = [
        (23.04, "23.04", 0.0),
        (9.45,  "9.45",  0.0),
        (7.35,  "7.35",  0.0),
        (5.34,  "5.34 (P4)", 0.35),
        (5.21,  "5.21 (B3v2)", -0.35),
        (3.95,  "3.95 / 3.83", 0.0),
    ]
    for y_val, text, dy in annotations:
        ax.text(len(snr_labels) - 1 + 0.09, y_val + dy, text, va="center", ha="left",
                fontsize=9.2, color="#222222")

    ax.set_yscale("log")
    ax.set_ylim(0.5, 40)
    yticks = [0.5, 1, 2, 5, 10, 20, 40]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks], fontsize=10, color="#555555")
    ax.yaxis.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels, fontsize=10.5, color="#555555")
    ax.set_xlim(-0.22, len(snr_labels) - 1 + 0.72)
    ax.set_xlabel("Test SNR", fontsize=11, color="#555555", labelpad=6)
    ax.set_ylabel("Test WER (%, log scale)", fontsize=11, color="#555555", labelpad=6)
    ax.grid(True, axis="y", color="#dcdcdc", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b0b0b0")
    ax.spines["bottom"].set_color("#b0b0b0")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gate_vs_snr(out_path: Path) -> None:
    """2-panel figure showing Model P fusion gate g_t (audio weight) vs (1 - g_t) (visual weight) across SNR."""
    snr_labels = ["Clean", "10 dB", "5 dB", "0 dB", "\u22125 dB"]
    x = np.arange(len(snr_labels))

    p4_audio_g = np.array([0.8043, 0.7856, 0.7689, 0.7401, 0.6912]) * 100.0
    p4_vis_g = 100.0 - p4_audio_g

    p3_audio_g = np.array([0.8840, 0.8758, 0.8677, 0.8550, 0.8339]) * 100.0
    p3_vis_g = 100.0 - p3_audio_g

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)

    # Panel 1: Mean Fusion Gate g_bar (Audio Weight %) vs SNR
    ax1 = axes[0]
    ax1.plot(snr_labels, p4_audio_g, marker="D", markersize=7, linewidth=2.5, color="#2ca02c",
             label="P4: B1 + B2 v2 (ResNet-18) — Audio Gate g (\u0394 = \u221211.3%)")
    ax1.plot(snr_labels, p3_audio_g, marker="^", markersize=7, linewidth=2.5, color="#1f77b4",
             label="P3: B1 + B2 v1 (Conv3D) — Audio Gate g (\u0394 = \u22125.0%)")
    for i, (v4, v3) in enumerate(zip(p4_audio_g, p3_audio_g)):
        ax1.text(i, v4 - 2.3, f"{v4:.1f}%", ha="center", va="top", fontsize=8.8, color="#1b661b", fontweight="bold")
        ax1.text(i, v3 + 1.4, f"{v3:.1f}%", ha="center", va="bottom", fontsize=8.8, color="#15507a", fontweight="bold")
    ax1.set_title("Mean Audio Gate Weight g\u0304 (%) Across Noise Levels", fontsize=11.2, fontweight="bold")
    ax1.set_xlabel("Acoustic SNR Condition", fontsize=10.2)
    ax1.set_ylabel("Audio Stream Weight g\u0304 (%)", fontsize=10.2)
    ax1.set_ylim(60, 96)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(frameon=True, fontsize=8.8, loc="lower left")

    # Panel 2: Stacked Audio vs Visual Reliance in Model P4 (ResNet-18)
    ax2 = axes[1]
    width = 0.52
    ax2.bar(x, p4_audio_g, width, label="Audio Stream Weight (g\u0304)", color="#2b6fd6", edgecolor="black", linewidth=0.6)
    ax2.bar(x, p4_vis_g, width, bottom=p4_audio_g, label="Visual Lip Stream Weight (1 \u2212 g\u0304)", color="#ff7f0e", edgecolor="black", linewidth=0.6)
    for i, (va, vv) in enumerate(zip(p4_audio_g, p4_vis_g)):
        ax2.text(i, va / 2, f"Audio\n{va:.1f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        ax2.text(i, va + vv / 2, f"Lips\n{vv:.1f}%", ha="center", va="center", fontsize=8.5, color="black", fontweight="bold")
    ax2.set_title("Model P4: Dynamic Shift from Audio to Lips Under Noise", fontsize=11.2, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(snr_labels, fontsize=10.0)
    ax2.set_xlabel("Acoustic SNR Condition", fontsize=10.2)
    ax2.set_ylabel("Modality Contribution (%)", fontsize=10.2)
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=True, fontsize=8.8, loc="lower left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    p1 = REPORTS_DIR / "model_p_4way_curriculum.png"
    p2 = REPORTS_DIR / "all_models_snr_and_summary.png"
    p3 = REPO_ROOT / "documentation" / "5_submissions" / "week3_report" / "figures" / "snr_comparison_curves.png"
    p4 = REPORTS_DIR / "model_p_gate_vs_snr.png"
    plot_model_p_4way(p1)
    plot_all_models_comparison(p2)
    plot_week3_snr_figure(p3)
    plot_gate_vs_snr(p4)
    if ARTIFACT_DIR.exists():
        for p in (p1, p2, p3, p4):
            shutil.copy2(p, ARTIFACT_DIR / p.name)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Saved: {p4}")


if __name__ == "__main__":
    main()

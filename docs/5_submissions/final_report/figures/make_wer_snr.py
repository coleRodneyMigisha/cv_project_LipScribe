"""Figure: WER vs SNR, audio-only vs audio + lips, one panel per source.

Numbers are copied from their source files (do not edit by hand):
  GRID  (greedy):       analysis/01_grid/tables/test_wer_greedy.md (B1, P4)
  TED, films (beam+LM): reports/flix_v5/final_decoder.md, as in Table tab:realworld (twin_v5, ws_v5)
Run from the repository root:
  python docs/5_submissions/final_report/figures/make_wer_snr.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO, LIPS = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
INK, MUTED = "#0b0b0b", "#52514e"

panels = [
    ("GRID (greedy)", ["clean", "10", "5", "0", "−5"],
     ("B1", [1.96, 3.02, 3.79, 5.44, 9.45]), ("P4", [1.02, 1.53, 1.87, 2.79, 5.34])),
    ("TED (beam + LM)", ["clean", "0", "−5", "−10"],
     ("twin_v5", [15.7, 21.5, 29.6, 42.3]), ("ws_v5", [16.9, 20.5, 24.7, 28.5])),
    ("Films (beam + LM)", ["clean", "0", "−5", "−10"],
     ("twin_v5", [27.7, 39.8, 53.1, 64.5]), ("ws_v5", [27.3, 34.8, 41.8, 49.2])),
]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
for ax, (title, xs, (na, ya), (nl, yl)) in zip(axes, panels):
    x = range(len(xs))
    ax.plot(x, ya, color=AUDIO, lw=2, ls="--", marker="o", ms=4, label=f"{na} (audio only)")
    ax.plot(x, yl, color=LIPS, lw=2, marker="s", ms=4, label=f"{nl} (audio + lips)")
    ax.set_xticks(list(x)); ax.set_xticklabels(xs)
    ax.set_title(title, color=INK, fontsize=8.5)
    ax.set_xlabel("SNR (dB)")
    ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left" if title.startswith("GRID") else "lower right")
axes[0].set_ylabel("Test WER (%)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "wer_vs_snr.pdf"))
fig.savefig(os.path.join(HERE, "wer_vs_snr.png"), dpi=200)

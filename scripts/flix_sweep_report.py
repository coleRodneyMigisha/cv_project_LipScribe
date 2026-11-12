"""X6 'Flix': report for scripts/flix_sweep.py -> <out>/flix_sweep_report.md + flix_sweep_wer.png.

Compares, on the same 43 Flix test clips: the original P4 copy (zero-shot), the first
fine-tune (reports/flix, P4 + GRID mix), and the sweep's best configuration (mean and
spread over 3 seeds, Flix-only training).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.flix_report import new_words

SETS = [("flix", "Films: lips + sound"), ("flix_lips_only", "Films: lips only"),
        ("flix_sound_only", "Films: sound only"), ("grid", "GRID test")]
C = ["#2a78d6", "#eb6834", "#1baf7a"]      # validated default categorical slots 1-3
INK, INK2, SURF = "#1f1f1e", "#5c5b55", "#fcfcfb"


def res(d: Path, tag: str):
    p = d / f"{tag}.json"
    return json.loads(p.read_text())["results"] if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/flix_sweep"))
    ap.add_argument("--first", type=Path, default=Path("reports/flix"))
    ap.add_argument("--stats", type=Path, default=Path("data/flix_only/stats.json"))
    a = ap.parse_args()
    with open(a.out / "sweep.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    best = json.loads((a.out / "best.json").read_text())
    stats = json.loads(a.stats.read_text())
    gw = set(stats.get("grid_words", []))
    seeds = sorted({r["seed"] for r in rows if r["trial"] == best["trial"]})

    models = {"Original P4 (no film training)": [(a.first, "base")],
              "First fine-tune (P4 + GRID mix)": [(a.first, "tuned")],
              f"Retrained on Flix, best of sweep ({len(seeds)} seeds, {best.get('final_epochs', '?')} epochs)": [(a.out, f"sweep_s{s}") for s in seeds]}
    table, chart = {}, {}
    for mname, srcs in models.items():
        for s, _ in SETS:
            vals = [res(d, f"{t}_{s}") for d, t in srcs]
            vals = [v for v in vals if v]
            if not vals:
                continue
            w = [100 * v["clean"]["wer"] for v in vals]; c = [100 * v["clean"]["cer"] for v in vals]
            table[(mname, s)] = (w, c)
            chart.setdefault(mname, {})[s] = st.mean(w)
        nw = [new_words(d, f"{t}_flix", gw) for d, t in srcs]
        nw = [x for x in nw if x]
        if nw:
            table[(mname, "new")] = nw

    fmt = lambda xs: f"{st.mean(xs):.1f}" + (f" ± {st.stdev(xs):.1f}" if len(xs) > 1 else "")
    L = ["# Flix: Model P retrained on film clips, with a hyperparameter sweep", "",
         f"Data: {stats['flix_clips']} clips ({stats['minutes']} min, {stats['distinct_words']} distinct words). "
         f"Train {stats['by_partition']['train']} / val {stats['by_partition']['val']} / "
         f"test {stats['by_partition']['test']} clips; titles never shared between splits "
         f"(val: {', '.join(stats['titles_by_partition']['val'])}; test: {', '.join(stats['titles_by_partition']['test'])}).",
         "Training uses Flix only (no GRID). Runs are chosen on validation CER (letters wrong), "
         "because WER stays near 100% on film speech. The test titles were scored once, at the end.", "",
         "## Test results (clean sound; WER % / CER %)", "",
         "| Model | " + " | ".join(n for _, n in SETS) + " | Non-GRID words right |",
         "|---|" + "---:|" * (len(SETS) + 1)]
    for mname in models:
        cells = []
        for s, _ in SETS:
            v = table.get((mname, s))
            cells.append(f"{fmt(v[0])} / {fmt(v[1])}" if v else "–")
        nw = table.get((mname, "new"))
        cells.append(" / ".join(f"{x['tokens_right']}" for x in nw) + f" of {nw[0]['tokens']}" if nw else "–")
        L.append(f"| {mname} | " + " | ".join(cells) + " |")
    nwb = table.get((list(models)[-1], "new"))
    if nwb:
        words = sorted({w for x in nwb for w in x["distinct_right"]})
        L += ["", f"Distinct non-GRID words the retrained model wrote correctly (any seed): {', '.join(words) or 'none'}"]

    L += ["", "![chart](flix_sweep_wer.png)", "", f"## Sweep ({len([r for r in rows if r['seed'] == '1' and not r['trial'].startswith('final')])} configurations, seed 1, sorted by val CER)", "",
          "| Trial | init | lr | freeze | dropout | wd | audio drop | batch | noise | best epoch | val CER % | val WER % |",
          "|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    ok = sorted((r for r in rows if r["seed"] == "1" and r["val_cer"] != "nan"
                 and not r["trial"].startswith("final")), key=lambda r: float(r["val_cer"]))
    for r in ok:
        L.append(f"| {r['trial']} | {r['init']} | {float(r['lr']):g} | {r['freeze']} | {r['dropout']} | "
                 f"{float(r['weight_decay']):g} | {r['audio_dropout']} | {r['batch_size']} | {r['snrs']} | "
                 f"{r['best_epoch']}/{r['epochs']} | {100 * float(r['val_cer']):.1f} | {100 * float(r['val_wer']):.1f} |")
    failed = [r["trial"] for r in rows if r["val_cer"] == "nan"]
    if failed:
        L.append(f"\nFailed runs: {', '.join(failed)} (logs in `logs/`).")
    b = [r for r in rows if r["trial"] == best["trial"]]
    L += ["", f"Best configuration **{best.get('from_trial', best['trial'])}**: `{best['hparams']}`; "
          f"final {best.get('final_epochs', '?')}-epoch runs, val CER over seeds: "
          + ", ".join(f"{100 * float(r['val_cer']):.1f}%" for r in b),
          "", "Anchors: `p4_recipe` = the P4 recipe (B1 + B2-v2 encoders, new fusion and CTC head) on Flix; "
          "`from_p4` = start from the whole P4. `noise: mild` = training SNRs {clean, clean, 10, 5}.", ""]
    (a.out / "flix_sweep_report.md").write_text("\n".join(L))
    print("\n".join(L))

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200, facecolor=SURF); ax.set_facecolor(SURF)
    keys = [s for s, _ in SETS]; w = 0.26
    for i, (mname, vals) in enumerate(chart.items()):
        xs = [j + (i - 1) * (w + 0.02) for j, k in enumerate(keys) if k in vals]
        ys = [vals[k] for k in keys if k in vals]
        bars = ax.bar(xs, ys, width=w, color=C[i], label=mname, zorder=2)
        for bb, y in zip(bars, ys):
            ax.text(bb.get_x() + bb.get_width() / 2, y + 1.5, f"{y:.0f}%" if y >= 10 else f"{y:.1f}%",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(keys)), [n for _, n in SETS], fontsize=9.5, color=INK)
    ax.set_ylabel("Words wrong (WER %, clean) — lower is better", fontsize=9, color=INK2)
    ax.grid(axis="y", color="#e4e3df", lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2"); ax.tick_params(length=0, colors=INK2, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK)
    fig.tight_layout(); fig.savefig(a.out / "flix_sweep_wer.png", facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    main()

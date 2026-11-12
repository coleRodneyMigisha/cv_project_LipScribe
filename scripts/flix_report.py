"""X6 'Flix': collect the flix_eval JSONs into flix_report.md + flix_wer.png (+ results.json).

Usage: python -m scripts.flix_report --dir reports/flix --run runs/flix_finetune_p4_<stamp>
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETS = [("flix", "Films: lips + sound"), ("flix_lips_only", "Films: lips only"),
        ("flix_sound_only", "Films: sound only"), ("grid", "GRID test (unchanged task)")]
MODELS = [("base", "Original P4 (copy)"), ("tuned", "Fine-tuned on Flix")]
# validated default categorical slots 1-2 (dataviz palette), text in neutral ink
C_SERIES = ["#2a78d6", "#eb6834"]
INK, INK2, SURF = "#1f1f1e", "#5c5b55", "#fcfcfb"


def load(d: Path, tag: str) -> dict | None:
    p = d / f"{tag}.json"
    return json.loads(p.read_text())["results"] if p.exists() else None


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def chart(d: Path, res: dict) -> None:
    labels = [name for _, name in SETS]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    w, gap = 0.36, 0.02
    for i, (m, mname) in enumerate(MODELS):
        xs, ys = [], []
        for j, (s, _) in enumerate(SETS):
            r = res.get(f"{m}_{s}")
            if r and "clean" in r:
                xs.append(j + (i - 0.5) * (w + gap)); ys.append(100 * r["clean"]["wer"])
        if not ys:
            continue
        bars = ax.bar(xs, ys, width=w, color=C_SERIES[i], label=mname, zorder=2)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 1.5, f"{y:.0f}%", ha="center",
                    va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(SETS)), labels, fontsize=9.5, color=INK)
    top = max([100] + [100 * r["clean"]["wer"] for r in res.values() if r and "clean" in r])
    ax.set_ylim(0, top * 1.12)
    ax.grid(axis="y", color="#e4e3df", lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(axis="y", colors=INK2, labelsize=8.5, length=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, labelcolor=INK)
    ax.set_ylabel("Words wrong (WER %, clean sound) — lower is better", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(d / "flix_wer.png", facecolor=SURF)
    plt.close(fig)


def examples(d: Path, n: int = 10) -> list[tuple[str, str, str]]:
    def read(tag):
        p = d / f"{tag}_hyps.csv"
        if not p.exists():
            return {}
        with open(p, newline="") as fh:
            return {r["utt_id"]: r for r in csv.DictReader(fh) if r["snr"] == "clean"}
    base, tuned = read("base_flix"), read("tuned_flix")
    ids = sorted(set(base) & set(tuned), key=lambda u: float(tuned[u]["wer"]))
    pick = ids[: n // 2] + ids[len(ids) // 2: len(ids) // 2 + n // 2]
    return [(base[u]["ref"], base[u]["hyp"], tuned[u]["hyp"]) for u in pick]


def new_words(d: Path, tag: str, grid_words: set[str]) -> dict | None:
    """Film words outside GRID's vocabulary that the model wrote correctly (clean sound).

    Bag-of-words match per clip: a reference word counts once for each time it also
    appears in the hypothesis, so word order and extra words are ignored.
    """
    p = d / f"{tag}_hyps.csv"
    if not p.exists():
        return None
    hit_tokens = total = 0
    hit_types: set[str] = set()
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["snr"] != "clean":
                continue
            hyp = Counter(r["hyp"].split())
            for w in r["ref"].split():
                if w in grid_words:
                    continue
                total += 1
                if hyp[w] > 0:
                    hyp[w] -= 1; hit_tokens += 1; hit_types.add(w)
    return {"tokens_right": hit_tokens, "tokens": total, "distinct_right": sorted(hit_types)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("reports/flix"))
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--stats", type=Path, default=Path("data/flix/stats.json"))
    a = ap.parse_args()
    res = {f"{m}_{s}": load(a.dir, f"{m}_{s}") for m, _ in MODELS for s, _ in SETS}
    stats = json.loads(a.stats.read_text()) if a.stats.exists() else {}
    (a.dir / "results.json").write_text(json.dumps({"results": res, "stats": stats}, indent=2))
    chart(a.dir, res)

    snrs = ["clean", "0"]
    L = ["# Flix test: Model P4 on film and TV clips", "",
         "Mouth clips mined from local films/series (`scripts/mine_flix.py`): one frontal face, "
         "lips moving, no camera cut, 0.8–3 s subtitle cue. Labels are the subtitle text, "
         "which is often paraphrased, so part of every error rate comes from the labels. "
         "Titles are split so no film appears in more than one of train/val/test.", ""]
    if stats:
        bp = stats["by_partition"]
        L += ["## Data", "",
              f"- Clips kept: **{stats['flix_clips']}** ({stats['minutes']} min) — train {bp['train']}, "
              f"val {bp['val']}, test {bp['test']}; dropped {stats['dropped']}",
              f"- Test titles: {', '.join(stats['titles_by_partition']['test'])}",
              f"- Distinct words: {stats['distinct_words']} (GRID knows {stats['grid_vocab']}; "
              f"{stats['distinct_words_in_grid']} shared)",
              f"- Test words also seen in Flix training: {stats['test_tokens_seen_in_flix_train']}"
              f" / {stats['test_tokens']}",
              f"- Fine-tuning mix: {stats['train_mix_rows']} rows (Flix train ×8 + a GRID subset so GRID is not forgotten)",
              ""]
    L += ["## Results (greedy decoding, WER % / CER %)", "",
          "| Test set | Model | " + " | ".join(f"{s} dB" if s != "clean" else "clean" for s in snrs) + " |",
          "|---|---|" + "---:|" * len(snrs)]
    for s, sname in SETS:
        for m, mname in MODELS:
            r = res.get(f"{m}_{s}")
            if r is None:
                continue
            cells = [f"{pct(r[x]['wer'])} / {pct(r[x]['cer'])}" if x in r else "–" for x in snrs]
            L.append(f"| {sname} | {mname} | " + " | ".join(cells) + " |")
    L += ["", "![WER chart](flix_wer.png)", "",
          "*Lips only* = sound silenced at test time; *sound only* = video blanked. "
          "Comparing them shows which input the model relies on. The GRID row checks "
          "that fine-tuning did not break the original task.", ""]
    gw = set(stats.get("grid_words", []))
    nw = {f"{m}_{s}": new_words(a.dir, f"{m}_{s}", gw) for m, _ in MODELS for s in ("flix", "flix_lips_only")}
    if gw and any(nw.values()):
        L += ["## New words (not in GRID's vocabulary) written correctly on Flix test, clean sound", "",
              "| Model | Lips + sound | Lips only | Distinct new words right (lips + sound) |", "|---|---:|---:|---|"]
        for m, mname in MODELS:
            a_, v_ = nw.get(f"{m}_flix"), nw.get(f"{m}_flix_lips_only")
            if not a_:
                continue
            cell = lambda x: f"{x['tokens_right']} / {x['tokens']}" if x else "–"
            L.append(f"| {mname} | {cell(a_)} | {cell(v_)} | {', '.join(a_['distinct_right'][:40]) or '–'} |")
        L.append("")
        (a.dir / "new_words.json").write_text(json.dumps(nw, indent=2))
    ex = examples(a.dir)
    if ex:
        L += ["## Example transcripts (Flix test, clean sound)", "",
              "| Subtitle | Original P4 | Fine-tuned |", "|---|---|---|"]
        L += [f"| {r} | {b or '∅'} | {t or '∅'} |" for r, b, t in ex]
        L.append("")
    if a.run:
        L += [f"Fine-tune run: `{a.run}` (best checkpoint copied to `checkpoints/flix_finetuned_p4.pt`). "
              "The main model `../cv_project_LipScribe/checkpoints/p_proposed_best.pt` was only read; "
              "see `main_model_md5.txt`.", ""]
    (a.dir / "flix_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

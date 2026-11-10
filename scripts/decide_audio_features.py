"""Apply the W3-08 (Experiment X5) decision rule once all three B1 feature
configs have been trained and evaluated.

Reads ``experiments/results.csv`` (appended to by ``scripts/train.py`` and
``scripts/evaluate.py``), takes the most recent row for each of
``b1_audio_only`` (A1), ``b1_mfcc39`` (A2) and ``b1_logmel_mfcc`` (A3), and
applies the rule from the card: keep A1 unless another configuration strictly
improves mean validation WER.

Usage::

    python -m scripts.decide_audio_features
    python -m scripts.decide_audio_features --out reports/w3_08_audio_features_decision.md
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "experiments" / "results.csv"
DEFAULT_OUT = REPO_ROOT / "reports" / "w3_08_audio_features_decision.md"

CONFIGS = {
    "b1_audio_only": "A1 (80 log-Mel)",
    "b1_mfcc39": "A2 (13 MFCC + delta + delta-delta)",
    "b1_logmel_mfcc": "A3 (log-Mel + MFCC, 119 channels)",
}
BASELINE = "b1_audio_only"


def latest_rows(results_csv: Path) -> dict[str, dict]:
    """Most recent row per experiment name, keyed by ``experiment``."""
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    latest: dict[str, dict] = {}
    for row in rows:
        if row.get("experiment") in CONFIGS:
            latest[row["experiment"]] = row  # last one wins; rows are append-only
    return latest


def decide(latest: dict[str, dict]) -> tuple[str, str]:
    """-> (decision text, markdown report)."""
    missing = [name for name in CONFIGS if name not in latest]
    lines = [
        "# W3-08 (Experiment X5): audio feature configuration decision",
        "",
        "Rule (card W3-08): keep A1 unless another configuration improves validation WER.",
        "",
        "| Experiment | Features | Mean val WER | Mean val CER | Run |",
        "|---|---|---:|---:|---|",
    ]
    for name, label in CONFIGS.items():
        row = latest.get(name)
        if row is None:
            lines.append(f"| `{name}` | {label} | — (not yet run) | — | — |")
            continue
        lines.append(
            f"| `{name}` | {label} | {float(row['best_val_wer_mean']):.4f} | "
            f"{float(row['best_val_cer_mean']):.4f} | `{Path(row['run_dir']).name}` |"
        )

    if missing:
        decision = (
            f"Decision pending: no results.csv row yet for {', '.join(missing)}. "
            "Train and evaluate the missing config(s), then re-run this script."
        )
        lines += ["", f"**{decision}**"]
        return decision, "\n".join(lines) + "\n"

    base_wer = float(latest[BASELINE]["best_val_wer_mean"])
    challengers = {name: float(latest[name]["best_val_wer_mean"])
                   for name in CONFIGS if name != BASELINE}
    best_name, best_wer = min(challengers.items(), key=lambda kv: kv[1])

    if best_wer < base_wer:
        decision = (
            f"Switch to {CONFIGS[best_name]} (`{best_name}`): mean val WER "
            f"{best_wer:.4f} improves on A1's {base_wer:.4f}."
        )
    else:
        decision = (
            f"Keep A1 (`{BASELINE}`): mean val WER {base_wer:.4f}; neither A2 nor A3 "
            f"improved on it (best challenger {best_name} = {best_wer:.4f})."
        )
    lines += ["", f"**Decision: {decision}**"]
    return decision, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.results_csv.exists():
        raise FileNotFoundError(f"{args.results_csv} not found; run training first")

    decision, report = decide(latest_rows(args.results_csv))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

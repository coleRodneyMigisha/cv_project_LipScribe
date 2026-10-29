"""Speaker-independent train/val/test split (card W1-08).

    python -m scripts.make_splits                     # reads reports/integrity_report.csv

Splits by speaker, never by utterance, so no test speaker is ever seen in
training. Speakers with fewer than half of their utterances passing the
integrity check are excluded (in our copy of GRID: s21, which has no video).
Test gets 4 speakers (2 female, 2 male), validation 3 (mixed gender), training
all the rest. Generated once with a fixed seed: the script refuses to overwrite
an existing ``data/splits.csv`` unless ``--force`` is given. Never regenerate it
after week 1.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_REPORT = Path("reports/integrity_report.csv")
DEFAULT_OUT = Path("data/splits.csv")
SEED = 2026
MIN_PASS_FRACTION = 0.5

# GRID speaker sexes: 18 male, 16 female (Cooke et al., 2006, GRID corpus speaker list).
FEMALE = {f"s{n}" for n in (4, 7, 11, 15, 16, 18, 20, 21, 22, 23, 24, 25, 29, 31, 33, 34)}
MALE = {f"s{n}" for n in (1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 17, 19, 26, 27, 28, 30, 32)}


def gender(speaker: str) -> str:
    if speaker in FEMALE:
        return "F"
    if speaker in MALE:
        return "M"
    raise ValueError(f"unknown GRID speaker {speaker!r}")


def usable_counts(report: Path) -> dict[str, tuple[int, int]]:
    """speaker -> (utterances passing the integrity check, utterances listed)."""
    counts: dict[str, list[int]] = {}
    with open(report, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = counts.setdefault(row["speaker"].lower(), [0, 0])
            c[0] += row["overall_pass"] == "True"
            c[1] += 1
    return {s: (p, t) for s, (p, t) in counts.items()}


def make_splits(counts: dict[str, tuple[int, int]], seed: int = SEED,
                min_pass_fraction: float = MIN_PASS_FRACTION) -> list[dict]:
    """-> rows ``speaker, partition, gender, usable_utterances`` (partition ``excluded`` for dropped speakers)."""
    usable = sorted((s for s, (p, t) in counts.items() if t and p / t >= min_pass_fraction),
                    key=lambda s: int(s[1:]))
    rng = random.Random(seed)
    female = [s for s in usable if gender(s) == "F"]
    male = [s for s in usable if gender(s) == "M"]
    rng.shuffle(female)
    rng.shuffle(male)
    if len(female) < 4 or len(male) < 3:
        raise ValueError("not enough usable speakers of each sex for test and validation")

    test = female[:2] + male[:2]
    # Validation: 3 speakers, at least one of each sex; which sex gets two is drawn too.
    val = female[2:4] + male[2:3] if rng.random() < 0.5 else female[2:3] + male[2:4]
    partition = {s: "test" for s in test} | {s: "val" for s in val}

    rows = []
    for s in sorted(counts, key=lambda s: int(s[1:])):
        part = partition.get(s, "train") if s in usable else "excluded"
        rows.append({"speaker": s, "partition": part, "gender": gender(s),
                     "usable_utterances": counts[s][0]})
    return rows


def check_disjoint(rows: list[dict]) -> None:
    seen: dict[str, str] = {}
    for r in rows:
        if r["speaker"] in seen and seen[r["speaker"]] != r["partition"]:
            raise ValueError(f"speaker {r['speaker']} is in {seen[r['speaker']]} and {r['partition']}")
        seen[r["speaker"]] = r["partition"]


def summary(rows: list[dict]) -> str:
    lines = []
    for part in ("train", "val", "test", "excluded"):
        members = [r for r in rows if r["partition"] == part]
        sexes = Counter(r["gender"] for r in members)
        utts = sum(int(r["usable_utterances"]) for r in members)
        lines.append(f"  {part:9s} {len(members):2d} speakers (F={sexes['F']}, M={sexes['M']}), "
                     f"{utts:5d} usable utterances: {' '.join(r['speaker'] for r in members)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true", help="overwrite an existing splits file")
    args = p.parse_args(argv)

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists and must not be regenerated (use --force only if the "
              f"whole team agrees).", file=sys.stderr)
        return 1
    rows = make_splits(usable_counts(args.report), seed=args.seed)
    check_disjoint(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "partition", "gender", "usable_utterances"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} (seed {args.seed}):\n{summary(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

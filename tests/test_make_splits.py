import csv
from collections import Counter
from pathlib import Path

import pytest

from scripts.make_splits import FEMALE, MALE, check_disjoint, gender, main, make_splits

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits.csv"
# Our copy of GRID: 34 speakers, s21 has no video, s8 lost 108 clips.
COUNTS = {f"s{n}": (1000, 1000) for n in range(1, 35)} | {"s21": (0, 1000), "s8": (892, 1000)}


def _by_partition(rows):
    parts = {}
    for r in rows:
        parts.setdefault(r["partition"], set()).add(r["speaker"])
    return parts


def test_grid_has_18_male_16_female():
    assert len(MALE) == 18 and len(FEMALE) == 16 and not MALE & FEMALE


def test_split_sizes_and_balance():
    rows = make_splits(COUNTS)
    parts = _by_partition(rows)
    assert parts["excluded"] == {"s21"} and "s8" in parts["train"] | parts["val"] | parts["test"]
    assert len(parts["test"]) == 4 and len(parts["val"]) == 3 and len(parts["train"]) == 26
    assert Counter(gender(s) for s in parts["test"]) == {"F": 2, "M": 2}
    assert set(Counter(gender(s) for s in parts["val"]).values()) == {1, 2}


def test_partitions_pairwise_disjoint():
    parts = _by_partition(make_splits(COUNTS))
    assert not parts["train"] & parts["val"]
    assert not parts["train"] & parts["test"]
    assert not parts["val"] & parts["test"]
    with pytest.raises(ValueError):
        check_disjoint([{"speaker": "s1", "partition": "train"}, {"speaker": "s1", "partition": "test"}])


def test_fixed_seed_is_reproducible():
    assert make_splits(COUNTS) == make_splits(COUNTS)


def test_refuses_to_overwrite(tmp_path):
    out = tmp_path / "splits.csv"
    out.write_text("keep me")
    assert main(["--out", str(out)]) == 1
    assert out.read_text() == "keep me"


@pytest.mark.skipif(not SPLITS.exists(), reason="data/splits.csv not generated")
def test_committed_splits_are_disjoint_and_balanced():
    with open(SPLITS, newline="") as f:
        rows = list(csv.DictReader(f))
    check_disjoint(rows)
    assert len({r["speaker"] for r in rows}) == len(rows)
    parts = _by_partition(rows)
    assert Counter(gender(s) for s in parts["test"]) == {"F": 2, "M": 2}
    assert not (parts["train"] & parts["test"]) and not (parts["train"] & parts["val"])

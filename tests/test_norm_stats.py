from pathlib import Path

import numpy as np
import pytest

from lipscribe.preprocessing.features import normalise
from lipscribe.preprocessing.norm_stats import RunningMoments, load_norm_stats, split_hash

STATS_FILE = Path(__file__).resolve().parents[1] / "data" / "stats" / "feature_norm.npz"


def test_running_moments_matches_numpy():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(1000, 4))
    rm = RunningMoments(4)
    for chunk in np.array_split(x, 7):
        rm.update(chunk)
    np.testing.assert_allclose(rm.mean, x.mean(0), atol=1e-10)
    np.testing.assert_allclose(rm.std, x.std(0), atol=1e-10)


def test_heldout_features_are_standardised():
    """Stats fitted on one sample must standardise a different sample (ticket: within 0.05)."""
    rng = np.random.default_rng(1)
    scale = np.linspace(1, 4, 80)
    utts = [rng.normal(-4.0, 1.0, size=(400, 80)) * scale for _ in range(60)]
    fit, held = utts[:30], utts[30:]

    rm = RunningMoments(80)
    for u in fit:
        rm.update(u)
    z = normalise(np.concatenate(held), {"audio_mean": rm.mean, "audio_std": rm.std})
    assert np.abs(z.mean(axis=0)).max() < 0.05
    assert np.abs(z.std(axis=0) - 1).max() < 0.05


def test_split_hash_ignores_order():
    assert split_hash(["a", "b", "c"]) == split_hash(["c", "a", "b"])
    assert split_hash(["a", "b"]) != split_hash(["a", "c"])


@pytest.mark.skipif(not STATS_FILE.exists(), reason="stats file not generated yet")
def test_committed_stats_file_is_valid():
    s = load_norm_stats(STATS_FILE)
    assert s["audio_mean"].shape == s["audio_std"].shape == (80,)
    assert np.isfinite(s["audio_mean"]).all() and (s["audio_std"] > 0).all()
    assert 0.0 < float(s["pixel_mean"]) < 1.0 and float(s["pixel_std"]) > 0
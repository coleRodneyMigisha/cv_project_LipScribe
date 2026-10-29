"""W2-05: noise mixer with SNR verification."""
import math
import random

import numpy as np
import pytest

from lipscribe.preprocessing.noise import CLEAN, SNR_CONDITIONS, NoisePool, mix_noise


def _speech(n=48000, seed=0):
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal(n)).astype(np.float32)


@pytest.mark.parametrize("snr", SNR_CONDITIONS)
def test_all_five_conditions(snr):
    s = _speech()
    y, info = mix_noise(s, _speech(30000, seed=1), snr, rng=random.Random(0), return_info=True)
    assert y.shape == s.shape and y.dtype == np.float32
    assert np.max(np.abs(y)) <= 1.0
    if math.isinf(snr):
        assert np.array_equal(y, s)
    else:
        assert abs(info["measured_snr_db"] - snr) < 0.01
        noise_part = y / info["peak_scale"] - s
        measured = 10 * np.log10(np.mean(s.astype(np.float64) ** 2) / np.mean(noise_part.astype(np.float64) ** 2))
        assert abs(measured - snr) < 0.01


def test_clean_string_returns_original():
    s = _speech()
    assert np.array_equal(mix_noise(s, _speech(), "clean"), s)


def test_short_noise_is_looped():
    s = _speech(48000)
    y, info = mix_noise(s, _speech(1000, seed=2), 5.0, rng=random.Random(1), return_info=True)
    assert y.shape == s.shape
    assert 0 <= info["offset"] <= 48000 * 2


def test_loud_mixture_is_peak_normalised_and_keeps_snr():
    s = (0.9 * np.sign(_speech())).astype(np.float32)      # near full scale
    y, info = mix_noise(s, _speech(seed=3), -5.0, rng=random.Random(0), return_info=True)
    assert info["peak_scale"] < 1.0
    assert np.max(np.abs(y)) == pytest.approx(1.0, abs=1e-6)
    noise_part = y.astype(np.float64) / info["peak_scale"] - s
    assert abs(10 * np.log10(np.mean(s.astype(np.float64) ** 2) / np.mean(noise_part ** 2)) + 5) < 0.01


def test_seeded_rng_is_reproducible():
    s, w = _speech(), _speech(100000, seed=4)
    a = mix_noise(s, w, 0.0, rng=random.Random(42))
    b = mix_noise(s, w, 0.0, rng=random.Random(42))
    c = mix_noise(s, w, 0.0, rng=random.Random(43))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_silent_inputs_raise():
    with pytest.raises(ValueError):
        mix_noise(np.zeros(1000, np.float32), _speech(), 5.0)
    with pytest.raises(ValueError):
        mix_noise(_speech(), np.zeros(1000, np.float32), 5.0)


def test_pool_draws_files_and_reports_which():
    files = {"a.wav": _speech(20000, 5), "b.wav": _speech(20000, 6)}
    pool = NoisePool(list(files), loader=lambda p: files[str(p)])
    y, info = pool.mix(_speech(), 10.0, random.Random(0))
    assert info["noise_file"] in files and abs(info["measured_snr_db"] - 10) < 0.01
    y, info = pool.mix(_speech(), CLEAN, random.Random(0))
    assert info["noise_file"] == ""


def test_dataset_uses_this_mixer():
    from lipscribe.data.dataset import _default_mixer
    s, w = _speech(), _speech(60000, seed=7)
    mixer = _default_mixer()
    assert np.array_equal(mixer(s, w, 5.0, random.Random(3)), mix_noise(s, w, 5.0, rng=random.Random(3)))

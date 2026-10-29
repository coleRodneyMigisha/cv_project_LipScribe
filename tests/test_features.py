import librosa
import numpy as np
import pytest
import torch

from lipscribe.preprocessing.features import (
    FEATURE_DIMS, get_feature_fn, logmel, logmel_mfcc, mfcc39, normalise, rms_normalise,
)

SR = 16_000


def _clip(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    wave = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)
    return wave.astype(np.float32)


def _dbfs(wave):
    return 20 * np.log10(np.sqrt(np.mean(np.square(wave))))


def test_three_seconds_gives_300_by_80():
    feats = logmel(_clip())
    assert feats.shape == (300, 80) and feats.dtype == np.float32
    assert logmel(_clip(), n_frames=75).shape == (300, 80)


def test_matches_librosa():
    wave = _clip()
    ref = librosa.feature.melspectrogram(
        y=wave, sr=SR, n_fft=512, hop_length=160, win_length=400, window="hann",
        center=True, pad_mode="constant", power=2.0, n_mels=80, fmin=20, fmax=8000)
    ref = np.log(ref + 1e-6).T[:300]
    np.testing.assert_allclose(logmel(wave), ref, atol=1e-3)


def test_batched_tensor_matches_single_numpy():
    waves = np.stack([_clip(seed=0), _clip(seed=1)])
    batch = logmel(torch.from_numpy(waves))
    assert torch.is_tensor(batch) and batch.shape == (2, 300, 80)
    np.testing.assert_allclose(batch[1].numpy(), logmel(waves[1]), atol=1e-4)


def test_trims_and_pads_to_four_per_video_frame():
    assert logmel(_clip(), n_frames=70).shape == (280, 80)
    assert logmel(_clip(seconds=2.9), n_frames=75).shape == (300, 80)


@pytest.mark.parametrize("name", ["logmel80", "mfcc39", "logmel_mfcc"])
def test_config_switch_dimensions(name):
    feats = get_feature_fn(name)(_clip(), n_frames=75)
    assert feats.shape == (300, FEATURE_DIMS[name])
    assert np.isfinite(feats).all()


def test_logmel_mfcc_is_concatenation():
    wave = _clip()
    np.testing.assert_allclose(logmel_mfcc(wave), np.concatenate([logmel(wave), mfcc39(wave)], -1),
                               atol=1e-5)


def test_unknown_feature_name():
    with pytest.raises(ValueError):
        get_feature_fn("spectrogram")


def test_rms_normalise_to_minus_25_dbfs():
    assert _dbfs(rms_normalise(_clip())) == pytest.approx(-25.0, abs=1e-3)


def test_rms_gain_capped_at_20_db():
    quiet = _clip() * 1e-3                       # far below -45 dBFS
    out = rms_normalise(quiet)
    assert _dbfs(out) - _dbfs(quiet) == pytest.approx(20.0, abs=1e-3)


def test_normalise_with_stats():
    feats = logmel(np.stack([_clip(seed=s) for s in range(4)]))
    flat = feats.reshape(-1, 80)
    stats = {"audio_mean": flat.mean(0), "audio_std": flat.std(0)}
    out = normalise(feats, stats).reshape(-1, 80)
    assert np.abs(out.mean(0)).max() < 0.05 and np.abs(out.std(0) - 1).max() < 0.05
    out_t = normalise(torch.from_numpy(feats), stats)
    np.testing.assert_allclose(out_t.numpy().reshape(-1, 80), out, atol=1e-5)

"""Audio features shared by training and the app (card W2-07).

Feature configurations (config key ``model.audio_features``):

    logmel80     A1  80 log-Mel bands                         -> [T_a, 80]
    mfcc39       A2  13 MFCC + delta + delta-delta            -> [T_a, 39]
    logmel_mfcc  A3  log-Mel and MFCC39 concatenated          -> [T_a, 119]

Log-Mel: 25 ms Hann window, 10 ms hop, 512-point FFT, 80 Mel bands between
20 Hz and 8 kHz, ``log(power + 1e-6)``. Output is trimmed (or padded with
silence) to ``4 * n_frames`` feature frames, i.e. 300 for a 3 s, 75-frame clip.

Every function accepts one waveform ``[N]`` or a batch ``[B, N]``, as a NumPy
array or a torch tensor, and returns the same kind it was given. Tensors stay
on their device, so training can compute features on the GPU.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

SAMPLE_RATE = 16_000
WIN_LENGTH = 400   # 25 ms
HOP_LENGTH = 160   # 10 ms
N_FFT = 512
N_MELS = 80
F_MIN = 20.0
F_MAX = 8_000.0
LOG_EPS = 1e-6
N_MFCC = 13
DELTA_WIDTH = 5
FRAMES_PER_VIDEO_FRAME = 4   # 100 fps features vs 25 fps video

TARGET_DBFS = -25.0
MAX_GAIN_DB = 20.0

FEATURE_DIMS = {"logmel80": N_MELS, "mfcc39": 3 * N_MFCC, "logmel_mfcc": N_MELS + 3 * N_MFCC}


@lru_cache(maxsize=None)
def _mel_filters(device: torch.device) -> torch.Tensor:
    # Slaney-style filters, the librosa default.
    return AF.melscale_fbanks(N_FFT // 2 + 1, F_MIN, F_MAX, N_MELS, SAMPLE_RATE,
                              norm="slaney", mel_scale="slaney").to(device)


@lru_cache(maxsize=None)
def _window(device: torch.device) -> torch.Tensor:
    return torch.hann_window(WIN_LENGTH, periodic=True, device=device)


@lru_cache(maxsize=None)
def _dct(device: torch.device) -> torch.Tensor:
    return AF.create_dct(N_MFCC, N_MELS, norm="ortho").to(device)


def _as_batch(wave) -> tuple[torch.Tensor, bool, bool]:
    """-> (float32 tensor [B, N], was_numpy, was_unbatched)."""
    was_numpy = not torch.is_tensor(wave)
    x = torch.as_tensor(np.asarray(wave) if was_numpy else wave, dtype=torch.float32)
    if x.ndim not in (1, 2):
        raise ValueError(f"expected a waveform [N] or batch [B, N], got shape {tuple(x.shape)}")
    unbatched = x.ndim == 1
    return (x.unsqueeze(0) if unbatched else x), was_numpy, unbatched


def _restore(x: torch.Tensor, was_numpy: bool, unbatched: bool):
    if unbatched:
        x = x.squeeze(0)
    return x.cpu().numpy() if was_numpy else x


def num_feature_frames(n_samples: int, n_frames: int | None = None) -> int:
    """Feature frames for a clip: ``4 * n_frames`` if the video frame count is known."""
    return FRAMES_PER_VIDEO_FRAME * n_frames if n_frames is not None else n_samples // HOP_LENGTH


def _logmel(x: torch.Tensor, n_frames: int | None) -> torch.Tensor:
    """[B, N] tensor -> [B, T_a, 80] tensor."""
    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                      window=_window(x.device), center=True, pad_mode="constant",
                      return_complex=True)
    power = spec.real.square() + spec.imag.square()             # [B, 257, frames]
    feats = torch.log(power.transpose(1, 2) @ _mel_filters(x.device) + LOG_EPS)
    t_a = num_feature_frames(x.shape[-1], n_frames)
    if feats.shape[1] < t_a:
        # Short audio: pad with silence so audio and video frames stay aligned.
        feats = F.pad(feats, (0, 0, 0, t_a - feats.shape[1]), value=float(np.log(LOG_EPS)))
    return feats[:, :t_a]


def _mfcc39(logmel_feats: torch.Tensor) -> torch.Tensor:
    """[B, T_a, 80] log-Mel -> [B, T_a, 39] MFCC + delta + delta-delta."""
    mfcc = logmel_feats @ _dct(logmel_feats.device)               # [B, T_a, 13]
    delta = AF.compute_deltas(mfcc.transpose(1, 2), win_length=DELTA_WIDTH)
    delta2 = AF.compute_deltas(delta, win_length=DELTA_WIDTH)
    return torch.cat([mfcc, delta.transpose(1, 2), delta2.transpose(1, 2)], dim=-1)


def logmel(wave, n_frames: int | None = None):
    """Waveform ``[N]`` / ``[B, N]`` at 16 kHz -> log-Mel ``[T_a, 80]`` / ``[B, T_a, 80]``."""
    x, was_numpy, unbatched = _as_batch(wave)
    return _restore(_logmel(x, n_frames), was_numpy, unbatched)


def mfcc39(wave, n_frames: int | None = None):
    """Waveform -> 13 MFCC + delta + delta-delta, ``[T_a, 39]`` / ``[B, T_a, 39]``."""
    x, was_numpy, unbatched = _as_batch(wave)
    return _restore(_mfcc39(_logmel(x, n_frames)), was_numpy, unbatched)


def logmel_mfcc(wave, n_frames: int | None = None):
    """Waveform -> log-Mel and MFCC39 concatenated, ``[T_a, 119]`` / ``[B, T_a, 119]``."""
    x, was_numpy, unbatched = _as_batch(wave)
    mel = _logmel(x, n_frames)
    return _restore(torch.cat([mel, _mfcc39(mel)], dim=-1), was_numpy, unbatched)


FEATURES = {"logmel80": logmel, "mfcc39": mfcc39, "logmel_mfcc": logmel_mfcc}


def get_feature_fn(name: str):
    """Config switch ``audio_features: logmel80 | mfcc39 | logmel_mfcc``."""
    try:
        return FEATURES[name]
    except KeyError:
        raise ValueError(f"unknown audio_features {name!r}; choose from {sorted(FEATURES)}") from None


def rms_normalise(wave, target_dbfs: float = TARGET_DBFS, max_gain_db: float = MAX_GAIN_DB):
    """Scale each waveform to ``target_dbfs`` RMS, with the gain capped at ``max_gain_db``.

    Applied after noise mixing, so the network sees a consistent level whatever
    the SNR. The cap stops near-silent clips being amplified into pure noise.
    """
    x, was_numpy, unbatched = _as_batch(wave)
    rms = x.square().mean(dim=-1, keepdim=True).sqrt()
    target = 10.0 ** (target_dbfs / 20.0)
    gain = (target / rms.clamp_min(1e-12)).clamp_max(10.0 ** (max_gain_db / 20.0))
    return _restore(x * gain, was_numpy, unbatched)


def normalise(features, stats: Mapping, eps: float = 1e-5):
    """Standardise ``[..., T_a, D]`` features with training-set ``audio_mean`` / ``audio_std`` ``[D]``.

    ``stats`` is the contents of ``data/stats/feature_norm.npz`` (card W2-08).
    """
    mean, std = stats["audio_mean"], stats["audio_std"]
    if torch.is_tensor(features):
        mean = torch.as_tensor(mean, dtype=features.dtype, device=features.device)
        std = torch.as_tensor(std, dtype=features.dtype, device=features.device)
        return (features - mean) / std.clamp_min(eps)
    return ((features - np.asarray(mean)) / np.maximum(np.asarray(std), eps)).astype(np.float32)

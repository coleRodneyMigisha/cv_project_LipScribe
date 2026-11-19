"""Mix MUSAN noise into clean speech at a target SNR (W2-05).

Implements Algorithm 1 of the implementation document (Section 4.4):

    if SNR is "clean": return s
    w~    = segment of length N from a noise file, random offset (file looped if shorter)
    Ps    = mean(s^2),  Pw = mean(w~^2)            (powers over the whole clip)
    alpha = sqrt(Ps / (Pw * 10^(SNR/10)))
    y     = s + alpha * w~
    assert |10 log10(Ps / mean((alpha w~)^2)) - SNR| < 0.01
    if max|y| > 1: y = y / max|y|                  (scales speech and noise together: SNR unchanged)

Mixing changes only the audio waveform. The video and the transcript of the
utterance are unchanged, so every noisy copy keeps the transcript of its clean
original and the same sentences can be scored at every SNR.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np

CLEAN = float("inf")
SNR_CONDITIONS = (CLEAN, 10.0, 5.0, 0.0, -5.0)   # clean, 10, 5, 0, -5 dB
SNR_TOLERANCE_DB = 0.01


def is_clean(snr_db) -> bool:
    return isinstance(snr_db, str) and snr_db.lower() == "clean" or not math.isfinite(float(snr_db))


def noise_segment(noise: np.ndarray, n: int, rng: random.Random | None = None) -> tuple[np.ndarray, int]:
    """Length-n segment of ``noise`` at a random offset; the file is looped if shorter."""
    rng = rng or random.Random()
    noise = np.asarray(noise, dtype=np.float32)
    if noise.ndim != 1 or len(noise) == 0:
        raise ValueError(f"noise must be a non-empty 1-D waveform, got shape {noise.shape}")
    if len(noise) < n:
        noise = np.tile(noise, math.ceil(n / len(noise)))
    offset = rng.randint(0, len(noise) - n)
    return noise[offset:offset + n], offset


def mix_noise(clean: np.ndarray, noise: np.ndarray, snr_db, rng: random.Random | None = None,
              return_info: bool = False):
    """Mix ``noise`` into ``clean`` at ``snr_db`` (dB, or "clean"/inf).

    Args:
        clean: float waveform [N] at 16 kHz.
        noise: float noise waveform (any length; looped if shorter than N).
        snr_db: target SNR in dB; "clean" or +inf returns the clean audio unchanged.
        rng: random.Random used for the noise offset (pass a seeded one for fixed mixtures).
        return_info: also return a dict with offset, alpha, measured SNR and peak scaling.

    Returns:
        float32 [N] noisy waveform with peak <= 1, or (waveform, info) if return_info.
    """
    s = np.asarray(clean, dtype=np.float32)
    if s.ndim != 1:
        raise ValueError(f"clean must be a 1-D waveform, got shape {s.shape}")
    if is_clean(snr_db):
        out = s.copy()
        info = {"offset": -1, "alpha": 0.0, "measured_snr_db": CLEAN, "peak_scale": 1.0}
        return (out, info) if return_info else out

    target = float(snr_db)
    w, offset = noise_segment(noise, len(s), rng)
    s64, w64 = s.astype(np.float64), w.astype(np.float64)
    ps, pw = float(np.mean(s64 ** 2)), float(np.mean(w64 ** 2))
    if ps <= 0:
        raise ValueError("clean speech is silent (zero power); SNR is undefined")
    if pw <= 0:
        raise ValueError("noise segment is silent (zero power); pick another noise file")

    alpha = math.sqrt(ps / (pw * 10 ** (target / 10)))
    scaled = alpha * w64
    measured = 10 * math.log10(ps / float(np.mean(scaled ** 2)))
    assert abs(measured - target) < SNR_TOLERANCE_DB, f"measured {measured:.4f} dB vs target {target} dB"

    y = s64 + scaled
    peak = float(np.max(np.abs(y)))
    peak_scale = 1.0 / peak if peak > 1.0 else 1.0
    y = (y * peak_scale).astype(np.float32)

    info = {"offset": offset, "alpha": alpha, "measured_snr_db": measured, "peak_scale": peak_scale}
    return (y, info) if return_info else y


class NoisePool:
    """A list of noise files with a small in-memory cache, for repeated mixing."""

    def __init__(self, files: Sequence[str | Path], loader=None) -> None:
        self.files = [Path(f) for f in files]
        if not self.files:
            raise ValueError("Empty noise pool")
        if loader is None:
            from lipscribe.data.dataset import load_wav as loader
        self._load = loader
        self._cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.files)

    def get(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            self._cache[path] = self._load(path)
        return self._cache[path]

    def mix(self, clean: np.ndarray, snr_db, rng: random.Random):
        """Draw a file with ``rng`` and mix. Returns (waveform, info) with info["noise_file"]."""
        if is_clean(snr_db):
            y, info = mix_noise(clean, np.zeros(1, np.float32), snr_db, return_info=True)
            info["noise_file"] = ""
            return y, info
        path = rng.choice(self.files)
        y, info = mix_noise(clean, self.get(path), snr_db, rng=rng, return_info=True)
        info["noise_file"] = str(path)
        return y, info

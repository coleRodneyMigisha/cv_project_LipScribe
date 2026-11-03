"""Helpers for computing and loading feature-normalisation statistics."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

STATS_PATH = Path("data/stats/feature_norm.npz")


class RunningMoments:
    """Streaming mean/std over rows (float64). dim=None pools every value into one scalar."""

    def __init__(self, dim: int | None = None):
        shape = () if dim is None else (dim,)
        self.n = 0
        self.mean = np.zeros(shape, np.float64)
        self.m2 = np.zeros(shape, np.float64)

    def update(self, x) -> None:
        x = np.asarray(x, np.float64)
        x = x.reshape(-1) if self.mean.ndim == 0 else x.reshape(-1, self.mean.shape[0])
        n_b = x.shape[0]
        if n_b == 0:
            return
        mean_b = x.mean(axis=0)
        m2_b = ((x - mean_b) ** 2).sum(axis=0)
        delta, n_tot = mean_b - self.mean, self.n + n_b
        self.mean = self.mean + delta * n_b / n_tot
        self.m2 = self.m2 + m2_b + delta**2 * self.n * n_b / n_tot
        self.n = n_tot

    @property
    def std(self):
        return np.sqrt(self.m2 / max(self.n, 1))


def split_hash(utt_ids) -> str:
    """Fingerprint of the training utterance list, to detect stale stats."""
    return hashlib.sha256("\n".join(sorted(utt_ids)).encode()).hexdigest()[:16]


def load_norm_stats(path: str | Path = STATS_PATH) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def normalise_pixels(crops: np.ndarray, stats: dict, eps: float = 1e-5) -> np.ndarray:
    """crops: 0-1 float (or uint8) mouth crops -> standardised float32."""
    x = crops.astype(np.float32)
    if crops.dtype == np.uint8:
        x /= 255.0
    return ((x - stats["pixel_mean"]) / max(float(stats["pixel_std"]), eps)).astype(np.float32)
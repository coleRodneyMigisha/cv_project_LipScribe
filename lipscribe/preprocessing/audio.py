"""Shared audio I/O: load any audio as 16 kHz mono (W2-04)."""
from __future__ import annotations

import numpy as np
import soundfile as sf

TARGET_SR = 16000

try:
    import soxr                      # high-quality band-limited resampler
    _HAVE_SOXR = True
except ImportError:
    from scipy.signal import resample_poly   # polyphase fallback
    _HAVE_SOXR = False


def load_audio(path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read a WAV -> 1-D float32 mono waveform at target_sr.

    - averages multi-channel audio to mono
    - resamples with a band-limited resampler (soxr, or scipy's
      resample_poly if soxr isn't installed)

    Shared by prepare_audio.py, the data loader and the live demo,
    so training and inference always see identical audio handling.
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                      # average to mono
    if sr != target_sr:
        if _HAVE_SOXR:
            audio = soxr.resample(audio, sr, target_sr)
        else:
            from math import gcd
            g = gcd(int(sr), int(target_sr))
            audio = resample_poly(audio, target_sr // g, int(sr) // g)
    return np.ascontiguousarray(audio, dtype=np.float32)
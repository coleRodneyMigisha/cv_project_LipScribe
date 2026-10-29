"""Shared audio I/O: load and resample to 16 kHz mono."""
import numpy as np
import soundfile as sf
from pathlib import Path

try:
    import soxr  # high-quality band-limited resampler
    _HAVE_SOXR = True
except ImportError:
    from scipy.signal import resample_poly
    _HAVE_SOXR = False

TARGET_SR = 16000

def load_audio(path, target_sr=TARGET_SR):
    """Read any WAV -> (float32 mono waveform @ target_sr)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # average to mono
    if sr != target_sr:
        if _HAVE_SOXR:
            audio = soxr.resample(audio, sr, target_sr)
        else:
            # band-limited (polyphase) resampling
            from math import gcd
            g = gcd(sr, target_sr)
            audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio
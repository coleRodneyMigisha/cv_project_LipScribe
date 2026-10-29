"""GRID dataset and collate function shared by every LipScribe model (W2-09).

Item and batch keys follow docs/INTERFACES.md (Section 2).

Training items: cached mouth crops + clean 16 kHz audio mixed with MUSAN noise
on the fly at an SNR drawn uniformly from {clean, 10, 5, 0, -5} dB.
Val/test items: the fixed pre-mixed noisy files for one chosen SNR
(written by make_noisy_test.py, W2-06), so every model hears identical audio.

Video and audio are kept on one 40 ms grid: 1 video frame = 640 samples at
16 kHz = 4 log-Mel frames (10 ms hop). Audio is cut or zero-padded to exactly
640 * T samples so that T_a = 4T and the encoder length is T.
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from lipscribe.data.tokenizer import encode

SAMPLE_RATE = 16_000
HOP = 160                       # 10 ms at 16 kHz
SAMPLES_PER_FRAME = 640         # 40 ms video frame at 16 kHz
SNR_CHOICES = ("clean", 10, 5, 0, -5)
CLEAN_SNR_VALUE = 30.0          # "clean" mapped to 30 dB where a number is needed (Section 7)


# ---------------------------------------------------------------- helpers

def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def check_speaker_disjoint(rows: Sequence[dict]) -> None:
    """Raise if any speaker appears in more than one partition."""
    seen: dict[str, str] = {}
    for r in rows:
        spk, part = r["speaker"], r.get("partition", "")
        if not part:
            continue
        if spk in seen and seen[spk] != part:
            raise ValueError(f"Speaker leakage: {spk} is in both {seen[spk]!r} and {part!r}")
        seen[spk] = part


def ctc_min_frames(target: Sequence[int]) -> int:
    """Frames CTC needs for a target: its length plus one blank per repeated pair."""
    repeats = sum(1 for a, b in zip(target, target[1:]) if a == b)
    return len(target) + repeats


def load_crops(mouths_path: str, valid_path: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Cached crops uint8 [T,96,96] and validity bool [T] (all True if no file)."""
    if mouths_path.endswith(".npz"):
        with np.load(mouths_path) as z:
            crops = z["crops"]
            valid = z["valid"].astype(bool) if "valid" in z else np.ones(len(crops), bool)
    else:
        crops = np.load(mouths_path)
        valid = np.load(valid_path).astype(bool) if valid_path else np.ones(len(crops), bool)
    if crops.ndim != 3 or crops.shape[1:] != (96, 96):
        raise ValueError(f"{mouths_path}: expected [T,96,96] crops, got {crops.shape}")
    return crops.astype(np.uint8, copy=False), valid


def load_wav(path: str | Path) -> np.ndarray:
    wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz audio, got {sr} Hz "
                         "(run the audio standardisation step, W2-04)")
    return wave.astype(np.float32, copy=False)


def fit_length(wave: np.ndarray, n_samples: int) -> np.ndarray:
    """Cut or zero-pad at the end to exactly n_samples (keeps the start aligned)."""
    if len(wave) >= n_samples:
        return wave[:n_samples]
    return np.pad(wave, (0, n_samples - len(wave)))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: random.Random | None = None) -> np.ndarray:
    """y = s + alpha * w with alpha = sqrt(Ps / (Pw * 10^(snr/10))) (Section 4.4).

    Fallback used only when Lisa's mixer (W2-05) is not importable; it uses the
    same formula. A random noise segment of the right length is chosen, and
    short noise is looped.
    """
    rng = rng or random.Random()
    if len(noise) == 0:
        raise ValueError("Empty noise signal")
    if len(noise) < len(clean):
        noise = np.tile(noise, math.ceil(len(clean) / len(noise)))
    start = rng.randint(0, len(noise) - len(clean))
    w = noise[start:start + len(clean)]
    ps = float(np.mean(clean ** 2)) + 1e-12
    pw = float(np.mean(w ** 2)) + 1e-12
    alpha = math.sqrt(ps / (pw * 10 ** (snr_db / 10)))
    return (clean + alpha * w).astype(np.float32)


def _default_mixer() -> Callable:
    """Use the team's noise mixer (W2-05) if merged, else the fallback above."""
    try:
        from lipscribe.preprocessing.noise import mix_noise  # type: ignore
        return lambda clean, noise, snr, rng: mix_noise(clean, noise, snr)
    except Exception:
        return mix_at_snr


# ---------------------------------------------------------------- dataset

class GridDataset(Dataset):
    """One item per usable utterance of one partition.

    Args:
        manifest: path to data/manifest.csv (scripts/build_manifest.py).
        partition: "train", "val" or "test".
        noise_files: training noise pool (MUSAN train split). Required for
            partition="train" unless train_snrs is ("clean",).
        snr: val/test only. "clean" or one of 10, 5, 0, -5.
        noisy_dir: val/test only. Folder with <snr>/<utt_id>.wav pre-mixed files.
        train_snrs: SNRs sampled uniformly for training items.
        seed: makes the training noise draws reproducible.
        mixer: optional callable(clean, noise, snr_db, rng) -> noisy.
    """

    def __init__(
        self,
        manifest: str | Path,
        partition: str,
        noise_files: Sequence[str | Path] | None = None,
        snr: str | int = "clean",
        noisy_dir: str | Path | None = None,
        train_snrs: Sequence[str | int] = SNR_CHOICES,
        seed: int = 1,
        mixer: Callable | None = None,
    ) -> None:
        rows = read_manifest(manifest)
        check_speaker_disjoint(rows)
        self.rows = [r for r in rows
                     if r.get("partition") == partition and str(r.get("usable")).lower() == "true"]
        if not self.rows:
            raise ValueError(f"No usable utterances for partition {partition!r} in {manifest}")
        self.partition = partition
        self.train = partition == "train"
        self.train_snrs = tuple(train_snrs)
        self.snr = snr
        self.noisy_dir = Path(noisy_dir) if noisy_dir else None
        self.noise_files = [Path(p) for p in (noise_files or [])]
        self.mixer = mixer or _default_mixer()
        self.rng = random.Random(seed)
        self._noise_cache: dict[Path, np.ndarray] = {}

        if self.train and any(s != "clean" for s in self.train_snrs) and not self.noise_files:
            raise ValueError("Training with noise needs noise_files (the MUSAN train pool)")
        if not self.train and snr != "clean" and self.noisy_dir is None:
            raise ValueError("Noisy val/test needs noisy_dir (output of make_noisy_test.py)")

    def __len__(self) -> int:
        return len(self.rows)

    def _noise(self) -> np.ndarray:
        path = self.rng.choice(self.noise_files)
        if path not in self._noise_cache:
            self._noise_cache[path] = load_wav(path)
        return self._noise_cache[path]

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        crops, valid = load_crops(r["mouths_path"], r.get("valid_path", ""))
        T = crops.shape[0]
        n_samples = T * SAMPLES_PER_FRAME

        if self.train:
            snr = self.rng.choice(self.train_snrs)
            wave = fit_length(load_wav(r["audio_path"]), n_samples)
            if snr != "clean":
                wave = self.mixer(wave, self._noise(), float(snr), self.rng)
        else:
            snr = self.snr
            if snr == "clean":
                wave = load_wav(r["audio_path"])
            else:
                wave = load_wav(self.noisy_dir / str(snr) / f"{r['utt_id']}.wav")
            wave = fit_length(wave, n_samples)

        text = r["transcript"]
        return {
            "utt_id": r["utt_id"],
            "speaker": r["speaker"],
            "video": crops,
            "video_valid": valid,
            "audio": np.asarray(wave, dtype=np.float32),
            "snr_db": CLEAN_SNR_VALUE if snr == "clean" else float(snr),
            "text": text,
            "target": np.asarray(encode(text), dtype=np.int64),
        }


# ---------------------------------------------------------------- collate

def collate(items: Sequence[dict]) -> dict:
    """Pad a list of items into the batch dictionary of docs/INTERFACES.md.

    video          float32 [B,1,T,96,96]  scaled to 0-1, zero-padded in time
    audio          float32 [B,N_max]      zero-padded waveforms (log-Mel is done on the GPU)
    audio_lengths  int64 [B]              true samples
    feat_lengths   int64 [B]              true log-Mel frames T_a = samples // 160
    lengths        int64 [B]              encoder frames ceil(T_a / 4)
    mask           bool  [B,T]            True on real frames
    video_valid    bool  [B,T]            Face Mesh success per frame (False on padding)
    targets        int64 [sum L]          concatenated, for nn.CTCLoss
    target_lengths int64 [B]
    """
    if not items:
        raise ValueError("Empty batch")
    B = len(items)
    t_video = [it["video"].shape[0] for it in items]
    n_audio = [len(it["audio"]) for it in items]
    T_max, N_max = max(t_video), max(n_audio)

    video = torch.zeros(B, 1, T_max, 96, 96, dtype=torch.float32)
    video_valid = torch.zeros(B, T_max, dtype=torch.bool)
    audio = torch.zeros(B, N_max, dtype=torch.float32)
    for b, it in enumerate(items):
        t = t_video[b]
        video[b, 0, :t] = torch.from_numpy(np.ascontiguousarray(it["video"])).float().div_(255.0)
        video_valid[b, :t] = torch.from_numpy(np.asarray(it["video_valid"], dtype=bool))
        audio[b, : n_audio[b]] = torch.from_numpy(it["audio"])

    audio_lengths = torch.tensor(n_audio, dtype=torch.long)
    feat_lengths = audio_lengths // HOP
    lengths = torch.div(feat_lengths + 3, 4, rounding_mode="floor")  # ceil(T_a / 4)
    mask = torch.arange(T_max)[None, :] < lengths[:, None]

    targets, target_lengths = [], []
    for b, it in enumerate(items):
        tgt = [int(x) for x in it["target"]]
        need = ctc_min_frames(tgt)
        if int(lengths[b]) < need:
            raise ValueError(
                f"CTC condition fails for {it['utt_id']}: {int(lengths[b])} frames "
                f"< {need} needed for {it['text']!r}")
        targets.extend(tgt)
        target_lengths.append(len(tgt))

    return {
        "utt_ids": [it["utt_id"] for it in items],
        "video": video,
        "video_valid": video_valid,
        "audio": audio,
        "audio_lengths": audio_lengths,
        "feat_lengths": feat_lengths,
        "lengths": lengths,
        "mask": mask,
        "targets": torch.tensor(targets, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": [it["text"] for it in items],
        "snr_db": torch.tensor([it["snr_db"] for it in items], dtype=torch.float32),
    }

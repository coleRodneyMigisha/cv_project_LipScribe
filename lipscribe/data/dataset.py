"""GRID dataset and collate function shared by every LipScribe model (W2-09).

Matches the calls in scripts/train.py:
    GridDataset(manifest, splits, "train", snrs=[inf, 10, 5, 0, -5])
    GridDataset(manifest, splits, "val", snr_db=5.0)
    DataLoader(..., collate_fn=collate)
Item and batch keys follow docs/INTERFACES.md (Section 2). "clean" is snr_db = +inf.

Training items: cached mouth crops + clean 16 kHz audio mixed with MUSAN noise
on the fly at an SNR drawn uniformly from ``snrs``.
Val/test items: the fixed pre-mixed noisy file for one SNR
(<noisy_dir>/<snr>/<utt_id>.wav, from make_noisy_test.py, W2-06). Until those
files exist, the same mixture is made on the fly from the val/test noise pool
with a seed fixed per (utterance, SNR), so every model still hears identical audio.

Video and audio share one 40 ms grid: 1 video frame = 640 samples at 16 kHz =
4 log-Mel frames (10 ms hop). Audio is cut or zero-padded to exactly 640 * T
samples, so T_a = 4T and the encoder length is T.

Our GRID audio is end-pointed (trimmed to the speech, ~1.2 s of a 3 s clip), so
a trimmed file is first shifted right by the manifest's ``speech_start_s`` to
sit under the lips (see ``place_on_video_grid``). After noise mixing every
waveform is RMS-normalised to -25 dBFS (gain capped at +20 dB), as the app does.
"""
from __future__ import annotations

import csv
import math
import random
import zlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, get_worker_info

from lipscribe.data.tokenizer import encode
from lipscribe.preprocessing.features import rms_normalise

SAMPLE_RATE = 16_000
HOP = 160                       # 10 ms at 16 kHz
SAMPLES_PER_FRAME = 640         # 40 ms video frame at 16 kHz
CLEAN = float("inf")            # INTERFACES.md: snr_db for clean audio
DEFAULT_TRAIN_SNRS = (CLEAN, 10.0, 5.0, 0.0, -5.0)
DEFAULT_NOISY_DIR = Path("data/processed/noisy_test")
# MUSAN pool lists written by scripts/make_noise_splits.py (columns: file, pool).
DEFAULT_NOISE_SPLITS = (Path("data/noise_splits.csv"), Path("Data/Noise_splits/splits.csv"))


# ---------------------------------------------------------------- helpers

def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def read_splits(path: str | Path) -> dict[str, str]:
    """speaker -> partition from data/splits.csv. Raises on speaker leakage."""
    mapping: dict[str, str] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            spk = row["speaker"].strip().lower()
            part = row["partition"].strip().lower()
            if spk in mapping and mapping[spk] != part:
                raise ValueError(f"Speaker leakage: {spk} is in both {mapping[spk]!r} and {part!r}")
            mapping[spk] = part
    return mapping


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
    """Cached crops uint8 [T,96,96] and validity bool [T] (all True if no mask file)."""
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


def place_on_video_grid(wave: np.ndarray, n_samples: int, speech_start_s) -> np.ndarray:
    """Put a waveform on the video's time grid: exactly n_samples, speech under the lips.

    A file clearly shorter than the video (more than 2 frames, 80 ms) is an
    end-pointed GRID file that starts at the first word, so it is shifted right
    by ``speech_start_s`` (alignment times are relative to the full video). A
    full-length file is only cut or padded at the end, so nothing is shifted twice.
    """
    try:
        start = float(speech_start_s)
    except (TypeError, ValueError):
        start = math.nan
    if len(wave) < n_samples - 2 * SAMPLES_PER_FRAME and math.isfinite(start) and start > 0:
        wave = np.pad(wave, (int(round(start * SAMPLE_RATE)), 0))
    return fit_length(wave, n_samples)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: random.Random | None = None) -> np.ndarray:
    """y = s + alpha * w with alpha = sqrt(Ps / (Pw * 10^(snr/10))) (Section 4.4).

    Used only when the team's mixer (W2-05, lipscribe.preprocessing.noise) is not
    merged; same formula. A random noise segment is taken; short noise is looped.
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


def _team_mixer(clean, noise, snr_db, rng):
    from lipscribe.preprocessing.noise import mix_noise  # type: ignore
    return mix_noise(clean, noise, snr_db)


def _default_mixer() -> Callable:
    # Module-level functions, not a lambda, so the dataset pickles into
    # DataLoader workers on Windows (spawn).
    try:
        from lipscribe.preprocessing.noise import mix_noise  # type: ignore  # noqa: F401
        return _team_mixer
    except Exception:
        return mix_at_snr


def read_noise_pool(noise_splits: str | Path | None, pool: str) -> list[Path]:
    """Noise files of one pool ("train", "val" or "test") from make_noise_splits.py output."""
    candidates = [Path(noise_splits)] if noise_splits else list(DEFAULT_NOISE_SPLITS)
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        files = []
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["pool"].strip().lower() != pool:
                    continue
                p = Path(row["file"])
                if not p.is_absolute():
                    # make_noise_splits.py stores paths relative to the folder above Noise_splits/
                    for base in (csv_path.parent.parent, csv_path.parent, Path.cwd()):
                        if (base / p).exists():
                            p = base / p
                            break
                files.append(p)
        return files
    return []


def snr_label(snr_db: float) -> str:
    """Folder name for an SNR: 10.0 -> "10", -5.0 -> "-5"."""
    return str(int(snr_db)) if float(snr_db).is_integer() else str(snr_db)


# ---------------------------------------------------------------- dataset

class GridDataset(Dataset):
    """One item per usable utterance of one partition.

    Args:
        manifest: data/manifest.csv (scripts/build_manifest.py).
        splits: data/splits.csv (speaker -> train / val / test / excluded).
        partition: "train", "val" or "test".
        snrs: training only; SNRs in dB sampled uniformly per item (inf = clean).
        snr_db: val/test only; the one SNR this dataset serves (inf = clean).
        noise_files: optional explicit noise list; otherwise read from noise_splits.
        noise_splits: MUSAN pool CSV (file, pool); defaults to data/noise_splits.csv
            or Data/Noise_splits/splits.csv.
        noisy_dir: folder with <snr>/<utt_id>.wav fixed noisy val/test files.
        seed: reproducible noise draws.
        mixer: optional callable(clean, noise, snr_db, rng) -> noisy.
    """

    def __init__(
        self,
        manifest: str | Path,
        splits: str | Path,
        partition: str,
        snrs: Sequence[float] = DEFAULT_TRAIN_SNRS,
        snr_db: float = CLEAN,
        noise_files: Sequence[str | Path] | None = None,
        noise_splits: str | Path | None = None,
        noisy_dir: str | Path = DEFAULT_NOISY_DIR,
        seed: int = 1,
        mixer: Callable | None = None,
    ) -> None:
        speaker_part = read_splits(splits)
        rows = read_manifest(manifest)
        self.rows = [r for r in rows
                     if speaker_part.get(r["speaker"].strip().lower()) == partition
                     and str(r.get("usable")).strip().lower() == "true"]
        if not self.rows:
            raise ValueError(f"No usable utterances for partition {partition!r} "
                             f"({manifest} + {splits}). Has the manifest been rebuilt?")
        self.partition = partition
        self.train = partition == "train"
        self.snrs = tuple(float(s) for s in snrs)
        self.snr_db = float(snr_db)
        self.noisy_dir = Path(noisy_dir)
        self.mixer = mixer or _default_mixer()
        self.seed = seed
        self.rng = random.Random(seed)
        self._rng_seed: int | None = None
        self._noise_cache: dict[Path, np.ndarray] = {}

        needs_noise = any(math.isfinite(s) for s in self.snrs) if self.train else math.isfinite(self.snr_db)
        self.noise_files = [Path(p) for p in noise_files] if noise_files is not None \
            else read_noise_pool(noise_splits, "train" if self.train else partition)
        if needs_noise and not self.noise_files and self.train:
            raise ValueError(
                "Noisy training needs the MUSAN train pool: pass noise_files=[...] or "
                "noise_splits=<make_noise_splits.py CSV>, or train clean-only with snrs=[inf].")

    def __len__(self) -> int:
        return len(self.rows)

    def _train_rng(self) -> random.Random:
        """RNG for SNR and noise draws, distinct per DataLoader worker and per epoch.

        Every worker receives a copy of this dataset with the same ``self.rng``
        state, which would make all workers draw the same SNRs, and the same
        ones again every epoch. Inside a worker, ``torch.initial_seed()`` is
        base_seed + worker_id with a fresh base_seed each epoch (drawn from the
        main process's seeded torch RNG), so reseeding from it is both varied
        and reproducible.
        """
        if get_worker_info() is not None:
            seed = torch.initial_seed()
            if seed != self._rng_seed:
                self.rng, self._rng_seed = random.Random(seed), seed
        return self.rng

    def _noise(self, rng: random.Random) -> np.ndarray:
        path = rng.choice(self.noise_files)
        if path not in self._noise_cache:
            self._noise_cache[path] = load_wav(path)
        return self._noise_cache[path]

    def _eval_audio(self, r: dict, n_samples: int) -> np.ndarray:
        start = r.get("speech_start_s")
        clean = place_on_video_grid(load_wav(r["audio_path"]), n_samples, start)
        if not math.isfinite(self.snr_db):
            return clean
        fixed = self.noisy_dir / snr_label(self.snr_db) / f"{r['utt_id']}.wav"
        if fixed.exists():
            return place_on_video_grid(load_wav(fixed), n_samples, start)
        if not self.noise_files:
            raise FileNotFoundError(
                f"No fixed noisy file {fixed} and no {self.partition} noise pool to mix one")
        # Deterministic stand-in for the W2-06 files: same noise for every model and run.
        rng = random.Random(zlib.crc32(f"{r['utt_id']}|{self.snr_db}".encode()))
        return self.mixer(clean, self._noise(rng), self.snr_db, rng)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        crops, valid = load_crops(r["mouths_path"], r.get("valid_path", ""))
        T = crops.shape[0]
        n_samples = T * SAMPLES_PER_FRAME

        if self.train:
            rng = self._train_rng()
            snr = rng.choice(self.snrs)
            wave = place_on_video_grid(load_wav(r["audio_path"]), n_samples, r.get("speech_start_s"))
            if math.isfinite(snr):
                wave = self.mixer(wave, self._noise(rng), snr, rng)
        else:
            snr = self.snr_db
            wave = self._eval_audio(r, n_samples)
        wave = rms_normalise(np.asarray(wave, dtype=np.float32))

        text = r["transcript"]
        return {
            "utt_id": r["utt_id"],
            "speaker": r["speaker"],
            "video": crops,
            "video_valid": valid,
            "audio": np.asarray(wave, dtype=np.float32),
            "snr_db": float(snr),
            "text": text,
            "target": np.asarray(encode(text), dtype=np.int64),
        }


# ---------------------------------------------------------------- collate

def collate(items: Sequence[dict]) -> dict:
    """Pad a list of items into the batch dictionary of docs/INTERFACES.md.

    video          float32 [B,1,T,96,96]  scaled to 0-1, zero-padded in time
    video_valid    bool  [B,T]            Face Mesh success per frame (False on padding)
    audio          float32 [B,640*T]      zero-padded waveforms (log-Mel is done on the GPU)
    audio_lengths  int64 [B]              true samples
    feat_lengths   int64 [B]              true log-Mel frames T_a = samples // 160
    lengths        int64 [B]              encoder frames ceil(T_a / 4)
    mask           bool  [B,T]            True on real frames
    targets        int64 [sum L]          concatenated, for nn.CTCLoss
    target_lengths int64 [B]
    """
    if not items:
        raise ValueError("Empty batch")
    B = len(items)
    t_video = [it["video"].shape[0] for it in items]
    n_audio = [len(it["audio"]) for it in items]
    T_max = max(t_video)
    N_max = max(max(n_audio), T_max * SAMPLES_PER_FRAME)

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

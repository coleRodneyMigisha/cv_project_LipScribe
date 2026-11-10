"""Live inference for the LipScribe demo (W3-10, Mode A).

The demo must give exactly the caption evaluate.py gives for the same clip, so
this module reuses evaluate.py's chain unchanged:

    item dict (same keys and audio handling as GridDataset)
      -> lipscribe.data.dataset.collate
      -> scripts.train.build_feature_step   (log-Mel + training-set normalisation)
      -> the checkpoint's model             (built by lipscribe.models.build_model)
      -> lipscribe.decoding.get_decoder     (greedy / beam / grammar)

Only the source of the inputs differs: mouth crops come from MediaPipe Face Mesh
run live on the uploaded or recorded video (the same crop_video() that
extract_mouths.py used for the cached crops), and audio comes from the uploaded
file, the microphone, or the video's own audio track (extracted with ffmpeg).

    from lipscribe.app.inference import LipScribe
    engine = LipScribe("runs/b1_audio_only_20261103-193743/best.pt")
    engine.transcribe("clip.mp4").caption
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lipscribe.data.dataset import SAMPLES_PER_FRAME, collate, place_on_video_grid
from lipscribe.decoding import get_decoder
from lipscribe.preprocessing.audio import load_audio
from lipscribe.preprocessing.features import rms_normalise

VIDEO_FPS = 25
DEFAULT_FACE_MODEL = Path("models/face_landmarker.task")
MIN_FACE_FRACTION = 0.5     # below this a clip counts as "no face"
MAX_SECONDS = 10.0          # longer uploads are cut (the models were trained on 3 s clips)


class FriendlyError(Exception):
    """An input problem to show to the user as-is (no stack trace)."""


@dataclass
class Transcription:
    caption: str
    n_frames: int
    face_fraction: float | None        # share of frames with a detected face (None: no video)
    crops: np.ndarray | None            # uint8 [T, 96, 96] for the mouth strip
    gate: np.ndarray | None             # per-frame audio weight (fused model P only)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


# ------------------------------------------------------------------ ffmpeg helpers

def find_ffmpeg() -> str:
    """ffmpeg from imageio-ffmpeg (bundled, works on Windows) or the system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise FriendlyError("ffmpeg is not installed. Run: pip install imageio-ffmpeg")


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, text=True)


def video_fps_and_frames(video_path: str | Path) -> tuple[float, int]:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FriendlyError("Could not open the video. Try an .mp4, .webm or .mpg file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), frames


def to_25_fps(video_path: str | Path, workdir: Path) -> Path:
    """Re-time a video to 25 fps (the GRID rate). A 25 fps file is used unchanged,
    so a GRID clip is cropped from exactly the pixels extract_mouths.py saw."""
    fps, _ = video_fps_and_frames(video_path)
    if abs(fps - VIDEO_FPS) < 0.01:
        return Path(video_path)
    out = workdir / "video_25fps.mp4"
    res = _run_ffmpeg(["-i", str(video_path), "-t", str(MAX_SECONDS), "-r", str(VIDEO_FPS),
                       "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
    if res.returncode != 0 or not out.exists():
        raise FriendlyError(f"Could not convert the video to 25 fps: {res.stderr.strip()[:200]}")
    return out


def extract_audio(video_path: str | Path, workdir: Path) -> Path | None:
    """The video's audio track as 16 kHz mono WAV, or None if it has no audio."""
    out = workdir / "audio_16k.wav"
    res = _run_ffmpeg(["-i", str(video_path), "-t", str(MAX_SECONDS), "-vn",
                       "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)])
    if res.returncode != 0 or not out.exists() or out.stat().st_size <= 44:
        return None
    return out


# ------------------------------------------------------------------ engine

class LipScribe:
    """A loaded checkpoint plus everything needed to caption one clip."""

    def __init__(self, checkpoint: str | Path, device: str | None = None,
                 decode: str | None = "beam", beam_width: int = 10,
                 face_model: str | Path = DEFAULT_FACE_MODEL, feature_stats: str | Path | None = None):
        from lipscribe.models import build_model
        from scripts.train import build_feature_step, load_checkpoint

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FriendlyError(f"Checkpoint not found: {self.checkpoint_path}")
        ckpt = load_checkpoint(self.checkpoint_path, map_location="cpu")
        self.cfg = ckpt["config"]
        if feature_stats is not None:
            self.cfg["data"]["feature_stats"] = str(feature_stats)
        self.model = build_model(self.cfg["model"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.streams = list(self.cfg["model"].get("streams", ["audio"]))
        self.uses_audio = "audio" in self.streams
        self.uses_video = "video" in self.streams or "visual" in self.streams
        self.prepare = build_feature_step(self.cfg, self.device)      # same as evaluate.py
        decode_cfg = dict(self.cfg.get("decode") or {})
        if decode:
            decode_cfg["method"] = decode
        decode_cfg.setdefault("beam_width", beam_width)
        self.decode_method = decode_cfg.get("method", "greedy")
        self.decode_fn = get_decoder(decode_cfg)
        self.face_model = Path(face_model)

    @property
    def name(self) -> str:
        return f"{self.cfg.get('experiment', self.cfg['model'].get('name'))} ({self.decode_method})"

    # -------------------------------------------------------------- inputs

    def _crops(self, video_path: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.face_model.exists():
            raise FriendlyError(f"Face Mesh model not found at {self.face_model} "
                                "(run python -m scripts.extract_mouths once to download it).")
        from lipscribe.preprocessing.crop import crop_video
        result = crop_video(video_path, self.face_model, target_frame_count=n_frames)
        return result.crops.astype(np.uint8), result.valid.astype(bool)

    def build_item(self, video_path: str | Path | None = None, audio_path: str | Path | None = None,
                   speech_start_s=None, n_frames: int | None = None, workdir: Path | None = None,
                   crop: bool = True) -> tuple[dict, list[str]]:
        """One dataset-style item from live inputs (same keys and audio handling as GridDataset)."""
        warnings: list[str] = []
        workdir = workdir or Path(tempfile.mkdtemp(prefix="lipscribe_"))
        if video_path is None and audio_path is None:
            raise FriendlyError("Please upload or record a video (or at least an audio clip).")

        video = to_25_fps(video_path, workdir) if video_path else None

        # Audio: an explicit file/microphone wins; otherwise the video's own track.
        wave = None
        if audio_path:
            wave = load_audio(audio_path)
        elif video is not None:
            track = extract_audio(video_path, workdir)
            if track is not None:
                wave = load_audio(track)
        if wave is not None and (wave.size == 0 or float(np.abs(wave).max()) < 1e-4):
            warnings.append("The audio is silent.")
            wave = None
        if wave is None and self.uses_audio:
            raise FriendlyError("This clip has no audio track. Record with the microphone on, "
                                "or upload the audio file as well.")

        # Number of 40 ms frames: from the video if there is one, otherwise from the audio.
        if n_frames is None:
            if video is not None:
                _, n_frames = video_fps_and_frames(video)
            if not n_frames:
                n_frames = math.ceil(len(wave) / SAMPLES_PER_FRAME)
        n_frames = int(min(n_frames, MAX_SECONDS * VIDEO_FPS))
        if n_frames < 10:
            raise FriendlyError("The clip is too short (under 0.4 s).")

        # Crops feed the visual stream; for an audio-only model they are only shown on the page.
        if video is not None and (self.uses_video or (crop and self.face_model.exists())):
            crops, valid = self._crops(video, n_frames)
        else:
            crops, valid = np.zeros((n_frames, 96, 96), np.uint8), np.zeros(n_frames, bool)

        if wave is None:
            wave = np.zeros(n_frames * SAMPLES_PER_FRAME, np.float32)
        wave = place_on_video_grid(wave, n_frames * SAMPLES_PER_FRAME, speech_start_s)
        wave = rms_normalise(np.asarray(wave, dtype=np.float32))

        item = {"utt_id": "live", "speaker": "live", "video": crops, "video_valid": valid,
                "audio": np.asarray(wave, dtype=np.float32), "snr_db": float("inf"),
                "text": "", "target": np.zeros(0, dtype=np.int64)}
        return item, warnings

    # -------------------------------------------------------------- model

    def caption_items(self, items: list[dict]) -> tuple[list[str], dict]:
        """Run evaluate.py's exact chain on prepared items."""
        from scripts.train import to_device
        batch = self.prepare(to_device(collate(items), self.device))
        use_amp = bool(self.cfg["train"].get("amp")) and self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=use_amp):
            out = self.model(batch)
        return self.decode_fn(out["log_probs"].float(), out["lengths"]), out

    def transcribe(self, video_path: str | Path | None = None, audio_path: str | Path | None = None,
                   speech_start_s=None, n_frames: int | None = None) -> Transcription:
        start = time.time()
        with tempfile.TemporaryDirectory(prefix="lipscribe_") as tmp:
            item, warnings = self.build_item(video_path, audio_path, speech_start_s, n_frames,
                                             workdir=Path(tmp))
        face = None
        if video_path is not None and not self.uses_video and not self.face_model.exists():
            warnings.append("Mouth crops not shown (Face Mesh model file missing).")
        elif video_path is not None:
            face = float(item["video_valid"].mean()) if len(item["video_valid"]) else 0.0
            if face < MIN_FACE_FRACTION:
                if self.uses_video and not self.uses_audio:
                    raise FriendlyError("No face was found in the video. Face the camera, "
                                        "with your mouth visible and the room well lit.")
                warnings.append(f"A face was found in only {face:.0%} of frames; "
                                "the caption relies on the audio.")
                if self.uses_video:
                    item["video_valid"][:] = False   # fused models fall back to the missing-video path
        captions, out = self.caption_items([item])
        gate = out.get("gate")
        if gate is not None:
            gate = gate.float().squeeze(0).cpu().numpy()
        caption = captions[0].strip()
        if not caption:
            warnings.append("No speech was recognised.")
        return Transcription(caption=caption, n_frames=len(item["audio"]) // SAMPLES_PER_FRAME,
                             face_fraction=face, crops=item["video"] if video_path else None,
                             gate=gate, warnings=warnings, seconds=time.time() - start)


def mouth_strip(crops: np.ndarray | None, n: int = 12) -> np.ndarray | None:
    """n evenly spaced crops side by side, for display: uint8 [96, 96*n]."""
    if crops is None or len(crops) == 0:
        return None
    idx = np.linspace(0, len(crops) - 1, num=min(n, len(crops))).round().astype(int)
    return np.concatenate([crops[i] for i in idx], axis=1)


_ENGINE: LipScribe | None = None


def transcribe(video_path: str | Path) -> str:
    """INTERFACES.md: transcribe(video_path) -> str, using $LIPSCRIBE_CHECKPOINT."""
    global _ENGINE
    if _ENGINE is None:
        ckpt = os.environ.get("LIPSCRIBE_CHECKPOINT")
        if not ckpt:
            raise FriendlyError("Set LIPSCRIBE_CHECKPOINT to a best.pt file first.")
        _ENGINE = LipScribe(ckpt)
    return _ENGINE.transcribe(video_path).caption

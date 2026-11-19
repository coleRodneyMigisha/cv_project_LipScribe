"""Live capture (mode B): camera and microphone threads, ring buffers and window extraction.

Camera frames and audio chunks are stamped with one shared monotonic clock
(``time.monotonic``). Each frame goes through Face Mesh on arrival; only the 96x96
grey mouth crop is kept. A window ending at ``t_end`` is built from the 75 frames
nearest to the 40 ms targets and the matching 48,000 audio samples.
Every class takes an injectable clock and source, so tests need no hardware.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from lipscribe.preprocessing.crop import crop_with_edge_padding, detect_lip_box

SAMPLE_RATE = 16_000
BUFFER_SECONDS = 4.0
WINDOW_SECONDS = 3.0
WINDOW_FRAMES = 75
FRAME_STEP = 0.04            # 25 fps targets
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
CROP_SIZE = 96
DEFAULT_FACE_MODEL = Path("models/face_landmarker.task")

Clock = Callable[[], float]


class LiveError(RuntimeError):
    """A capture problem with a message fit to show the user."""


# ------------------------------------------------------------------ buffers

class FrameRingBuffer:
    """Thread-safe store of (timestamp, mouth crop, valid) for the last ``duration`` seconds."""

    def __init__(self, duration: float = BUFFER_SECONDS):
        self.duration = float(duration)
        self._items: deque[tuple[float, np.ndarray, bool]] = deque()
        self._lock = threading.Lock()

    def append(self, t: float, crop: np.ndarray, valid: bool) -> None:
        """Add a frame and drop anything older than ``duration`` before it."""
        with self._lock:
            self._items.append((float(t), crop, bool(valid)))
            cutoff = self._items[-1][0] - self.duration
            while self._items[0][0] < cutoff:
                self._items.popleft()

    def snapshot(self) -> list[tuple[float, np.ndarray, bool]]:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def latest_time(self) -> float | None:
        with self._lock:
            return self._items[-1][0] if self._items else None


class AudioRingBuffer:
    """Thread-safe mono sample buffer holding exactly the last ``duration`` seconds.

    Chunks are contiguous; ``write`` is given the time at which the chunk's last
    sample was captured, which anchors the whole timeline.
    """

    def __init__(self, duration: float = BUFFER_SECONDS, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = int(sample_rate)
        self.capacity = int(round(duration * sample_rate))
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._count = 0                     # valid samples held (<= capacity)
        self._t_end: float | None = None    # time of the newest sample
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray, t_end: float) -> None:
        """Append mono samples whose last sample was captured at ``t_end``."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        if x.size >= self.capacity:
            x = x[-self.capacity:]
        with self._lock:
            n = x.size
            self._data = np.concatenate([self._data[n:], x])
            self._count = min(self.capacity, self._count + n)
            self._t_end = float(t_end)

    def __len__(self) -> int:
        with self._lock:
            return self._count

    def latest_time(self) -> float | None:
        with self._lock:
            return self._t_end

    def read_last(self) -> np.ndarray:
        """All held samples, oldest first."""
        with self._lock:
            return self._data[self.capacity - self._count:].copy()

    def window(self, t_end: float, n_samples: int = WINDOW_SAMPLES) -> tuple[np.ndarray, int]:
        """``n_samples`` ending at ``t_end``, zero-padded where nothing was captured.

        Returns the samples and how many of them are real captured audio.
        """
        out = np.zeros(n_samples, dtype=np.float32)
        with self._lock:
            if self._t_end is None or self._count == 0:
                return out, 0
            shift = int(round((self._t_end - t_end) * self.sample_rate))  # samples newer than t_end
            held = self._data[self.capacity - self._count:]
            stop = len(held) - shift          # exclusive end index into `held`
            start = stop - n_samples
            lo, hi = max(start, 0), min(stop, len(held))
            if hi <= lo:
                return out, 0
            out[lo - start:hi - start] = held[lo:hi]
            return out, hi - lo


# --------------------------------------------------------------- lip tracking

@dataclass
class LipObservation:
    """What the tracker returns for one frame."""
    crop: np.ndarray                 # uint8 [96, 96] grey mouth crop
    valid: bool                      # a face was found in this frame
    box: np.ndarray | None           # smoothed [cx, cy, side] used for the crop


def make_face_detector(model_path: str | Path = DEFAULT_FACE_MODEL,
                       clock: Clock = time.monotonic) -> Callable[[np.ndarray], np.ndarray | None]:
    """Face Mesh (video mode) as ``detect(frame_rgb) -> [cx, cy, side] or None``."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise LiveError(f"Face Mesh model not found at {model_path} "
                        "(run python -m scripts.extract_mouths once to download it).")
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from lipscribe.preprocessing.crop import get_lip_indices

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    lip_indices = get_lip_indices()
    origin = clock()
    last_ms = [-1]

    def detect(frame_rgb: np.ndarray) -> np.ndarray | None:
        ms = max(int((clock() - origin) * 1000), last_ms[0] + 1)   # strictly increasing
        last_ms[0] = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = landmarker.detect_for_video(image, ms)
        if not result.face_landmarks:
            return None
        h, w = frame_rgb.shape[:2]
        return detect_lip_box(result.face_landmarks[0], w, h, lip_indices)

    detect.close = landmarker.close  # type: ignore[attr-defined]
    return detect


class CausalLipTracker:
    """Per-frame lip box with causal EMA smoothing (no look-ahead) and 96x96 grey crops.

    ``detector(frame_rgb)`` returns ``[cx, cy, side]`` or ``None``. With no face the last
    smoothed box is held for the crop and the frame is marked invalid.
    ``alpha`` is the weight of the newest detection (1 = no smoothing).
    """

    def __init__(self, detector: Callable[[np.ndarray], np.ndarray | None], alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.detector = detector
        self.alpha = float(alpha)
        self._box: np.ndarray | None = None

    def reset(self) -> None:
        self._box = None

    def update(self, frame_bgr: np.ndarray) -> LipObservation:
        """Process one BGR frame, using only this and earlier frames."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw = self.detector(rgb)
        valid = raw is not None
        if valid:
            raw = np.asarray(raw, dtype=np.float32)
            self._box = raw if self._box is None else self.alpha * raw + (1.0 - self.alpha) * self._box
        if self._box is None:
            return LipObservation(np.zeros((CROP_SIZE, CROP_SIZE), np.uint8), False, None)
        crop = crop_with_edge_padding(rgb, *[float(v) for v in self._box])
        return LipObservation(crop, valid, self._box.copy())


# ------------------------------------------------------------------ capture

class CameraThread:
    """Reads frames in a thread, runs the tracker and stores crops stamped with ``clock()``.

    ``capture`` is any object with ``read()``, ``isOpened()`` and ``release()``
    (default: ``cv2.VideoCapture(index)`` opened in ``start``).
    """

    def __init__(self, buffer: FrameRingBuffer, tracker: CausalLipTracker | None = None,
                 capture=None, index: int = 0, clock: Clock = time.monotonic,
                 expected_fps: float = 30.0, max_failures: int = 50):
        self.buffer = buffer
        self.tracker = tracker
        self.clock = clock
        self.index = index
        self.expected_fps = expected_fps
        self.max_failures = max_failures
        self._capture = capture
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: tuple[float, np.ndarray, LipObservation | None] | None = None
        self.frames = 0
        self.failed_reads = 0
        self.dropped = 0
        self.error: str | None = None
        self._last_t: float | None = None

    def start(self) -> None:
        if self._capture is None:
            self._capture = cv2.VideoCapture(self.index)
        if not self._capture.isOpened():
            self._capture.release()
            raise LiveError(f"Could not open the camera (index {self.index}). "
                            "Check that it is plugged in and not used by another program.")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lipscribe-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> tuple[float, np.ndarray, LipObservation | None] | None:
        """Newest (time, BGR frame, observation) for the display."""
        with self._lock:
            return self._latest

    def _note_gap(self, t: float) -> None:
        if self._last_t is not None and self.expected_fps > 0:
            step = 1.0 / self.expected_fps
            missed = int(round((t - self._last_t) / step)) - 1
            if (t - self._last_t) > 1.5 * step and missed > 0:
                self.dropped += missed
        self._last_t = t

    def _run(self) -> None:
        consecutive = 0
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            t = self.clock()
            if not ok or frame is None:
                self.failed_reads += 1
                consecutive += 1
                if consecutive >= self.max_failures:
                    self.error = "The camera stopped delivering frames."
                    break
                time.sleep(0.005)
                continue
            consecutive = 0
            obs = self.tracker.update(frame) if self.tracker is not None else None
            crop = obs.crop if obs else np.zeros((CROP_SIZE, CROP_SIZE), np.uint8)
            self.buffer.append(t, crop, obs.valid if obs else False)
            self._note_gap(t)
            self.frames += 1
            with self._lock:
                self._latest = (t, frame, obs)


class AudioCapture:
    """16 kHz mono sounddevice InputStream feeding an ``AudioRingBuffer``.

    ``sounddevice`` is imported inside ``start`` so this module loads without PortAudio.
    """

    def __init__(self, buffer: AudioRingBuffer, clock: Clock = time.monotonic,
                 sample_rate: int = SAMPLE_RATE, device=None, blocksize: int = 1024):
        self.buffer = buffer
        self.clock = clock
        self.sample_rate = sample_rate
        self.device = device
        self.blocksize = blocksize
        self._stream = None
        self.status_flags = 0     # input overflows reported by PortAudio

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.status_flags += 1
        t_end = self.clock()
        try:   # shift to the time the last sample was actually captured
            lag = time_info.currentTime - (time_info.inputBufferAdcTime + frames / self.sample_rate)
            if 0.0 <= lag < 1.0:
                t_end -= lag
        except (AttributeError, TypeError):
            pass
        self.buffer.write(np.asarray(indata)[:, 0], t_end)

    def start(self) -> None:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise LiveError("Microphone capture needs the sounddevice package and PortAudio "
                            f"({exc}). Install them, or run without audio.") from exc
        try:
            self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                          blocksize=self.blocksize, device=self.device,
                                          callback=self._callback)
            self._stream.start()
        except Exception as exc:  # PortAudioError and friends
            self._stream = None
            raise LiveError(f"Could not open the microphone: {exc}") from exc

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# ------------------------------------------------------------------- windows

def extract_window(frames: FrameRingBuffer, audio: AudioRingBuffer, t_end: float,
                   n_frames: int = WINDOW_FRAMES, step: float = FRAME_STEP,
                   n_samples: int = WINDOW_SAMPLES) -> dict:
    """Model input item for the window ending at ``t_end``.

    Frame k is the stored frame nearest to the target ``t_end - 3.0 + 0.04*k`` (k = 0..74). A frame farther than one step from its
    target, or without a face, is marked invalid (and its crop is black).
    """
    stored = frames.snapshot()
    video = np.zeros((n_frames, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    valid = np.zeros(n_frames, dtype=bool)
    times = np.full(n_frames, np.nan)
    targets = t_end - WINDOW_SECONDS + step * np.arange(n_frames)
    if stored:
        ts = np.asarray([s[0] for s in stored])
        idx = np.clip(np.searchsorted(ts, targets), 1, len(ts) - 1) if len(ts) > 1 else np.zeros(n_frames, int)
        if len(ts) > 1:
            left = idx - 1
            idx = np.where(np.abs(ts[left] - targets) <= np.abs(ts[idx] - targets), left, idx)
        for k, i in enumerate(idx):
            if abs(ts[i] - targets[k]) > step:
                continue
            _, crop, ok = stored[i]
            times[k] = ts[i]
            if ok:
                video[k] = crop
                valid[k] = True
    wave, real = audio.window(t_end, n_samples)
    return {"utt_id": "live", "speaker": "live", "video": video, "video_valid": valid,
            "audio": wave, "snr_db": float("inf"), "text": "", "target": np.zeros(0, dtype=np.int64),
            "t_end": float(t_end), "frame_times": times, "audio_real_samples": int(real)}


# ------------------------------------------------------------------- display

def compose_view(frame_bgr: np.ndarray | None, obs: LipObservation | None, item: dict,
                 strip_n: int = 12, width: int = 640) -> np.ndarray:
    """Camera view with the lip box above a strip of mouth crops from the window."""
    if frame_bgr is None:
        view = np.zeros((int(width * 0.75), width, 3), dtype=np.uint8)
        cv2.putText(view, "waiting for camera", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        view = frame_bgr.copy()
        if obs is not None and obs.box is not None:
            cx, cy, side = [float(v) for v in obs.box]
            colour = (0, 200, 0) if obs.valid else (0, 0, 255)
            cv2.rectangle(view, (int(cx - side / 2), int(cy - side / 2)),
                          (int(cx + side / 2), int(cy + side / 2)), colour, 2)
        scale = width / view.shape[1]
        view = cv2.resize(view, (width, max(1, int(view.shape[0] * scale))))
    video, valid = item["video"], item["video_valid"]
    picks = np.linspace(0, len(video) - 1, strip_n).astype(int)
    tile = width // strip_n
    strip = np.zeros((tile, tile * strip_n, 3), dtype=np.uint8)
    for j, i in enumerate(picks):
        t = cv2.cvtColor(cv2.resize(video[i], (tile, tile)), cv2.COLOR_GRAY2BGR)
        if not valid[i]:
            cv2.rectangle(t, (0, 0), (tile - 1, tile - 1), (0, 0, 255), 2)
        strip[:, j * tile:(j + 1) * tile] = t
    out = np.zeros((view.shape[0] + tile, width, 3), dtype=np.uint8)
    out[:view.shape[0]] = view
    out[view.shape[0]:, :strip.shape[1]] = strip
    return out


@dataclass
class DisplayStats:
    shown: int = 0
    quit_by_key: bool = False
    elapsed: float = 0.0
    extra: dict = field(default_factory=dict)


def run_display(camera: CameraThread, frames: FrameRingBuffer, audio: AudioRingBuffer,
                seconds: float | None = None, clock: Clock = time.monotonic,
                show: Callable[[str, np.ndarray], None] = cv2.imshow,
                wait_key: Callable[[int], int] = cv2.waitKey,
                sleep: Callable[[float], None] = time.sleep, fps: float = 15.0,
                title: str = "LipScribe live (q to quit)") -> DisplayStats:
    """Show the current window until ``seconds`` have passed or q is pressed."""
    stats = DisplayStats()
    start = clock()
    while True:
        now = clock()
        if seconds is not None and now - start >= seconds:
            break
        newest = frames.latest_time()
        if newest is not None:
            item = extract_window(frames, audio, newest)
            latest = camera.latest()
            frame, obs = (latest[1], latest[2]) if latest else (None, None)
            show(title, compose_view(frame, obs, item))
            stats.shown += 1
        key = wait_key(1) & 0xFF
        if key == ord("q"):
            stats.quit_by_key = True
            break
        if not camera.running and camera.error:
            break
        sleep(max(0.0, 1.0 / fps - (clock() - now)))
    stats.elapsed = clock() - start
    return stats

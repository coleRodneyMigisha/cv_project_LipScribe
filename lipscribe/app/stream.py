"""Streaming input for the demo (F-05): bytes arrive in pieces, captions leave in batches.

    page bytes -> GrowingFile -> MediaReader (PyAV, a thread) -> Timeline (16 kHz audio + 25 fps frames)
                                                                    |
                         Silero VAD -> speech segments (<= 4 s) -> Captioner -> events for the page

* **GrowingFile** is a file whose bytes arrive out of order. A read of bytes that have not
  arrived yet blocks and records the offset it waits for (``wanted``), so the server can ask the
  page for exactly those bytes next. That is how an MP4 whose index (``moov``) sits at the end
  still starts at once: PyAV seeks to the end, the read blocks, the page sends the tail, and
  decoding then runs from the start while the rest uploads. Live recordings (no size, no seek)
  use the same class as a pipe that ends at ``finish()``.
* **MediaReader** decodes audio to 16 kHz mono float and samples video at 25 fps with the frame
  rule of ``scripts.flix_recrop.read_frames`` (the first frame at or after each 40 ms tick, with
  20 ms slack), downscaled to <= 640 px wide. It blocks when it is more than ``ahead_s`` seconds
  ahead of the captioner, so memory stays bounded for a two-hour film.
* **Timeline** keeps only what has not been captioned yet.
"""
from __future__ import annotations

import bisect
import io
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

SR, FPS, SPF = 16000, 25, 640


# ------------------------------------------------------------------ bytes

class GrowingFile(io.RawIOBase):
    """A file filled by ``write_at`` in any order; reads block until their bytes arrive."""

    def __init__(self, size: int | None = None, timeout_s: float = 600.0):
        self.size = size                     # None: live stream, length known only at finish()
        self.buf = bytearray(size or 0)
        self.have: list[list[int]] = []      # sorted, disjoint [start, end) ranges received
        self.pos = 0
        self.wanted: int | None = None       # offset a blocked read is waiting for
        self.finished = False
        self.aborted = False
        self.timeout_s = timeout_s
        self.cond = threading.Condition(threading.RLock())

    # -- writer side
    def write_at(self, offset: int, data: bytes) -> None:
        with self.cond:
            end = offset + len(data)
            if self.size is None:
                if end > len(self.buf):
                    self.buf.extend(b"\0" * (end - len(self.buf)))
            elif end > self.size:
                raise ValueError("chunk past the end of the file")
            self.buf[offset:end] = data
            self._add(offset, end)
            self.cond.notify_all()

    def finish(self) -> None:
        with self.cond:
            self.finished = True
            if self.size is None:
                self.size = len(self.buf)
            self.cond.notify_all()

    def abort(self) -> None:
        with self.cond:
            self.aborted = True
            self.cond.notify_all()

    def _add(self, a: int, b: int) -> None:
        r = self.have
        i = bisect.bisect_left(r, [a, a])
        if i > 0 and r[i - 1][1] >= a:
            i -= 1
        j = i
        while j < len(r) and r[j][0] <= b:
            a, b = min(a, r[j][0]), max(b, r[j][1])
            j += 1
        r[i:j] = [[a, b]]

    def received(self) -> int:
        return sum(b - a for a, b in self.have)

    def received_prefix(self) -> int:
        return self.have[0][1] if self.have and self.have[0][0] == 0 else 0

    def missing(self, start: int = 0) -> list[tuple[int, int]]:
        """Missing [a, b) ranges from ``start`` to the end (size must be known)."""
        out, cur = [], start
        for a, b in self.have:
            if b <= cur:
                continue
            if a > cur:
                out.append((cur, a))
            cur = max(cur, b)
        if self.size is not None and cur < self.size:
            out.append((cur, self.size))
        return out

    def covered(self, a: int, b: int) -> bool:
        """True if every byte of [a, b) has arrived."""
        with self.cond:
            return any(x <= a and b <= y for x, y in self.have)

    def _avail(self, pos: int) -> int:
        """Contiguous bytes available from pos."""
        for a, b in self.have:
            if a <= pos < b:
                return b - pos
        return 0

    # -- reader side (PyAV)
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return self.size is not None and not self._live

    _live = False

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 1:
            offset += self.pos
        elif whence == 2:
            offset += self.size or 0
        self.pos = max(0, offset)
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:
        with self.cond:
            deadline = time.time() + self.timeout_s
            while True:
                if self.aborted:
                    raise OSError("upload aborted")
                if self.size is not None and self.pos >= self.size and (self.finished or not self._live):
                    return 0
                n = self._avail(self.pos)
                if n:
                    break
                if self.finished:
                    return 0
                self.wanted = self.pos
                if not self.cond.wait(timeout=min(1.0, max(0.0, deadline - time.time()))) and time.time() > deadline:
                    raise OSError("timed out waiting for upload bytes")
            self.wanted = None
            n = min(n, len(b))
            b[:n] = self.buf[self.pos:self.pos + n]
            self.pos += n
            return n


class LiveStream(GrowingFile):
    """MediaRecorder chunks in order; not seekable (PyAV reads it like a pipe)."""
    _live = True

    def __init__(self):
        super().__init__(size=None)

    def seekable(self) -> bool:
        return False

    def append(self, data: bytes) -> None:
        with self.cond:
            self.write_at(len(self.buf), data)


# ------------------------------------------------------------------ decoded media

class Timeline:
    """Decoded audio (16 kHz float32) and 25 fps frames, from ``base`` onwards (older data dropped)."""

    def __init__(self):
        self.cond = threading.Condition()
        self.audio = np.zeros(0, np.float32)
        self.a0 = 0                     # sample index of audio[0]
        self.frames: dict[int, np.ndarray] = {}
        self.n_frames = 0               # frames 0..n_frames-1 decided (some may be dropped)
        self.last_frame: np.ndarray | None = None
        self.eof = False
        self.error: str | None = None
        self.duration: float | None = None
        self.has_audio = True
        self.has_video = True
        self.consumed_s = 0.0           # captioner position (for back-pressure)

    @property
    def audio_end_s(self) -> float:
        return (self.a0 + len(self.audio)) / SR

    @property
    def end_s(self) -> float:
        """Seconds of media available in both streams."""
        ends = []
        if self.has_audio:
            ends.append(self.audio_end_s)
        if self.has_video:
            ends.append(self.n_frames / FPS)
        if not ends:
            return 0.0
        return max(ends) if self.eof else min(ends)   # at the end, a shorter stream is padded

    def add_audio(self, start_sample: int, x: np.ndarray) -> None:
        with self.cond:
            end = self.a0 + len(self.audio)
            if start_sample > end:                       # gap: fill with silence
                x = np.concatenate([np.zeros(start_sample - end, np.float32), x])
            elif start_sample < end:                     # overlap: keep the new tail only
                x = x[end - start_sample:]
            if len(x):
                self.audio = np.concatenate([self.audio, x.astype(np.float32, copy=False)])
            self.cond.notify_all()

    def add_frame(self, k: int, rgb: np.ndarray) -> None:
        with self.cond:
            while self.n_frames < k:                     # a missing tick repeats the last frame
                if self.last_frame is not None:
                    self.frames[self.n_frames] = self.last_frame
                self.n_frames += 1
            self.frames[k] = rgb
            self.last_frame = rgb
            self.n_frames = k + 1
            self.cond.notify_all()

    def get_audio(self, t0: float, n: int) -> np.ndarray:
        """Exactly n samples from t0 (zeros where nothing was decoded)."""
        with self.cond:
            i0 = int(round(t0 * SR)) - self.a0
            out = np.zeros(n, np.float32)
            a, b = max(0, i0), min(len(self.audio), i0 + n)
            if b > a:
                out[a - i0:b - i0] = self.audio[a:b]
            return out

    def get_frames(self, k0: int, n: int) -> list[np.ndarray] | None:
        with self.cond:
            out, last = [], None
            for k in range(k0, k0 + n):
                f = self.frames.get(k, last)
                if f is None:
                    f = next((self.frames[j] for j in range(k, min(self.n_frames, k0 + n)) if j in self.frames), None)
                if f is None:
                    return None
                out.append(f); last = f
            return out

    def drop_before(self, t: float) -> None:
        with self.cond:
            cut = int(t * SR) - self.a0
            if cut > 0:
                self.audio = self.audio[cut:].copy()
                self.a0 += cut
            k = int(t * FPS)
            for j in [j for j in self.frames if j < k]:
                del self.frames[j]
            self.consumed_s = max(self.consumed_s, t)
            self.cond.notify_all()

    def wait_until(self, t: float, timeout: float = 1.0) -> bool:
        with self.cond:
            if self.end_s >= t or self.eof:
                return True
            self.cond.wait(timeout)
            return self.end_s >= t or self.eof


class MediaReader(threading.Thread):
    """Decode a (growing) file into a Timeline with PyAV."""

    def __init__(self, src: GrowingFile, timeline: Timeline, fmt: str | None = None,
                 ahead_s: float = 25.0, max_duration_s: float = 3 * 3600):
        super().__init__(daemon=True)
        self.src, self.tl, self.fmt, self.ahead_s, self.max_duration_s = src, timeline, fmt, ahead_s, max_duration_s

    def run(self) -> None:
        import av

        from lipscribe.app.realworld import downscale
        tl = self.tl
        try:
            live = isinstance(self.src, LiveStream)       # a pipe: probe little, or captions lag 5 s
            probe = {"probesize": "65536", "analyzeduration": "500000"} if live else \
                {"probesize": "5000000", "analyzeduration": "5000000"}
            container = av.open(self.src, mode="r", format=self.fmt, options=probe)
            a_st = container.streams.audio[0] if container.streams.audio else None
            v_st = container.streams.video[0] if container.streams.video else None
            tl.has_audio, tl.has_video = a_st is not None, v_st is not None
            if container.duration:
                tl.duration = container.duration / 1e6
            if tl.duration and tl.duration > self.max_duration_s:
                raise ValueError(f"the file is {tl.duration / 3600:.1f} h long (limit {self.max_duration_s / 3600:.0f} h)")
            if v_st is not None and not live:
                v_st.thread_type = "AUTO"      # frame threads hold frames back: only for files
            origin = None
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
            next_k, apos = 0, None
            streams = [s for s in (a_st, v_st) if s is not None]
            for packet in container.demux(*streams):
                for frame in packet.decode():
                    t = frame.time
                    if t is None:
                        continue
                    if origin is None:
                        origin = t
                    t -= origin
                    if packet.stream is a_st:
                        if apos is None:                 # first audio sample's place; then contiguous
                            apos = max(0, int(round(t * SR)))
                        for rf in resampler.resample(frame):
                            x = rf.to_ndarray().reshape(-1).astype(np.float32)
                            tl.add_audio(apos, x)
                            apos += len(x)
                    else:
                        # flix_recrop.read_frames: frame for tick k = first frame with t + 0.02 >= k / 25
                        if t + 0.02 < next_k / FPS:
                            continue
                        img = None
                        while t + 0.02 >= next_k / FPS:
                            if img is None:
                                img = downscale(frame.to_ndarray(format="rgb24"))
                            tl.add_frame(next_k, img)
                            next_k += 1
                    self._backpressure(t)
            if apos is not None:
                for rf in resampler.resample(None):
                    x = rf.to_ndarray().reshape(-1).astype(np.float32)
                    tl.add_audio(apos, x)
                    apos += len(x)
            container.close()
        except Exception as e:                   # noqa: BLE001 - reported to the page
            tl.error = f"{type(e).__name__}: {e}"[:300]
        finally:
            with tl.cond:
                tl.eof = True
                if tl.duration is None:
                    tl.duration = tl.end_s
                tl.cond.notify_all()

    def _backpressure(self, t: float) -> None:
        tl = self.tl
        with tl.cond:
            while t - tl.consumed_s > self.ahead_s and not self.src.aborted:
                tl.cond.wait(0.5)


# ------------------------------------------------------------------ speech segments

class VAD:
    """Silero VAD (pip silero-vad, bundled model; CPU)."""

    def __init__(self):
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.model = load_silero_vad()
        self._ts = get_speech_timestamps
        self.lock = threading.Lock()

    def speech(self, wave: np.ndarray, max_s: float = 4.0, min_silence_ms: int = 250) -> list[tuple[float, float]]:
        import torch
        if len(wave) < SR // 4:
            return []
        with self.lock:
            ts = self._ts(torch.from_numpy(np.ascontiguousarray(wave)), self.model, sampling_rate=SR,
                          threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=min_silence_ms,
                          max_speech_duration_s=max_s, speech_pad_ms=30)
        return [(d["start"] / SR, d["end"] / SR) for d in ts]


def noise_estimate_db(wave: np.ndarray, speech: list[tuple[float, float]], t0: float = 0.0) -> float | None:
    """Rough SNR: power inside speech vs outside it (the page's room-noise gauge)."""
    if not len(wave) or not speech:
        return None
    m = np.zeros(len(wave), bool)
    for a, b in speech:
        m[max(0, int((a - t0) * SR)):max(0, int((b - t0) * SR))] = True
    if m.sum() < SR // 4 or (~m).sum() < SR // 10:
        return 30.0 if m.sum() >= SR // 4 else None
    ps, pn = float(np.mean(wave[m] ** 2)), float(np.mean(wave[~m] ** 2)) + 1e-10
    return float(np.clip(10 * math.log10(max(ps - pn, 1e-10) / pn), -15, 40))


@dataclass
class Span:
    start: float          # speech start (s), for the subtitle cue
    end: float            # speech end (s)
    w0: float             # model window start: speech widened by PAD_S (as Flix v5)
    n: int                # model window length in frames


PAD_S = 0.32


def window(start: float, end: float, limit: float | None) -> Span:
    w0 = max(0.0, start - PAD_S)
    w1 = end + PAD_S if limit is None else min(limit, end + PAD_S)
    k0 = int(round(w0 * FPS))
    n = max(1, int(round(w1 * FPS)) - k0)
    return Span(start, end, k0 / FPS, n)

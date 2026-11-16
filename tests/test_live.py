"""Tests for live capture (W4-10): buffers, window extraction, tracker, capture threads."""
from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from lipscribe.app import live
from lipscribe.app.live import (AudioCapture, AudioRingBuffer, CameraThread, CausalLipTracker,
                                FrameRingBuffer, LiveError, extract_window, run_display)

SR = 16_000


class FakeClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def crop_with(value: int) -> np.ndarray:
    return np.full((96, 96), value, dtype=np.uint8)


def fill_frames(times, valid=lambda i: True) -> FrameRingBuffer:
    buf = FrameRingBuffer()
    for i, t in enumerate(times):
        buf.append(t, crop_with(i % 250 + 1), valid(i))
    return buf


# ------------------------------------------------------------------ buffers

def test_frame_buffer_keeps_last_four_seconds():
    buf = fill_frames([i * 0.04 for i in range(400)])      # 16 s at 25 fps
    items = buf.snapshot()
    newest = items[-1][0]
    assert all(t >= newest - 4.0 - 1e-9 for t, _, _ in items)
    assert len(items) == 101                                # 4.0 s / 0.04 + 1 inclusive
    assert items[0][0] > newest - 4.0 - 0.04


def test_audio_buffer_keeps_exactly_last_four_seconds():
    buf = AudioRingBuffer()
    data = np.arange(10 * SR, dtype=np.float32)
    for i in range(0, len(data), 1000):
        buf.write(data[i:i + 1000], t_end=(i + 1000) / SR)
    held = buf.read_last()
    assert len(held) == 4 * SR
    assert held[0] == data[-4 * SR] and held[-1] == data[-1]


def test_audio_buffer_oversized_write():
    buf = AudioRingBuffer()
    buf.write(np.arange(5 * SR, dtype=np.float32), 5.0)
    assert len(buf) == 4 * SR and buf.read_last()[-1] == 5 * SR - 1


def test_buffers_thread_safe_under_producer():
    frames, audio = FrameRingBuffer(), AudioRingBuffer()
    stop = threading.Event()

    def produce():
        for i in range(2000):
            frames.append(i * 0.01, crop_with(1), True)
            audio.write(np.full(160, i, np.float32), i * 0.01)
        stop.set()

    th = threading.Thread(target=produce)
    th.start()
    while not stop.is_set():
        snap = frames.snapshot()
        ts = [s[0] for s in snap]
        assert ts == sorted(ts)
        w, _ = audio.window(frames.latest_time() or 0.0)
        assert w.shape == (48_000,)
    th.join()
    assert len(audio) == 4 * SR
    assert len(frames) in (400, 401)     # float rounding at the 4 s edge


# ------------------------------------------------------------------- window

def test_window_exact_30fps_shapes_and_nearest():
    t_end = 10.0
    times = [t_end - 3.5 + i / 30 for i in range(int(4.0 * 30))]
    frames = fill_frames(times)
    audio = AudioRingBuffer()
    audio.write(np.arange(4 * SR, dtype=np.float32), t_end)
    item = extract_window(frames, audio, t_end)
    assert item["video"].shape == (75, 96, 96) and item["video"].dtype == np.uint8
    assert item["video_valid"].shape == (75,) and item["video_valid"].all()
    assert item["audio"].shape == (48_000,) and item["audio"].dtype == np.float32
    assert item["audio"][-1] == 4 * SR - 1 and item["audio"][0] == 4 * SR - 48_000
    targets = t_end - 3.0 + 0.04 * np.arange(75)
    assert np.all(np.abs(item["frame_times"] - targets) <= 1 / 60 + 1e-9)


def test_window_picks_nearest_with_jitter():
    rng = np.random.default_rng(0)
    base = np.arange(0, 6.0, 1 / 30)
    times = base + rng.uniform(-0.008, 0.008, base.size)
    frames = fill_frames(times)
    t_end = float(times[-1])
    item = extract_window(frames, AudioRingBuffer(), t_end)
    kept = np.asarray([s[0] for s in frames.snapshot()])
    targets = t_end - 3.0 + 0.04 * np.arange(75)
    for k in range(75):
        best = kept[np.argmin(np.abs(kept - targets[k]))]
        assert item["frame_times"][k] == best
    assert item["video_valid"].all()


def test_window_short_buffer_marks_invalid_and_pads_audio():
    t_end = 5.0
    frames = fill_frames([t_end - 1.0 + i * 0.04 for i in range(26)])    # only 1 s
    audio = AudioRingBuffer()
    audio.write(np.ones(SR, np.float32), t_end)
    item = extract_window(frames, audio, t_end)
    assert item["video"].shape[0] == 75 and len(item["audio"]) == 48_000
    assert not item["video_valid"][:49].any() and item["video_valid"][50:].all()
    assert (item["video"][:49] == 0).all()
    assert item["audio_real_samples"] == SR
    assert (item["audio"][:32_000] == 0).all() and (item["audio"][32_000:] == 1).all()


def test_window_empty_buffers():
    item = extract_window(FrameRingBuffer(), AudioRingBuffer(), 3.0)
    assert not item["video_valid"].any() and item["audio_real_samples"] == 0


def test_window_no_face_frames_invalid():
    times = [i * 0.04 for i in range(100)]
    frames = fill_frames(times, valid=lambda i: i % 2 == 0)
    item = extract_window(frames, AudioRingBuffer(), times[-1])
    assert 30 < item["video_valid"].sum() < 45
    assert (item["video"][~item["video_valid"]] == 0).all()


# ------------------------------------------------------------------ tracker

def scripted_detector(boxes):
    it = iter(boxes)
    return lambda rgb: next(it)


def test_ema_is_causal():
    rng = np.random.default_rng(1)
    boxes = [np.array([48 + rng.normal(0, 3), 48 + rng.normal(0, 3), 40.0], np.float32) for _ in range(20)]
    frame = np.random.default_rng(2).integers(0, 255, (96, 96, 3), dtype=np.uint8)

    def run(bs):
        tr = CausalLipTracker(scripted_detector(bs), alpha=0.4)
        return [tr.update(frame) for _ in bs]

    a = run(boxes)
    changed = boxes[:10] + [b + 25 for b in boxes[10:]]
    b = run(changed)
    for i in range(10):
        assert np.array_equal(a[i].box, b[i].box) and np.array_equal(a[i].crop, b[i].crop)
    assert not np.allclose(a[12].box, b[12].box)


def test_ema_smooths_noise_and_alpha_one_is_raw():
    rng = np.random.default_rng(3)
    boxes = [np.array([50 + rng.normal(0, 4), 50.0, 40.0], np.float32) for _ in range(200)]
    frame = np.zeros((100, 100, 3), np.uint8)
    tr = CausalLipTracker(scripted_detector(boxes), alpha=0.3)
    sm = np.array([tr.update(frame).box[0] for _ in boxes])
    assert sm[20:].std() < 0.7 * np.std([b[0] for b in boxes[20:]])
    tr1 = CausalLipTracker(scripted_detector(boxes[:5]), alpha=1.0)
    for b in boxes[:5]:
        assert np.allclose(tr1.update(frame).box, b)
    with pytest.raises(ValueError):
        CausalLipTracker(lambda r: None, alpha=0.0)


def test_tracker_no_face_holds_box_and_marks_invalid():
    frame = np.full((100, 100, 3), 120, np.uint8)
    tr = CausalLipTracker(scripted_detector([None, np.array([50, 50, 30], np.float32), None]))
    first = tr.update(frame)
    assert not first.valid and first.box is None and first.crop.shape == (96, 96) and not first.crop.any()
    second = tr.update(frame)
    assert second.valid and second.crop.dtype == np.uint8 and second.crop.shape == (96, 96)
    third = tr.update(frame)
    assert not third.valid and np.allclose(third.box, second.box)


# ------------------------------------------------------------------ capture

class FakeCapture:
    def __init__(self, n_frames=30, opened=True):
        self.n = n_frames
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.n <= 0:
            time.sleep(0.001)
            return False, None
        self.n -= 1
        time.sleep(0.002)
        return True, np.full((100, 100, 3), 90, np.uint8)

    def release(self):
        self.released = True


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        time.sleep(0.005)
    return cond()


def test_camera_thread_uses_shared_clock_and_tracker():
    ticks = iter(range(10_000))

    def clk():
        return 50.0 + next(ticks) * 0.04

    box = np.array([50, 50, 30], np.float32)
    cam = CameraThread(FrameRingBuffer(), CausalLipTracker(lambda rgb: box), capture=FakeCapture(10),
                       clock=clk, max_failures=5)
    cam.start()
    assert wait_for(lambda: cam.frames == 10)
    cam.stop()
    ts = [s[0] for s in cam.buffer.snapshot()]
    assert len(ts) == 10 and all(t >= 50.0 for t in ts) and ts == sorted(ts)
    assert all(s[2] for s in cam.buffer.snapshot())
    assert cam.latest() is not None and cam.dropped == 0


def test_camera_thread_counts_dropped_and_stops_on_failure():
    times = iter([0.0, 0.04, 0.08, 0.28, 0.32] + [1.0] * 100)
    cam = CameraThread(FrameRingBuffer(), None, capture=FakeCapture(5), clock=lambda: next(times),
                       expected_fps=25.0, max_failures=3)
    cam.start()
    assert wait_for(lambda: cam.error is not None)
    cam.stop()
    assert cam.dropped == 4                      # 0.08 -> 0.28 is 5 steps, 4 missing
    assert cam.error and cam.failed_reads >= 3


def test_camera_not_opened_is_friendly():
    cap = FakeCapture(opened=False)
    with pytest.raises(LiveError, match="camera"):
        CameraThread(FrameRingBuffer(), None, capture=cap).start()
    assert cap.released


class FakeStream:
    def __init__(self, **kw):
        self.kw = kw
        self.started = self.stopped = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def fake_sounddevice(monkeypatch, fail=False):
    mod = types.ModuleType("sounddevice")
    made = []

    def InputStream(**kw):
        if fail:
            raise RuntimeError("no device")
        s = FakeStream(**kw)
        made.append(s)
        return s

    mod.InputStream = InputStream
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return made


def test_audio_capture_callback_with_fake_sounddevice(monkeypatch):
    made = fake_sounddevice(monkeypatch)
    clock = FakeClock(20.0)
    buf = AudioRingBuffer()
    cap = AudioCapture(buf, clock=clock)
    cap.start()
    s = made[0]
    assert s.started and s.kw["samplerate"] == 16_000 and s.kw["channels"] == 1
    block = np.arange(1600, dtype=np.float32).reshape(-1, 1)
    s.kw["callback"](block, 1600, types.SimpleNamespace(currentTime=5.0, inputBufferAdcTime=4.8), None)
    # lag = 5.0 - (4.8 + 0.1) = 0.1 s
    assert buf.latest_time() == pytest.approx(19.9)
    s.kw["callback"](block, 1600, None, "overflow")
    assert buf.latest_time() == 20.0 and cap.status_flags == 1 and len(buf) == 3200
    cap.stop()
    assert s.stopped and s.closed


def test_audio_capture_errors_are_friendly(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)    # import raises ImportError
    with pytest.raises(LiveError, match="sounddevice"):
        AudioCapture(AudioRingBuffer()).start()
    fake_sounddevice(monkeypatch, fail=True)
    with pytest.raises(LiveError, match="microphone"):
        AudioCapture(AudioRingBuffer()).start()


def test_module_import_does_not_need_sounddevice():
    assert "sounddevice" not in vars(live)


def test_shared_clock_aligns_audio_and_video(monkeypatch):
    made = fake_sounddevice(monkeypatch)
    clock = FakeClock(10.0)
    frames, audio = FrameRingBuffer(), AudioRingBuffer()
    AudioCapture(audio, clock=clock).start()
    for i in range(110):
        clock.t = 10.0 + i * 0.04
        frames.append(clock.t, crop_with(5), True)
        made[0].kw["callback"](np.full((640, 1), i, np.float32), 640, None, None)
    item = extract_window(frames, audio, clock.t)
    assert item["video_valid"].all() and item["audio_real_samples"] == 48_000
    assert item["audio"][-1] == 109 and item["audio"][0] == 109 - 74


# ------------------------------------------------------------------ display

def test_run_display_quits_on_q_without_window():
    clock = FakeClock(0.0)
    frames, audio = fill_frames([0.0, 0.04, 0.08]), AudioRingBuffer()
    cam = CameraThread(frames, None, capture=FakeCapture())
    shown, keys = [], iter([-1, -1, ord("q"), -1])

    def sleep(dt):
        clock.t += 0.1

    stats = run_display(cam, frames, audio, clock=clock, show=lambda t, im: shown.append(im.shape),
                        wait_key=lambda ms: next(keys), sleep=sleep)
    assert stats.quit_by_key and stats.shown == 3 and shown[0][2] == 3


def test_run_display_stops_after_seconds():
    clock = FakeClock(0.0)
    frames = fill_frames([0.0, 0.04])
    cam = CameraThread(frames, None, capture=FakeCapture())
    stats = run_display(cam, frames, AudioRingBuffer(), seconds=1.0, clock=clock,
                        show=lambda t, im: None, wait_key=lambda ms: -1,
                        sleep=lambda dt: setattr(clock, "t", clock.t + 0.25))
    assert not stats.quit_by_key and stats.elapsed >= 1.0

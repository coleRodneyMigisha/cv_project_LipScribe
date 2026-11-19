"""lipscribe.app.stream: the growing upload file and segment windows (F-05). No models needed."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from lipscribe.app.stream import (
    FPS,
    PAD_S,
    GrowingFile,
    LiveStream,
    Timeline,
    noise_estimate_db,
    window,
)


def test_out_of_order_pieces_merge_and_report_missing():
    f = GrowingFile(size=100)
    f.write_at(60, b"x" * 40)
    f.write_at(0, b"a" * 20)
    assert f.missing() == [(20, 60)]
    assert f.covered(60, 100) and not f.covered(10, 30)
    f.write_at(20, b"b" * 40)
    assert f.missing() == [] and f.received() == 100
    assert f.have == [[0, 100]]


def test_blocked_read_records_the_wanted_offset_then_returns_the_bytes():
    f = GrowingFile(size=10, timeout_s=5)
    f.write_at(0, b"0123")
    f.seek(6)
    out = {}

    def reader():
        out["data"] = f.read(4)

    t = threading.Thread(target=reader)
    t.start()
    for _ in range(100):
        if f.wanted is not None:
            break
        time.sleep(0.01)
    assert f.wanted == 6                     # the server would ask the page for these bytes next
    f.write_at(4, b"456789")
    t.join(2)
    assert out["data"] == b"6789" and f.wanted is None


def test_read_past_the_end_is_eof():
    f = GrowingFile(size=4)
    f.write_at(0, b"abcd")
    assert f.read(10) == b"abcd" and f.read(10) == b""


def test_live_stream_is_a_pipe_that_ends_at_finish():
    s = LiveStream()
    assert not s.seekable()
    s.append(b"ab")
    s.append(b"cd")
    assert s.read(3) == b"abc"
    s.finish()
    assert s.read(10) == b"d" and s.read(10) == b""


def test_abort_wakes_a_blocked_reader():
    f = GrowingFile(size=10, timeout_s=5)
    err = {}

    def reader():
        try:
            f.read(4)
        except OSError as e:
            err["e"] = e

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    f.abort()
    t.join(2)
    assert "e" in err


def test_window_pads_speech_like_flix_v5_and_stops_at_the_end():
    w = window(1.0, 2.0, limit=None)
    assert w.start == 1.0 and w.end == 2.0
    assert w.w0 == pytest.approx(1.0 - PAD_S, abs=1 / FPS)
    assert w.n == round((2.0 + PAD_S) * FPS) - round((1.0 - PAD_S) * FPS)
    w = window(0.1, 2.9, limit=3.0)          # clipped at the start of the file and at its end
    assert w.w0 == 0.0 and w.n == 75


def test_timeline_pads_gaps_and_drops_old_data():
    tl = Timeline()
    tl.add_audio(0, np.ones(16000, np.float32))
    tl.add_audio(32000, np.ones(16000, np.float32))     # a 1 s gap becomes silence
    assert tl.audio_end_s == pytest.approx(3.0)
    assert tl.get_audio(1.0, 16000).max() == 0.0
    frame = np.zeros((4, 4, 3), np.uint8)
    tl.add_frame(0, frame)
    tl.add_frame(3, frame)                               # missing ticks repeat the last frame
    assert tl.n_frames == 4 and len(tl.get_frames(0, 4)) == 4
    tl.drop_before(2.0)
    assert tl.a0 == 32000 and tl.get_audio(2.0, 10).max() == 1.0


def test_noise_estimate_separates_speech_from_background():
    rng = np.random.default_rng(0)
    wave = 0.01 * rng.standard_normal(32000).astype(np.float32)
    wave[8000:24000] += np.sin(np.arange(16000) / 5.0).astype(np.float32) * 0.3
    db = noise_estimate_db(wave, [(0.5, 1.5)])
    assert db is not None and 20 < db < 40

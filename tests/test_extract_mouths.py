"""Tests for scripts/extract_mouths.py with a stand-in crop function (no MediaPipe)."""

import csv
from types import SimpleNamespace

import numpy as np

from scripts.extract_mouths import extract_mouths, read_log
from scripts.verify_dataset import _load_face_mesh_report


def fake_crop(video_path, model_path, frames):
    """Module-level so spawned worker processes can import it."""
    if "bad" in str(video_path):
        raise OSError("Could not open video")
    valid = np.ones(frames, dtype=bool)
    if "blur" in str(video_path):
        valid[:20] = False                       # 27 % failed -> excluded
    crops = np.full((frames, 96, 96), 7, dtype=np.uint8)
    return SimpleNamespace(crops=crops, valid=valid, exclude_utterance=(~valid).mean() > 0.1)


def _corpus(root, videos):
    """GRID 'sN_processed/<utt>.mpg' layout plus an audio-only speaker like s21."""
    for speaker, utt in videos:
        folder = root / f"{speaker}_processed"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{utt}.mpg").write_bytes(b"")
    (root / "audio" / "s21").mkdir(parents=True)
    (root / "audio" / "s21" / "bbaf2n.wav").write_bytes(b"")
    return root


def _run(tmp_path, **kwargs):
    return extract_mouths(tmp_path / "grid", tmp_path / "mouths", tmp_path / "valid",
                          model=tmp_path / "face_landmarker.task",
                          report=tmp_path / "face_mesh_report.csv",
                          crop_fn=fake_crop, frames=75, **kwargs)


def _log_rows(tmp_path):
    with open(tmp_path / "mouths" / "extraction_log.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_resumes_and_lists_every_utterance(tmp_path):
    _corpus(tmp_path / "grid", [("s1", "bbaf2n"), ("s1", "bbaf3s"), ("s2", "blurry")])

    _run(tmp_path, workers=1, limit=2)
    assert len(_log_rows(tmp_path)) == 3        # 2 processed + s21 without video

    log = _run(tmp_path, workers=1)              # resume: only the third clip runs
    assert len(_log_rows(tmp_path)) == 4
    assert {k: r["status"] for k, r in log.items()} == {
        ("s1", "bbaf2n"): "ok", ("s1", "bbaf3s"): "ok",
        ("s2", "blurry"): "excluded", ("s21", "bbaf2n"): "no_video"}

    crops = np.load(tmp_path / "mouths" / "s1" / "bbaf2n.npy")
    valid = np.load(tmp_path / "valid" / "s1" / "bbaf2n.npy")
    assert crops.shape == (75, 96, 96) and crops.dtype == np.uint8
    assert valid.shape == (75,) and valid.dtype == bool

    _run(tmp_path, workers=1)                    # nothing left to do
    assert len(_log_rows(tmp_path)) == 4
    assert not list((tmp_path / "mouths").rglob("*.tmp"))


def test_report_feeds_verify_dataset(tmp_path):
    _corpus(tmp_path / "grid", [("s1", "bbaf2n"), ("s2", "blurry")])
    _run(tmp_path, workers=1)
    rates = _load_face_mesh_report(tmp_path / "face_mesh_report.csv")
    assert rates == {("s1", "bbaf2n"): 1.0, ("s2", "blurry"): round(55 / 75, 4)}
    assert (tmp_path / "mouths" / "extraction_meta.json").exists()


def test_errors_are_logged_and_retried(tmp_path):
    _corpus(tmp_path / "grid", [("s1", "bbaf2n"), ("s1", "badclip")])
    log = _run(tmp_path, workers=1)
    assert log[("s1", "badclip")]["status"] == "error"
    assert "Could not open video" in log[("s1", "badclip")]["error"]

    _run(tmp_path, workers=1)
    statuses = [r["status"] for r in _log_rows(tmp_path) if r["utterance_id"] == "badclip"]
    assert statuses == ["error", "error"]        # retried, not skipped


def test_parallel_workers(tmp_path):
    utts = [("s1", f"u{i:02d}") for i in range(12)]
    _corpus(tmp_path / "grid", utts)
    log = _run(tmp_path, workers=2)
    assert all(log[k]["status"] == "ok" for k in utts)
    assert len(list((tmp_path / "mouths" / "s1").glob("*.npy"))) == 12
    assert read_log(tmp_path / "mouths" / "extraction_log.csv") == log

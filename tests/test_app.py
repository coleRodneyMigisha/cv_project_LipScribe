"""W3-10: Mode A demo inference. Runs without the real checkpoint or dataset.

A B1 model with random weights stands in for the trained checkpoint: the test is
that the app and evaluate.py's path give identical inputs and identical captions,
not that the caption is correct.
"""
import csv
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
import torch

from lipscribe.app.inference import FriendlyError, LipScribe, mouth_strip
from lipscribe.data.dataset import GridDataset
from lipscribe.models import build_model


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    cfg = {
        "experiment": "b1_test",
        "data": {"manifest": str(tmp_path / "manifest.csv"), "splits": str(tmp_path / "splits.csv"),
                 "feature_stats": str(tmp_path / "stats.npz"), "eval_snrs": ["clean"]},
        "model": {"name": "b1_audio_only", "streams": ["audio"], "audio_features": "logmel80"},
        "train": {"amp": False, "batch_size": 4, "num_workers": 0},
        "decode": {"method": "greedy"},
    }
    np.savez(tmp_path / "stats.npz", audio_mean=np.zeros(80, np.float32), audio_std=np.ones(80, np.float32))
    ckpt = tmp_path / "best.pt"
    torch.save({"config": cfg, "model": build_model(cfg["model"]).state_dict()}, ckpt)

    # One test utterance: end-pointed 1.2 s of audio starting at 0.8 s, 75 video frames
    rng = np.random.default_rng(0)
    wav = tmp_path / "s4_utt.wav"
    sf.write(wav, (0.1 * rng.standard_normal(19200)).astype(np.float32), 16000)
    np.save(tmp_path / "crops.npy", rng.integers(0, 255, (75, 96, 96), dtype=np.uint8))
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["utt_id", "speaker", "transcript", "n_frames", "usable",
                                           "speech_start_s", "mouths_path", "valid_path",
                                           "audio_path", "video_path"])
        w.writeheader()
        w.writerow({"utt_id": "s4_utt", "speaker": "s4", "transcript": "bin blue at f two now",
                    "n_frames": 75, "usable": "True", "speech_start_s": "0.8",
                    "mouths_path": str(tmp_path / "crops.npy"), "valid_path": "",
                    "audio_path": str(wav), "video_path": ""})
    (tmp_path / "splits.csv").write_text("speaker,partition\ns4,test\n")
    return {"ckpt": ckpt, "cfg": cfg, "wav": wav, "tmp": tmp_path}


def test_app_input_is_identical_to_evaluate_input(setup):
    engine = LipScribe(setup["ckpt"], device="cpu", decode="greedy")
    d = setup["cfg"]["data"]
    ds_item = GridDataset(d["manifest"], d["splits"], "test", snr_db=float("inf"))[0]
    app_item, _ = engine.build_item(audio_path=setup["wav"], speech_start_s="0.8", n_frames=75)
    assert app_item["audio"].shape == ds_item["audio"].shape == (48000,)
    assert np.array_equal(app_item["audio"], ds_item["audio"])     # same placement + levelling


@pytest.mark.parametrize("decode", ["greedy", "beam"])
def test_app_caption_equals_evaluate_caption(setup, decode):
    engine = LipScribe(setup["ckpt"], device="cpu", decode=decode)
    d = setup["cfg"]["data"]
    ds_item = GridDataset(d["manifest"], d["splits"], "test", snr_db=float("inf"))[0]
    evaluate_caption = engine.caption_items([ds_item])[0][0]
    app = engine.transcribe(audio_path=setup["wav"], speech_start_s="0.8", n_frames=75)
    assert app.caption == evaluate_caption.strip()
    assert app.n_frames == 75 and app.face_fraction is None


def test_audio_model_without_audio_gives_friendly_error(setup):
    engine = LipScribe(setup["ckpt"], device="cpu")
    with pytest.raises(FriendlyError, match="upload"):
        engine.transcribe()


def test_missing_checkpoint_is_friendly(tmp_path):
    with pytest.raises(FriendlyError, match="not found"):
        LipScribe(tmp_path / "nope.pt")


def test_silent_audio_is_reported(setup):
    silent = setup["tmp"] / "silent.wav"
    sf.write(silent, np.zeros(48000, np.float32), 16000)
    engine = LipScribe(setup["ckpt"], device="cpu")
    with pytest.raises(FriendlyError, match="no audio"):
        engine.transcribe(audio_path=silent)


def test_mouth_strip():
    strip = mouth_strip(np.zeros((75, 96, 96), np.uint8))
    assert strip.shape == (96, 96 * 12)
    assert mouth_strip(None) is None


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


@pytest.mark.skipif(_ffmpeg() is None, reason="ffmpeg not installed")
def test_30fps_webcam_style_video_with_audio(setup):
    """A 2 s, 30 fps clip with a sound track: re-timed to 25 fps, audio taken from the video."""
    video = setup["tmp"] / "webcam.mp4"
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=2",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)],
                   check=True)
    engine = LipScribe(setup["ckpt"], device="cpu", face_model=setup["tmp"] / "missing.task")
    result = engine.transcribe(video)
    assert 48 <= result.n_frames <= 52                  # 2 s at 25 fps
    assert isinstance(result.caption, str)
    assert any("Face Mesh model" in w for w in result.warnings)


@pytest.mark.skipif(_ffmpeg() is None, reason="ffmpeg not installed")
def test_video_without_audio_track_is_friendly(setup):
    video = setup["tmp"] / "mute.mp4"
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=160x120:rate=25:duration=2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    engine = LipScribe(setup["ckpt"], device="cpu", face_model=setup["tmp"] / "missing.task")
    with pytest.raises(FriendlyError, match="no audio track"):
        engine.transcribe(video)


def test_demo_page_builds(setup):
    pytest.importorskip("gradio")
    from lipscribe.app.demo import build_demo
    demo = build_demo(LipScribe(setup["ckpt"], device="cpu"))
    assert demo is not None

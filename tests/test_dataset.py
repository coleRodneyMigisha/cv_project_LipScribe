"""W2-09 integration test: build a manifest from tiny synthetic files and load a batch of 16.

Runs on any laptop and in CI: it never touches the real dataset.
"""
import csv

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from lipscribe.data.dataset import (
    GridDataset, check_speaker_disjoint, collate, ctc_min_frames, mix_at_snr,
)
from lipscribe.data.tokenizer import encode
from scripts.build_manifest import main as build_manifest

SENTENCES = ["bin blue at f two now", "place green by b four please",
             "lay red with s nine again", "set white in z zero soon"]
SPEAKERS = {"s1": "train", "s2": "train", "s3": "val", "s4": "test"}


@pytest.fixture
def tiny_corpus(tmp_path):
    """4 speakers x 5 utterances: crops, valid flags, 16 kHz audio, integrity report, splits."""
    rng = np.random.default_rng(0)
    mouths, audio = tmp_path / "mouths", tmp_path / "audio16k"
    report = tmp_path / "integrity_report.csv"
    fields = ["speaker", "utterance_id", "video_path", "video_frames", "audio_duration_seconds",
              "aligned_speech_start_seconds", "aligned_speech_end_seconds",
              "transcript", "face_mesh_success_rate", "overall_pass"]
    with open(report, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for spk in SPEAKERS:
            (mouths / spk).mkdir(parents=True)
            (audio / spk).mkdir(parents=True)
            for k in range(5):
                utt = f"u{k}"
                T = 75 if k else 70  # one shorter clip to exercise padding
                np.save(mouths / spk / f"{utt}.npy",
                        rng.integers(0, 256, (T, 96, 96), dtype=np.uint8))
                np.save(mouths / spk / f"{utt}_valid.npy", np.ones(T, bool))
                sf.write(audio / spk / f"{utt}.wav",
                         (0.1 * rng.standard_normal(T * 640)).astype(np.float32), 16000)
                w.writerow({"speaker": spk, "utterance_id": utt, "video_path": f"{spk}/{utt}.mpg",
                            "video_frames": T, "audio_duration_seconds": T * 0.04,
                            "aligned_speech_start_seconds": 0.5, "aligned_speech_end_seconds": 2.5,
                            "transcript": SENTENCES[k % 4], "face_mesh_success_rate": "",
                            "overall_pass": "True"})
        # one failed utterance and one with a broken face track: both must be excluded
        w.writerow({"speaker": "s1", "utterance_id": "bad", "video_path": "x", "video_frames": 75,
                    "transcript": SENTENCES[0], "overall_pass": "False"})
    np.save(mouths / "s2" / "u4_valid.npy", np.r_[np.ones(60, bool), np.zeros(15, bool)])

    splits = tmp_path / "splits.csv"
    with open(splits, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["speaker", "partition"])
        w.writerows(SPEAKERS.items())

    noise = tmp_path / "noise.wav"
    sf.write(noise, (0.1 * rng.standard_normal(16000)).astype(np.float32), 16000)

    manifest = tmp_path / "manifest.csv"
    assert build_manifest(["--integrity", str(report), "--mouths-dir", str(mouths),
                           "--audio-dir", str(audio), "--splits", str(splits),
                           "--out", str(manifest)]) == 0
    return {"manifest": manifest, "noise": noise, "tmp": tmp_path}


def test_manifest_excludes_bad_rows(tiny_corpus):
    rows = list(csv.DictReader(open(tiny_corpus["manifest"])))
    assert len(rows) == 21
    usable = {r["utt_id"] for r in rows if r["usable"] == "True"}
    assert "s1_bad" not in usable          # integrity failure
    assert "s2_u4" not in usable           # 80 % face detections < 90 %
    assert len(usable) == 19


def test_one_batch_of_16_has_consistent_shapes(tiny_corpus):
    ds = GridDataset(tiny_corpus["manifest"], "train", noise_files=[tiny_corpus["noise"]])
    assert len(ds) == 9  # s1 (5) + s2 (4 usable)
    big = torch.utils.data.ConcatDataset([ds, ds])  # 18 items so a full batch of 16 exists
    batch = next(iter(DataLoader(big, batch_size=16, shuffle=True, collate_fn=collate)))
    B, T = 16, 75
    assert batch["video"].shape == (B, 1, T, 96, 96)
    assert batch["video"].dtype == torch.float32 and batch["video"].max() <= 1.0
    assert batch["audio"].shape == (B, T * 640)
    assert batch["mask"].shape == (B, T)
    # one 40 ms grid: T_a = 4T, encoder length = T
    assert torch.equal(batch["feat_lengths"], batch["audio_lengths"] // 160)
    assert torch.equal(batch["lengths"], (batch["feat_lengths"] + 3) // 4)
    assert torch.equal(batch["mask"].sum(1), batch["lengths"])
    assert batch["targets"].numel() == int(batch["target_lengths"].sum())
    assert len(batch["texts"]) == len(batch["utt_ids"]) == B


def test_short_clip_is_padded_and_masked(tiny_corpus):
    ds = GridDataset(tiny_corpus["manifest"], "val")
    batch = collate([ds[i] for i in range(len(ds))])
    assert batch["video"].shape[2] == 75
    short = batch["utt_ids"].index("s3_u0")
    assert int(batch["lengths"][short]) == 70
    assert not batch["mask"][short, 70:].any()
    assert batch["video"][short, 0, 70:].abs().sum() == 0


def test_val_test_noisy_files_are_used(tiny_corpus):
    noisy = tiny_corpus["tmp"] / "noisy" / "5"
    noisy.mkdir(parents=True)
    for k in range(5):
        sf.write(noisy / f"s4_u{k}.wav", np.full((75 if k else 70) * 640, 0.5, np.float32), 16000)
    item = GridDataset(tiny_corpus["manifest"], "test", snr=5,
                       noisy_dir=tiny_corpus["tmp"] / "noisy")[1]
    assert item["snr_db"] == 5.0
    assert np.allclose(item["audio"], 0.5)


def test_speaker_leakage_raises():
    with pytest.raises(ValueError):
        check_speaker_disjoint([{"speaker": "s1", "partition": "train"},
                                {"speaker": "s1", "partition": "test"}])


def test_ctc_condition():
    assert ctc_min_frames(encode("ee")) == 3       # e, blank, e
    assert ctc_min_frames(encode("bin")) == 3
    item = {"utt_id": "x", "text": "green", "video": np.zeros((2, 96, 96), np.uint8),
            "video_valid": np.ones(2, bool), "audio": np.zeros(2 * 640, np.float32),
            "target": np.array(encode("green")), "snr_db": 30.0}
    with pytest.raises(ValueError, match="CTC condition"):
        collate([item])


def test_mix_at_snr_hits_target():
    rng = np.random.default_rng(1)
    clean = rng.standard_normal(48000).astype(np.float32)
    noise = rng.standard_normal(16000).astype(np.float32)
    for snr in (10, 5, 0, -5):
        noisy = mix_at_snr(clean, noise, snr)
        w = noisy - clean
        measured = 10 * np.log10(np.mean(clean ** 2) / np.mean(w ** 2))
        assert abs(measured - snr) < 0.01

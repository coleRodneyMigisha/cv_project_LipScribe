"""W2-09 integration test: build a manifest from tiny synthetic files and load a batch of 16.

Uses the same calls as scripts/train.py. Never touches the real dataset.
"""
import csv
import math

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, DataLoader

from lipscribe.data.dataset import (
    CLEAN, GridDataset, check_speaker_disjoint, collate, ctc_min_frames, mix_at_snr,
    place_on_video_grid, read_splits,
)
from lipscribe.data.tokenizer import encode
from scripts.build_manifest import main as build_manifest

SENTENCES = ["bin blue at f two now", "place green by b four please",
             "lay red with s nine again", "set white in z zero soon"]
SPEAKERS = {"s1": "train", "s2": "train", "s3": "val", "s4": "test", "s5": "excluded"}


@pytest.fixture
def corpus(tmp_path):
    """5 speakers x 5 utterances laid out like extract_mouths.py + audio prep output."""
    rng = np.random.default_rng(0)
    mouths, valid_dir = tmp_path / "mouths", tmp_path / "mouth_valid"
    audio = tmp_path / "audio16k"
    report = tmp_path / "integrity_report.csv"
    fields = ["speaker", "utterance_id", "video_path", "video_frames", "audio_duration_seconds",
              "aligned_speech_start_seconds", "aligned_speech_end_seconds",
              "transcript", "face_mesh_success_rate", "overall_pass"]
    with open(report, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for spk in SPEAKERS:
            for d in (mouths, valid_dir, audio):
                (d / spk).mkdir(parents=True)
            for k in range(5):
                utt = f"u{k}"
                T = 75 if k else 70  # one shorter clip to exercise padding
                np.save(mouths / spk / f"{utt}.npy", rng.integers(0, 256, (T, 96, 96), dtype=np.uint8))
                np.save(valid_dir / spk / f"{utt}.npy", np.ones(T, bool))
                sf.write(audio / spk / f"{utt}.wav",
                         (0.1 * rng.standard_normal(T * 640)).astype(np.float32), 16000)
                w.writerow({"speaker": spk, "utterance_id": utt, "video_path": f"{spk}/{utt}.mpg",
                            "video_frames": T, "audio_duration_seconds": T * 0.04,
                            "aligned_speech_start_seconds": 0.5, "aligned_speech_end_seconds": 2.5,
                            "transcript": SENTENCES[k % 4], "face_mesh_success_rate": "",
                            "overall_pass": "True"})
        w.writerow({"speaker": "s1", "utterance_id": "bad", "video_path": "x", "video_frames": 75,
                    "transcript": SENTENCES[0], "overall_pass": "False"})
    # s2_u4: only 80 % of frames have a face -> excluded (< 90 %)
    np.save(valid_dir / "s2" / "u4.npy", np.r_[np.ones(60, bool), np.zeros(15, bool)])

    splits = tmp_path / "splits.csv"
    with open(splits, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["speaker", "partition", "gender", "usable_utterances"])
        w.writerows([(s, p, "M", 5) for s, p in SPEAKERS.items()])

    # MUSAN pool list in make_noise_splits.py format: paths relative to the folder above it
    (tmp_path / "musan").mkdir()
    pools = []
    for pool in ("train", "val", "test"):
        f = tmp_path / "musan" / f"{pool}.wav"
        sf.write(f, (0.1 * rng.standard_normal(16000)).astype(np.float32), 16000)
        pools.append({"file": f"musan/{pool}.wav", "pool": pool})
    (tmp_path / "Noise_splits").mkdir()
    noise_splits = tmp_path / "Noise_splits" / "splits.csv"
    with open(noise_splits, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "pool"])
        w.writeheader()
        w.writerows(pools)

    manifest = tmp_path / "manifest.csv"
    assert build_manifest(["--integrity", str(report), "--mouths-dir", str(mouths),
                           "--valid-dir", str(valid_dir), "--audio-dir", str(audio),
                           "--splits", str(splits), "--out", str(manifest)]) == 0
    return {"manifest": manifest, "splits": splits, "noise_splits": noise_splits, "tmp": tmp_path}


def test_manifest_excludes_bad_rows(corpus):
    rows = list(csv.DictReader(open(corpus["manifest"])))
    assert len(rows) == 26
    usable = {r["utt_id"] for r in rows if r["usable"] == "True"}
    assert "s1_bad" not in usable                          # integrity failure
    assert "s2_u4" not in usable                           # 80 % face detections
    assert not any(u.startswith("s5_") for u in usable)    # excluded speaker
    assert len(usable) == 19
    assert all(r["valid_path"] for r in rows if r["usable"] == "True")


def test_one_batch_of_16_has_consistent_shapes(corpus):
    # Same call as scripts/train.py build_loaders()
    ds = GridDataset(corpus["manifest"], corpus["splits"], "train",
                     snrs=[CLEAN, 10.0, 5.0, 0.0, -5.0], noise_splits=corpus["noise_splits"])
    assert len(ds) == 9  # s1 (5) + s2 (4 usable)
    batch = next(iter(DataLoader(ConcatDataset([ds, ds]), batch_size=16, shuffle=True,
                                 collate_fn=collate)))
    B, T = 16, 75
    assert batch["video"].shape == (B, 1, T, 96, 96)
    assert batch["video"].dtype == torch.float32 and batch["video"].max() <= 1.0
    assert batch["audio"].shape == (B, T * 640)
    assert batch["mask"].shape == batch["video_valid"].shape == (B, T)
    assert torch.equal(batch["feat_lengths"], batch["audio_lengths"] // 160)
    assert torch.equal(batch["lengths"], (batch["feat_lengths"] + 3) // 4)
    assert torch.equal(batch["mask"].sum(1), batch["lengths"])
    assert batch["targets"].numel() == int(batch["target_lengths"].sum())
    assert len(batch["texts"]) == len(batch["utt_ids"]) == B


def test_features_line_up_with_video(corpus):
    features = pytest.importorskip("lipscribe.preprocessing.features")
    ds = GridDataset(corpus["manifest"], corpus["splits"], "val")
    batch = collate([ds[i] for i in range(len(ds))])
    feats = features.logmel(batch["audio"], n_frames=batch["video"].shape[2])
    assert feats.shape == (len(ds), 4 * 75, 80)  # T_a = 4T, as train.py expects


def test_short_clip_is_padded_and_masked(corpus):
    ds = GridDataset(corpus["manifest"], corpus["splits"], "val")
    batch = collate([ds[i] for i in range(len(ds))])
    short = batch["utt_ids"].index("s3_u0")
    assert batch["video"].shape[2] == 75
    assert int(batch["lengths"][short]) == 70
    assert not batch["mask"][short, 70:].any()
    assert batch["video"][short, 0, 70:].abs().sum() == 0


def test_clean_val_is_inf_snr(corpus):
    item = GridDataset(corpus["manifest"], corpus["splits"], "val", snr_db=CLEAN)[0]
    assert math.isinf(item["snr_db"])


def test_val_uses_fixed_noisy_files_when_present(corpus):
    noisy = corpus["tmp"] / "noisy"
    (noisy / "5").mkdir(parents=True)
    for k in range(5):
        sf.write(noisy / "5" / f"s4_u{k}.wav", np.full((75 if k else 70) * 640, 0.5, np.float32), 16000)
    item = GridDataset(corpus["manifest"], corpus["splits"], "test", snr_db=5.0, noisy_dir=noisy)[1]
    assert item["snr_db"] == 5.0
    # the constant 0.5 file, levelled to -25 dBFS
    assert np.allclose(item["audio"], 10 ** (-25 / 20), atol=1e-5)


def test_val_falls_back_to_deterministic_mix(corpus):
    kw = dict(snr_db=0.0, noisy_dir=corpus["tmp"] / "none", noise_splits=corpus["noise_splits"])
    a = GridDataset(corpus["manifest"], corpus["splits"], "val", **kw)[2]["audio"]
    b = GridDataset(corpus["manifest"], corpus["splits"], "val", seed=99, **kw)[2]["audio"]
    clean = GridDataset(corpus["manifest"], corpus["splits"], "val")[2]["audio"]
    assert np.array_equal(a, b)              # identical for every run and model
    assert not np.allclose(a, clean)         # noise was added


def test_training_without_noise_pool_raises(corpus):
    with pytest.raises(ValueError, match="noise"):
        GridDataset(corpus["manifest"], corpus["splits"], "train",
                    noise_splits=corpus["tmp"] / "missing.csv")


def test_speaker_leakage_raises(tmp_path):
    with pytest.raises(ValueError):
        check_speaker_disjoint([{"speaker": "s1", "partition": "train"},
                                {"speaker": "s1", "partition": "test"}])
    bad = tmp_path / "splits.csv"
    bad.write_text("speaker,partition\ns1,train\ns1,test\n")
    with pytest.raises(ValueError, match="leakage"):
        read_splits(bad)


def test_ctc_condition():
    assert ctc_min_frames(encode("ee")) == 3       # e, blank, e
    item = {"utt_id": "x", "text": "green", "video": np.zeros((2, 96, 96), np.uint8),
            "video_valid": np.ones(2, bool), "audio": np.zeros(2 * 640, np.float32),
            "target": np.array(encode("green")), "snr_db": CLEAN}
    with pytest.raises(ValueError, match="CTC condition"):
        collate([item])


def _dbfs(wave):
    return 20 * np.log10(np.sqrt(np.mean(np.square(wave))))


def test_audio_is_levelled_to_minus_25_dbfs(corpus):
    train = GridDataset(corpus["manifest"], corpus["splits"], "train",
                        noise_splits=corpus["noise_splits"])
    val = GridDataset(corpus["manifest"], corpus["splits"], "val")
    for item in [train[i] for i in range(len(train))] + [val[0]]:
        assert _dbfs(item["audio"]) == pytest.approx(-25.0, abs=0.01)


def test_place_on_video_grid():
    n = 75 * 640
    trimmed = np.ones(int(1.2 * 16000), np.float32)          # end-pointed GRID file
    out = place_on_video_grid(trimmed, n, "0.95")
    assert len(out) == n and np.flatnonzero(out)[0] == round(0.95 * 16000)
    full = np.ones(n - 100, np.float32)                       # already on the video grid
    assert np.flatnonzero(place_on_video_grid(full, n, 0.95))[0] == 0
    assert np.flatnonzero(place_on_video_grid(trimmed, n, ""))[0] == 0   # unknown start


def test_trimmed_audio_sits_under_the_lips(corpus):
    """An end-pointed file starts at the first word; the dataset shifts it by speech_start_s."""
    rows = list(csv.DictReader(open(corpus["manifest"])))
    row = next(r for r in rows if r["utt_id"] == "s3_u1")
    sf.write(row["audio_path"], np.full(int(1.2 * 16000), 0.1, np.float32), 16000)
    item = GridDataset(corpus["manifest"], corpus["splits"], "val")[1]
    assert item["utt_id"] == "s3_u1" and len(item["audio"]) == 75 * 640
    assert np.flatnonzero(item["audio"])[0] == round(float(row["speech_start_s"]) * 16000)


def test_workers_draw_fresh_snrs_each_epoch(corpus):
    ds = GridDataset(corpus["manifest"], corpus["splits"], "train", noise_splits=corpus["noise_splits"])
    torch.manual_seed(0)
    dl = DataLoader(ConcatDataset([ds] * 4), batch_size=1, num_workers=2, collate_fn=collate)
    epochs = [[float(b["snr_db"][0]) for b in dl] for _ in range(2)]
    assert epochs[0] != epochs[1]                             # fresh draws per epoch
    w0, w1 = epochs[0][0::2], epochs[0][1::2]                 # alternating workers
    assert w0 != w1                                           # workers differ
    torch.manual_seed(0)
    assert [float(b["snr_db"][0]) for b in dl] == epochs[0]   # reproducible from the seed


def test_mix_at_snr_hits_target():
    rng = np.random.default_rng(1)
    clean = rng.standard_normal(48000).astype(np.float32)
    noise = rng.standard_normal(16000).astype(np.float32)
    for snr in (10, 5, 0, -5):
        w = mix_at_snr(clean, noise, snr) - clean
        assert abs(10 * np.log10(np.mean(clean ** 2) / np.mean(w ** 2)) - snr) < 0.01

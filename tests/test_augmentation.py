"""Unit tests for visual data augmentation (W3-04)."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lipscribe.data.augmentation import (
    apply_visual_augmentation,
    random_crop_jitter,
    random_horizontal_flip,
)
from lipscribe.data.dataset import CLEAN, GridDataset
from scripts.train import build_loaders


def test_random_crop_jitter_shape_and_dtype():
    t, h, w = 75, 96, 96
    frames = np.random.randint(0, 256, size=(t, h, w), dtype=np.uint8)
    jittered = random_crop_jitter(frames, max_jitter=4, rng=random.Random(42))

    assert jittered.shape == (t, h, w)
    assert jittered.dtype == np.uint8


def test_random_crop_jitter_zero_jitter_is_identity():
    frames = np.random.randint(0, 256, size=(10, 96, 96), dtype=np.uint8)
    jittered = random_crop_jitter(frames, max_jitter=0)
    np.testing.assert_array_equal(jittered, frames)


def test_random_crop_jitter_same_offset_across_all_frames():
    """Verify every frame of a clip receives the exact same spatial translation."""
    t, h, w = 20, 96, 96
    base_frame = np.arange(h * w, dtype=np.int32).reshape(h, w)
    frames = np.stack([base_frame + t for t in range(t)])

    jittered = random_crop_jitter(frames, max_jitter=5, rng=random.Random(123))

    for t_idx in range(1, t):
        diff = jittered[t_idx] - jittered[0]
        assert np.all(diff == t_idx), f"Frame {t_idx} received different shift than frame 0!"


def test_random_crop_jitter_preserves_motion():
    """Moving a bright pixel across time: its trajectory is shifted by the exact same dx, dy."""
    t, h, w = 5, 96, 96
    frames = np.zeros((t, h, w), dtype=np.int32)
    trajectory = [(20 + t_idx, 30 + 2 * t_idx) for t_idx in range(t)]
    for t_idx, (y, x) in enumerate(trajectory):
        frames[t_idx, y, x] = 1000

    jittered = random_crop_jitter(frames, max_jitter=4, rng=random.Random(99))

    new_trajectory = []
    for t_idx in range(t):
        y, x = np.unravel_index(np.argmax(jittered[t_idx]), (h, w))
        new_trajectory.append((y, x))

    shifts = [(ny - oy, nx - ox) for (ny, nx), (oy, ox) in zip(new_trajectory, trajectory)]
    assert len(set(shifts)) == 1, f"Trajectory shifted inconsistently: {shifts}"


def test_random_horizontal_flip_shape_and_dtype():
    t, h, w = 50, 96, 96
    frames = np.random.randint(0, 256, size=(t, h, w), dtype=np.uint8)
    flipped = random_horizontal_flip(frames, p=1.0)

    assert flipped.shape == (t, h, w)
    assert flipped.dtype == np.uint8


def test_random_horizontal_flip_zero_probability():
    frames = np.random.randint(0, 256, size=(10, 96, 96), dtype=np.uint8)
    res = random_horizontal_flip(frames, p=0.0)
    np.testing.assert_array_equal(res, frames)


def test_random_horizontal_flip_all_frames_same():
    """Verify all frames in a clip are flipped together (either all or none)."""
    t, h, w = 25, 96, 96
    frames = np.zeros((t, h, w), dtype=np.uint8)
    frames[:, :, : w // 2] = 1

    flipped = random_horizontal_flip(frames, p=1.0)
    for t_idx in range(t):
        assert np.all(flipped[t_idx, :, w // 2 :] == 1)
        assert np.all(flipped[t_idx, :, : w // 2] == 0)

    rng = random.Random(77)
    for _ in range(50):
        out = random_horizontal_flip(frames, p=0.5, rng=rng)
        is_f0_flipped = bool(np.all(out[0, :, w // 2 :] == 1))
        for t_idx in range(1, t):
            is_ft_flipped = bool(np.all(out[t_idx, :, w // 2 :] == 1))
            assert is_ft_flipped == is_f0_flipped, f"Frame {t_idx} flip state differed from frame 0!"


def test_random_horizontal_flip_reverses_horizontal_motion_consistently():
    """Moving dot left->right (dx > 0) becomes right->left (dx < 0) smoothly after flip."""
    t, h, w = 6, 96, 96
    frames = np.zeros((t, h, w), dtype=np.int32)
    for t_idx in range(t):
        frames[t_idx, 40, 10 + 5 * t_idx] = 500  # x moves 10, 15, 20...

    flipped = random_horizontal_flip(frames, p=1.0)
    xs = [int(np.argmax(flipped[t_idx, 40, :])) for t_idx in range(t)]
    expected_xs = [w - 1 - (10 + 5 * t_idx) for t_idx in range(t)]
    assert xs == expected_xs


def test_apply_visual_augmentation_combined():
    t, h, w = 15, 96, 96
    frames = np.random.randint(0, 256, size=(t, h, w), dtype=np.uint8)
    augmented = apply_visual_augmentation(frames, max_jitter=3, p_flip=0.5, rng=random.Random(12))
    assert augmented.shape == (t, h, w)
    assert augmented.dtype == np.uint8


def test_all_frames_of_an_augmented_clip_get_the_same_transform():
    """Unit test required by W3-04: all frames of an augmented clip get the same transform."""
    t, h, w = 30, 96, 96
    # Construct distinct identifiable pattern: base + t
    base = np.arange(h * w, dtype=np.int32).reshape(h, w)
    frames = np.stack([base + t for t in range(t)])

    for seed in range(25):
        rng = random.Random(seed)
        augmented = apply_visual_augmentation(frames, max_jitter=5, p_flip=0.5, rng=rng)
        assert augmented.shape == (t, h, w)
        assert augmented.dtype == np.int32

        # Every frame received the identical spatial transform
        for t_idx in range(1, t):
            diff = augmented[t_idx] - augmented[0]
            assert np.all(diff == t_idx), (
                f"Seed {seed}: Frame {t_idx} received a different transform than frame 0!"
            )



@pytest.fixture
def synthetic_data_env(tmp_path):
    """Minimal dataset environment to test augmentation disabling for val/test."""
    splits_csv = tmp_path / "splits.csv"
    with open(splits_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "partition", "gender", "usable_utterances"])
        w.writerow(["s1", "train", "M", 1])
        w.writerow(["s2", "val", "F", 1])
        w.writerow(["s3", "test", "M", 1])

    mouths_dir = tmp_path / "mouths"
    audio_dir = tmp_path / "audio"
    for spk in ("s1", "s2", "s3"):
        (mouths_dir / spk).mkdir(parents=True)
        (audio_dir / spk).mkdir(parents=True)

    manifest_csv = tmp_path / "manifest.csv"
    fieldnames = ["utt_id", "speaker", "transcript", "video_frames", "mouths_path",
                  "valid_path", "audio_path", "speech_start_s", "speech_end_s", "usable"]
    rows = []
    # Distinct gradient frame for unmistakable detection of augmentation
    t, h, w = 10, 96, 96
    raw_crop = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    crops = np.stack([raw_crop for _ in range(t)])

    for spk, part in [("s1", "train"), ("s2", "val"), ("s3", "test")]:
        utt_id = f"{spk}_u1"
        m_path = mouths_dir / spk / f"{utt_id}.npy"
        np.save(m_path, crops)
        a_path = audio_dir / spk / f"{utt_id}.wav"
        sf.write(a_path, np.zeros(t * 640, dtype=np.float32), 16000)

        rows.append({
            "utt_id": utt_id,
            "speaker": spk,
            "transcript": "bin blue",
            "video_frames": str(t),
            "mouths_path": str(m_path),
            "valid_path": "",
            "audio_path": str(a_path),
            "speech_start_s": "0.0",
            "speech_end_s": "0.4",
            "usable": "True",
        })

    with open(manifest_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return {
        "manifest": str(manifest_csv),
        "splits": str(splits_csv),
        "crops": crops,
    }


def test_augmentation_disabled_for_val_and_test(synthetic_data_env):
    """Subtask 3: Augmentation must be disabled for validation, test, and inference."""
    env = synthetic_data_env
    manifest, splits, original_crops = env["manifest"], env["splits"], env["crops"]

    # Even if visual_augment=True is passed, val and test MUST have visual_augment=False
    val_ds = GridDataset(manifest, splits, "val", snr_db=CLEAN, visual_augment=True)
    assert val_ds.visual_augment is False, "visual_augment must be False for validation partition"
    val_item = val_ds[0]
    np.testing.assert_array_equal(val_item["video"], original_crops)

    test_ds = GridDataset(manifest, splits, "test", snr_db=CLEAN, visual_augment=True)
    assert test_ds.visual_augment is False, "visual_augment must be False for test partition"
    test_item = test_ds[0]
    np.testing.assert_array_equal(test_item["video"], original_crops)

    # For train partition:
    # 1. When visual_augment=False -> unaugmented
    train_clean_ds = GridDataset(manifest, splits, "train", snrs=[CLEAN], visual_augment=False)
    assert train_clean_ds.visual_augment is False
    train_clean_item = train_clean_ds[0]
    np.testing.assert_array_equal(train_clean_item["video"], original_crops)

    # 2. When visual_augment=True -> augmented (active)
    # Using fixed seed to guarantee transform differs from identity
    train_aug_ds = GridDataset(
        manifest, splits, "train", snrs=[CLEAN], visual_augment=True,
        max_jitter=4, p_flip=1.0, seed=42,
    )
    assert train_aug_ds.visual_augment is True
    train_aug_item = train_aug_ds[0]
    # With p_flip=1.0, it is guaranteed to be flipped horizontally, so not equal to original
    assert not np.array_equal(train_aug_item["video"], original_crops)


def test_augmentation_switchable_in_config(synthetic_data_env):
    """Subtask 3: build_loaders correctly propagates config switch for train and disables it for val."""
    env = synthetic_data_env
    cfg_enabled = {
        "data": {
            "manifest": env["manifest"],
            "splits": env["splits"],
            "train_snrs": ["clean"],
            "eval_snrs": ["clean"],
            "visual_augment": True,
        },
        "train": {
            "batch_size": 1,
            "num_workers": 0,
        },
    }

    train_loader, val_loaders = build_loaders(cfg_enabled)
    assert train_loader.dataset.visual_augment is True
    for snr_key, v_loader in val_loaders.items():
        assert v_loader.dataset.visual_augment is False

    cfg_disabled = {
        "data": {
            "manifest": env["manifest"],
            "splits": env["splits"],
            "train_snrs": ["clean"],
            "eval_snrs": ["clean"],
            "visual_augment": False,
        },
        "train": {
            "batch_size": 1,
            "num_workers": 0,
        },
    }

    train_loader_off, val_loaders_off = build_loaders(cfg_disabled)
    assert train_loader_off.dataset.visual_augment is False
    for snr_key, v_loader in val_loaders_off.items():
        assert v_loader.dataset.visual_augment is False

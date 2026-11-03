"""Unit tests for visual data augmentation (W3-04)."""
from __future__ import annotations

import random
import numpy as np
import pytest

from lipscribe.data.augmentation import (
    apply_visual_augmentation,
    random_crop_jitter,
    random_horizontal_flip,
)


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
    # Left side 1, right side 0
    frames = np.zeros((t, h, w), dtype=np.uint8)
    frames[:, :, : w // 2] = 1

    # Test with p=1.0 (guaranteed flip)
    flipped = random_horizontal_flip(frames, p=1.0)
    # Right side should now be 1, left side 0 for EVERY frame
    for t_idx in range(t):
        assert np.all(flipped[t_idx, :, w // 2 :] == 1)
        assert np.all(flipped[t_idx, :, : w // 2] == 0)

    # Test random sampling over 50 clips: every clip must be uniformly flipped or unflipped across all frames
    rng = random.Random(77)
    for _ in range(50):
        out = random_horizontal_flip(frames, p=0.5, rng=rng)
        # Check if frame 0 is flipped
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
    # Should move in opposite direction: (w - 1 - 10) - 5*t
    expected_xs = [w - 1 - (10 + 5 * t_idx) for t_idx in range(t)]
    assert xs == expected_xs


def test_apply_visual_augmentation_combined():
    t, h, w = 15, 96, 96
    frames = np.random.randint(0, 256, size=(t, h, w), dtype=np.uint8)
    augmented = apply_visual_augmentation(frames, max_jitter=3, p_flip=0.5, rng=random.Random(12))
    assert augmented.shape == (t, h, w)
    assert augmented.dtype == np.uint8

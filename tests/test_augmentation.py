"""Unit tests for visual data augmentation (W3-04)."""
from __future__ import annotations

import random
import numpy as np
import pytest

from lipscribe.data.augmentation import random_crop_jitter


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
    # Create frames where each frame has a distinct, non-zero gradient or coordinate stamp
    # For instance, frame[t, y, x] = (y * 100 + x) % 251 + t * 0 (so every frame is identical spatially)
    base_frame = np.arange(h * w, dtype=np.int32).reshape(h, w)
    frames = np.stack([base_frame + t for t in range(t)])

    jittered = random_crop_jitter(frames, max_jitter=5, rng=random.Random(123))

    # Calculate difference between each frame and frame 0
    # Since original frames[t] - frames[0] == t * np.ones((h, w)),
    # jittered[t] - jittered[0] must also be identically t * np.ones((h, w))
    # everywhere, confirming the spatial shift is identical across all t frames!
    for t_idx in range(1, t):
        diff = jittered[t_idx] - jittered[0]
        assert np.all(diff == t_idx), f"Frame {t_idx} received different shift than frame 0!"


def test_random_crop_jitter_preserves_motion():
    """Moving a bright pixel across time: its trajectory is shifted by the exact same dx, dy."""
    t, h, w = 5, 96, 96
    frames = np.zeros((t, h, w), dtype=np.int32)
    # Bright spot moves: (20 + t, 30 + 2*t)
    trajectory = [(20 + t_idx, 30 + 2 * t_idx) for t_idx in range(t)]
    for t_idx, (y, x) in enumerate(trajectory):
        frames[t_idx, y, x] = 1000

    jittered = random_crop_jitter(frames, max_jitter=4, rng=random.Random(99))

    # Find the peak in each jittered frame
    new_trajectory = []
    for t_idx in range(t):
        y, x = np.unravel_index(np.argmax(jittered[t_idx]), (h, w))
        new_trajectory.append((y, x))

    shifts = [(ny - oy, nx - ox) for (ny, nx), (oy, ox) in zip(new_trajectory, trajectory)]
    # All shifts must be identical
    assert len(set(shifts)) == 1, f"Trajectory shifted inconsistently: {shifts}"

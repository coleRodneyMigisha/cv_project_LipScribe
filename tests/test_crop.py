
from pathlib import Path

import cv2
import numpy as np

from lipscribe.preprocessing.crop import (
    OUTPUT_SIZE,
    centred_moving_average,
    crop_with_edge_padding,
    get_lip_indices,
)


def test_lip_indices_are_loaded_from_mediapipe_connections():
    indices = get_lip_indices()

    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int32
    assert len(indices) > 0


def test_centered_moving_average_preserves_shape():
    values = np.array(
        [
            [0.0, 10.0, 20.0],
            [2.0, 12.0, 22.0],
            [4.0, 14.0, 24.0],
            [6.0, 16.0, 26.0],
            [8.0, 18.0, 28.0],
        ],
        dtype=np.float32,
    )

    smoothed = centred_moving_average(values, radius=2)

    assert smoothed.shape == values.shape
    assert np.isfinite(smoothed).all()


def test_edge_padded_crop_shape_and_dtype():
    frame_rgb = np.zeros(
        (40, 60, 3),
        dtype=np.uint8,
    )

    crop = crop_with_edge_padding(
        frame_rgb,
        center_x=2,
        center_y=3,
        side=30,
    )

    assert crop.shape == OUTPUT_SIZE
    assert crop.dtype == np.uint8


def test_crop_output_for_all_five_clips():
    clips_dir = Path("../data/mouth_crop_clips")
    model_path = Path(
        "../models/face_landmarker.task"
    )

    clips = sorted(clips_dir.glob("*.mp4"))

    assert len(clips) == 5, (
        f"Expected 5 clips, found {len(clips)}"
    )

    # Import here so the simple unit tests above do not need to initialize
    # Face Landmarker unless this integration test is being run.
    from lipscribe.preprocessing.crop import crop_video

    for clip_path in clips:
        result = crop_video(
            video_path=clip_path,
            model_path=model_path,
        )

        assert result.crops.ndim == 3
        assert result.crops.shape[1:] == (96, 96)
        assert result.crops.dtype == np.uint8

        assert result.valid.ndim == 1
        assert result.valid.dtype == np.bool_
        assert len(result.valid) == len(result.crops)

        assert result.frame_indices.ndim == 1
        assert len(result.frame_indices) == len(result.crops)

        expected_indices = np.arange(
            len(result.crops),
            dtype=np.int64,
        )

        np.testing.assert_array_equal(
            result.frame_indices,
            expected_indices,
        )

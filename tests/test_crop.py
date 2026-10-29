
from pathlib import Path

import cv2
import numpy as np

from lipscribe.preprocessing.crop import (
    MAX_INVALID_FRACTION,
    OUTPUT_SIZE,
    centred_moving_average,
    crop_with_edge_padding,
    fill_missing_boxes,
    get_lip_indices,
    process_frames,
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


def test_missing_face_gets_interpolated_box_and_invalid_mask():
    frames = [
        np.full((40, 40, 3), 20, dtype=np.uint8),
        np.full((40, 40, 3), 40, dtype=np.uint8),
        np.full((40, 40, 3), 60, dtype=np.uint8),
    ]
    boxes = [
        np.array([10.0, 20.0, 12.0], dtype=np.float32),
        None,  # artificially missing face
        np.array([20.0, 20.0, 12.0], dtype=np.float32),
    ]

    result = process_frames(frames, boxes)

    assert result.crops.shape == (3, 96, 96)
    assert result.crops.dtype == np.uint8
    assert result.valid.tolist() == [True, False, True]
    assert result.boxes is not None
    np.testing.assert_allclose(result.boxes[1], [15.0, 20.0, 12.0])


def test_fill_missing_boxes_uses_nearest_box_at_clip_edges():
    boxes = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 12.0],
            [20.0, 20.0, 12.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid = np.array([False, True, True, False])

    filled = fill_missing_boxes(boxes, valid)

    assert filled is not None
    np.testing.assert_allclose(filled[0], filled[1])
    np.testing.assert_allclose(filled[3], filled[2])


def test_audio_alignment_pads_with_last_frame_and_marks_padding_invalid():
    frames = [
        np.full((40, 40, 3), 20, dtype=np.uint8),
        np.full((40, 40, 3), 40, dtype=np.uint8),
    ]
    boxes = [
        np.array([20.0, 20.0, 12.0], dtype=np.float32),
        np.array([20.0, 20.0, 12.0], dtype=np.float32),
    ]

    result = process_frames(frames, boxes, target_frame_count=4)

    assert result.crops.shape == (4, 96, 96)
    assert result.valid.tolist() == [True, True, False, False]
    np.testing.assert_array_equal(result.frame_indices, np.arange(4))
    assert result.exclude_utterance is True


def test_audio_alignment_truncates_without_dropping_alignment_indices():
    frames = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(4)]
    boxes = [np.array([20.0, 20.0, 12.0], dtype=np.float32) for _ in frames]

    result = process_frames(frames, boxes, target_frame_count=2)

    assert result.crops.shape == (2, 96, 96)
    np.testing.assert_array_equal(result.frame_indices, np.array([0, 1]))
    assert result.valid.all()


def test_more_than_ten_percent_invalid_frames_flags_utterance():
    frame_count = 10
    frames = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(frame_count)]
    boxes = [
        np.array([20.0, 20.0, 12.0], dtype=np.float32)
        if index not in (0, 1)
        else None
        for index in range(frame_count)
    ]

    result = process_frames(frames, boxes)

    assert result.invalid_fraction > MAX_INVALID_FRACTION
    assert result.exclude_utterance is True


def test_crop_output_for_all_five_clips():
    project_root = Path(__file__).resolve().parents[1]
    clips_dir = project_root / "data" / "mouth_crop_clips"
    model_path = project_root / "models" / "face_landmarker.task"

    clips = [
        clips_dir / f"clip_{index}.mp4"
        for index in range(1, 6)
    ]

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


def test_artificially_blanked_clip_interpolates_missing_frames():
    project_root = Path(__file__).resolve().parents[1]

    blanked_clip = (
        project_root
        / "data"
        / "mouth_crop_clips"
        / "clip_6_blank.mp4"
    )

    model_path = (
        project_root
        / "models"
        / "face_landmarker.task"
    )

    assert blanked_clip.exists(), (
        f"Missing blanked test clip: {blanked_clip}"
    )

    from lipscribe.preprocessing.crop import crop_video

    result = crop_video(
        video_path=blanked_clip,
        model_path=model_path,
    )

    # The crop sequence must preserve the decoded frame count.
    assert result.crops.ndim == 3
    assert result.crops.shape[1:] == (96, 96)
    assert result.crops.dtype == np.uint8

    assert result.valid.ndim == 1
    assert result.valid.dtype == np.bool_
    assert len(result.valid) == len(result.crops)

    assert result.frame_indices.ndim == 1
    assert len(result.frame_indices) == len(result.crops)

    # At least one artificially blanked frame should fail detection.
    assert np.any(~result.valid), (
        "The blanked clip did not produce any invalid frames"
    )

    # At least one successful detection must exist so that failed frames
    # can receive interpolated boxes.
    assert np.any(result.valid), (
        "No successful face detections were found in the clip"
    )

    # The crop function must still produce a crop for every failed frame.
    invalid_indices = np.flatnonzero(~result.valid)

    assert result.boxes is not None

    for index in invalid_indices:
        assert np.isfinite(result.boxes[index]).all()
        assert result.boxes[index, 2] > 0

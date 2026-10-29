
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


OUTPUT_SIZE = (96, 96)
SMOOTHING_RADIUS = 2  # centred 5-frame moving average
MAX_INVALID_FRACTION = 0.10


@dataclass
class CropResult:
    crops: np.ndarray
    valid: np.ndarray
    frame_indices: np.ndarray
    boxes: np.ndarray | None = None
    exclude_utterance: bool = False

    @property
    def invalid_fraction(self) -> float:
        """Fraction of returned frames marked invalid."""
        if len(self.valid) == 0:
            return 0.0
        return float(np.count_nonzero(~self.valid) / len(self.valid))


def get_lip_indices() -> np.ndarray:
    """Derive lip landmark indices from MediaPipe's official connections."""
    connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS

    indices = sorted(
        {
            index
            for connection in connections
            for index in (connection.start, connection.end)
        }
    )

    return np.asarray(indices, dtype=np.int32)


def centred_moving_average(
    values: np.ndarray,
    radius: int = SMOOTHING_RADIUS,
) -> np.ndarray:
    """Apply an edge-padded centred moving average to N x D values."""
    if len(values) == 0:
        return values.copy()

    width = 2 * radius + 1
    padded = np.pad(
        values,
        ((radius, radius), (0, 0)),
        mode="edge",
    )

    kernel = np.ones(width, dtype=np.float32) / width

    smoothed_columns = [
        np.convolve(
            padded[:, column],
            kernel,
            mode="valid",
        )
        for column in range(values.shape[1])
    ]

    return np.stack(smoothed_columns, axis=1)


def fill_missing_boxes(
    boxes: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray | None:
    """Interpolate failed detections from the nearest successful boxes.

    Interior gaps are linearly interpolated between the surrounding successful
    frames. Leading and trailing gaps use the nearest successful box, which
    keeps every decoded frame cropable without inventing a zero-sized box.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)

    if boxes.ndim != 2 or boxes.shape[1] != 3:
        raise ValueError("boxes must have shape (N, 3)")
    if len(boxes) != len(valid):
        raise ValueError("boxes and valid must have the same length")
    if not np.any(valid):
        return None

    filled = boxes.copy()
    frame_numbers = np.arange(len(boxes))
    valid_numbers = np.flatnonzero(valid)

    for column in range(boxes.shape[1]):
        filled[:, column] = np.interp(
            frame_numbers,
            valid_numbers,
            filled[valid_numbers, column],
        )

    return filled


def detect_lip_box(
    face_landmarks,
    width: int,
    height: int,
    lip_indices: np.ndarray,
) -> np.ndarray | None:
    """Return [center_x, center_y, square_side] in pixel coordinates."""
    points = np.asarray(
        [
            [
                face_landmarks[index].x * width,
                face_landmarks[index].y * height,
            ]
            for index in lip_indices
        ],
        dtype=np.float32,
    )

    if len(points) == 0 or not np.isfinite(points).all():
        return None

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    lip_width = x_max - x_min
    lip_height = y_max - y_min
    side = max(lip_width, lip_height)

    if side <= 0:
        return None

    side *= 1.25  # 25% margin

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    return np.asarray(
        [center_x, center_y, side],
        dtype=np.float32,
    )


def crop_with_edge_padding(
    frame_rgb: np.ndarray,
    center_x: float,
    center_y: float,
    side: float,
) -> np.ndarray:
    """Crop a square with replicated edge padding and return grayscale uint8."""
    height, width = frame_rgb.shape[:2]

    side_pixels = max(1, int(round(side)))

    x0 = int(round(center_x - side_pixels / 2))
    y0 = int(round(center_y - side_pixels / 2))
    x1 = x0 + side_pixels
    y1 = y0 + side_pixels

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)

    if pad_left or pad_top or pad_right or pad_bottom:
        frame_rgb = cv2.copyMakeBorder(
            frame_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    crop_rgb = frame_rgb[y0:y1, x0:x1]

    crop_rgb = cv2.resize(
        crop_rgb,
        OUTPUT_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    crop_gray = cv2.cvtColor(
        crop_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    return np.asarray(crop_gray, dtype=np.uint8)


def _empty_crops(count: int) -> np.ndarray:
    return np.zeros(
        (count, OUTPUT_SIZE[1], OUTPUT_SIZE[0]),
        dtype=np.uint8,
    )


def _align_frames_and_boxes(
    frames: list[np.ndarray],
    boxes: np.ndarray,
    valid: np.ndarray,
    target_frame_count: int | None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Align frames/boxes to the requested audio-derived frame count.

    Truncation keeps the earliest frames. Padding repeats the last successful
    face frame and its box, while setting padded validity to False.
    Frame indices remain contiguous and no frames are silently removed.
    """
    count = len(frames)
    if target_frame_count is None:
        target_frame_count = count
    if target_frame_count < 0:
        raise ValueError("target_frame_count must be non-negative")

    if count == 0:
        return [], np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool), np.empty(0, dtype=np.int64)

    if target_frame_count <= count:
        end = target_frame_count
        return (
            frames[:end],
            boxes[:end],
            valid[:end],
            np.arange(end, dtype=np.int64),
        )

    extra = target_frame_count - count
    valid_numbers = np.flatnonzero(valid)
    repeat_index = int(valid_numbers[-1]) if len(valid_numbers) else count - 1
    aligned_frames = frames + [frames[repeat_index].copy() for _ in range(extra)]
    aligned_boxes = np.concatenate(
        [boxes, np.repeat(boxes[repeat_index][None, :], extra, axis=0)],
        axis=0,
    )
    aligned_valid = np.concatenate(
        [valid, np.zeros(extra, dtype=bool)],
        axis=0,
    )
    return (
        aligned_frames,
        aligned_boxes,
        aligned_valid,
        np.arange(target_frame_count, dtype=np.int64),
    )


def process_frames(
    frames_rgb: list[np.ndarray],
    boxes: list[np.ndarray | None] | np.ndarray,
    target_frame_count: int | None = None,
) -> CropResult:
    """Crop already-decoded RGB frames from detected or missing lip boxes.

    ``boxes[i]`` is either ``[center_x, center_y, side]`` or ``None``. A
    missing box is invalid but receives an interpolated crop box whenever at
    least one successful detection exists. ``target_frame_count`` is the
    audio-derived number of video frames; the output is padded or truncated
    to that exact length without dropping alignment information.
    """
    if len(frames_rgb) != len(boxes):
        raise ValueError("frames_rgb and boxes must have the same length")

    raw_boxes = np.asarray(
        [
            np.zeros(3, dtype=np.float32) if box is None else box
            for box in boxes
        ],
        dtype=np.float32,
    ).reshape((-1, 3))
    valid = np.any(raw_boxes != 0, axis=1)

    filled_boxes = fill_missing_boxes(raw_boxes, valid)
    if filled_boxes is None:
        # No face was detected anywhere. Preserve frames, but produce black
        # crops and an all-invalid mask rather than fabricating a box.
        aligned_frames = list(frames_rgb)
        if target_frame_count is not None:
            aligned_frames = aligned_frames[:target_frame_count]
            if len(aligned_frames) < target_frame_count and aligned_frames:
                aligned_frames += [aligned_frames[-1].copy()] * (target_frame_count - len(aligned_frames))
        output_valid = np.zeros(len(aligned_frames), dtype=bool)
        output_indices = np.arange(len(aligned_frames), dtype=np.int64)
        return CropResult(
            _empty_crops(len(aligned_frames)),
            output_valid,
            output_indices,
            None,
            bool(len(output_valid) > 0),
        )

    aligned_frames, aligned_boxes, aligned_valid, frame_indices = _align_frames_and_boxes(
        list(frames_rgb),
        filled_boxes,
        valid,
        target_frame_count,
    )

    smoothed_boxes = centred_moving_average(aligned_boxes)
    crops = np.stack(
        [
            crop_with_edge_padding(frame, *box)
            for frame, box in zip(aligned_frames, smoothed_boxes)
        ],
        axis=0,
    ) if aligned_frames else _empty_crops(0)

    invalid_fraction = (
        float(np.count_nonzero(~aligned_valid) / len(aligned_valid))
        if len(aligned_valid)
        else 0.0
    )

    return CropResult(
        crops=crops,
        valid=aligned_valid,
        frame_indices=frame_indices,
        boxes=smoothed_boxes,
        exclude_utterance=invalid_fraction > MAX_INVALID_FRACTION,
    )


def crop_video(
    video_path: str | Path,
    model_path: str | Path,
    target_frame_count: int | None = None,
    audio_frame_count: int | None = None,
) -> CropResult:
    """Read a video and return aligned, stabilized grayscale mouth crops.

    ``target_frame_count`` is the audio-derived number of video/encoder frames.
    ``audio_frame_count`` is an equivalent alias for callers that name the
    source explicitly; passing both is an error. If omitted, one output entry
    is returned per decoded video frame.

    The output never drops a decoded frame silently. Failed detections are
    interpolated when possible and marked invalid. Padding repeats the last
    successful face frame and marks the padded entries invalid. ``exclude_utterance``
    is true when more than 10% of returned frames are invalid.
    """
    if target_frame_count is not None and audio_frame_count is not None:
        raise ValueError("pass only one of target_frame_count or audio_frame_count")
    if audio_frame_count is not None:
        target_frame_count = audio_frame_count

    video_path = Path(video_path)
    model_path = Path(model_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    lip_indices = get_lip_indices()

    frames: list[np.ndarray] = []
    boxes: list[np.ndarray | None] = []

    try:
        base_options = python.BaseOptions(model_asset_path=str(model_path))

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            frame_index = 0

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                box = None
                if result.face_landmarks:
                    box = detect_lip_box(
                        result.face_landmarks[0],
                        frame_rgb.shape[1],
                        frame_rgb.shape[0],
                        lip_indices,
                    )

                frames.append(frame_rgb)
                boxes.append(box)
                frame_index += 1
    finally:
        cap.release()

    return process_frames(
        frames_rgb=frames,
        boxes=boxes,
        target_frame_count=target_frame_count,
    )


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


OUTPUT_SIZE = (96, 96)
SMOOTHING_RADIUS = 2  # centered 5-frame moving average


@dataclass
class CropResult:
    crops: np.ndarray
    valid: np.ndarray
    frame_indices: np.ndarray


def get_lip_indices() -> np.ndarray:
    """Derive lip landmark indices from MediaPipe's official connections."""
    connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS

    indices = sorted({
        index
        for connection in connections
        for index in (connection.start, connection.end)
    })

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
    """Interpolate failed detections using neighbouring valid boxes."""
    if not np.any(valid):
        return None

    filled = boxes.astype(np.float32, copy=True)
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


def crop_video(
    video_path: str | Path,
    model_path: str | Path,
) -> CropResult:
    """Read a video and return stabilized grayscale mouth crops.

    The returned arrays contain one entry per decoded frame:

    crops:
        Shape (N, 96, 96), dtype uint8.

    valid:
        Shape (N,), dtype bool. False means that MediaPipe failed to
        detect the face/lips in the original frame. The crop for that
        frame is generated using an interpolated box when possible.

    frame_indices:
        Original OpenCV frame indices, starting at zero.
    """
    video_path = Path(video_path)
    model_path = Path(model_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    lip_indices = get_lip_indices()

    frames = []
    boxes = []
    frame_indices = []

    try:
        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

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

                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                timestamp_ms = int(round(frame_index * 1000.0 / fps))

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                box = None

                if result.face_landmarks:
                    box = detect_lip_box(
                        result.face_landmarks[0],
                        frame_rgb.shape[1],
                        frame_rgb.shape[0],
                        lip_indices,
                    )

                frames.append(frame_rgb)
                boxes.append(
                    np.zeros(3, dtype=np.float32)
                    if box is None
                    else box
                )
                frame_indices.append(frame_index)

                frame_index += 1

    finally:
        cap.release()

    valid = np.asarray(
        [np.any(box) for box in boxes],
        dtype=bool,
    )

    frame_indices = np.asarray(
        frame_indices,
        dtype=np.int64,
    )

    if not frames:
        empty = np.empty(
            (0, OUTPUT_SIZE[1], OUTPUT_SIZE[0]),
            dtype=np.uint8,
        )
        return CropResult(empty, valid, frame_indices)

    raw_boxes = np.asarray(boxes, dtype=np.float32)
    filled_boxes = fill_missing_boxes(raw_boxes, valid)

    if filled_boxes is None:
        crops = np.zeros(
            (len(frames), OUTPUT_SIZE[1], OUTPUT_SIZE[0]),
            dtype=np.uint8,
        )
        return CropResult(crops, valid, frame_indices)

    smoothed_boxes = centred_moving_average(filled_boxes)

    crops = np.stack(
        [
            crop_with_edge_padding(frame, *box)
            for frame, box in zip(frames, smoothed_boxes)
        ],
        axis=0,
    )

    return CropResult(crops, valid, frame_indices)

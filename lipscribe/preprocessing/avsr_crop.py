"""Auto-AVSR-style mouth crops from MediaPipe face-mesh landmarks.

Re-implements the crop of Auto-AVSR's preparation code (mpc001/auto_avsr @ 182b628,
preparation/detectors/retinaface/video_process.py), which the pretrained lip-reading encoder
was trained on:

1. landmarks per frame; frames without a face are linearly interpolated (ends: nearest);
2. each frame's landmarks are smoothed over +-6 frames and re-centred on the frame's own mean;
3. a similarity transform (cv2.estimateAffinePartial2D, LMEDS) maps the 8 stable points
   (iBUG-68 ids 28, 33, 36, 39, 42, 45, 48, 54) onto the 20-words mean face in a 256x256 frame;
4. the greyscale frame is warped to 256x256 and a 96x96 patch is cut around the mean of the
   20 mouth points (iBUG 48-67).

Auto-AVSR uses RetinaFace + a 68-point FAN; we map MediaPipe's 478-point mesh onto the iBUG
ids we need (``IBUG_TO_MP``), so no extra detector is needed. The reference mean face
(``20words_mean_face.npy``, 68x2) is downloaded to ``pretrained/avsr_prep/``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MEAN_FACE = Path("pretrained/avsr_prep/retinaface_20words_mean_face.npy")
STABLE = (28, 33, 36, 39, 42, 45, 48, 54)
MOUTH = tuple(range(48, 68))
# iBUG-68 id -> MediaPipe face-mesh id (image-left = subject's right, as in both schemes)
IBUG_TO_MP = {
    28: 197, 33: 2,                                   # nose bridge, subnasale
    36: 33, 39: 133, 42: 362, 45: 263,                # eye corners (outer R, inner R, inner L, outer L)
    48: 61, 49: 40, 50: 37, 51: 0, 52: 267, 53: 270,  # outer lip, top: corner -> corner
    54: 291, 55: 321, 56: 314, 57: 17, 58: 84, 59: 91,  # outer lip, bottom
    60: 78, 61: 81, 62: 13, 63: 311, 64: 308, 65: 402, 66: 14, 67: 178,  # inner lip
}
IDS = tuple(sorted(set(STABLE) | set(MOUTH)))         # the 28 iBUG ids we carry
_POS = {k: i for i, k in enumerate(IDS)}
_MP = np.array([IBUG_TO_MP[k] for k in IDS])
CROP, STD, WINDOW = 96, 256, 12


def mesh_points(face, w: int, h: int) -> np.ndarray:
    """MediaPipe landmark list -> [28, 2] pixel coordinates in ``IDS`` order."""
    return np.array([[face[i].x * w, face[i].y * h] for i in _MP], dtype=np.float64)


def pick_face(faces, w: int, h: int, prev: np.ndarray | None) -> np.ndarray | None:
    """The face whose mouth is nearest the previous frame's (else the widest face)."""
    if not faces:
        return None
    pts = [mesh_points(f, w, h) for f in faces]
    if prev is None:
        return max(pts, key=lambda p: np.ptp(p[:, 0]))
    m0 = prev[[_POS[k] for k in MOUTH]].mean(0)
    return min(pts, key=lambda p: np.linalg.norm(p[[_POS[k] for k in MOUTH]].mean(0) - m0))


def interpolate(lms: list[np.ndarray | None]) -> list[np.ndarray] | None:
    idx = [i for i, l in enumerate(lms) if l is not None]
    if not idx:
        return None
    out = list(lms)
    for a, b in zip(idx, idx[1:]):
        for k in range(1, b - a):
            out[a + k] = out[a] + k / (b - a) * (out[b] - out[a])
    out[:idx[0]] = [out[idx[0]]] * idx[0]
    out[idx[-1]:] = [out[idx[-1]]] * (len(out) - idx[-1])
    return out


_REF: np.ndarray | None = None


def _stable_reference() -> np.ndarray:
    global _REF
    if _REF is None:
        ref = np.load(MEAN_FACE)
        _REF = np.vstack([ref[k] for k in STABLE]).astype(np.float64)   # 256 -> 256: no shift
    return _REF


def crop_sequence(rgbs: list[np.ndarray], lms: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Frames (RGB) + interpolated landmarks -> ([T, 96, 96] uint8 grey, n frames that overflowed).

    Auto-AVSR drops a clip when the mouth centre lands more than 53 px from the warped frame's
    centre; we instead clamp the patch (edge-padded) and report how often it happened.
    """
    ref = _stable_reference()
    st = [_POS[k] for k in STABLE]; mo = [_POS[k] for k in MOUTH]
    half, n, out, over = CROP // 2, len(rgbs), [], 0
    for i, rgb in enumerate(rgbs):
        m = min(WINDOW // 2, i, n - 1 - i)
        sm = np.mean(lms[i - m:i + m + 1], axis=0)
        sm = sm + (lms[i].mean(0) - sm.mean(0))
        tf, _ = cv2.estimateAffinePartial2D(sm[st], ref, method=cv2.LMEDS)
        if tf is None:                                 # degenerate points: keep the last transform
            tf = last_tf if i else cv2.estimateAffinePartial2D(sm[st], ref)[0]
        last_tf = tf
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        warped = cv2.warpAffine(grey, tf, (STD, STD), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        c = sm[mo] @ tf[:, :2].T + tf[:, 2]
        cx, cy = c.mean(0)
        if abs(cy - STD / 2) > half + 5 or abs(cx - STD / 2) > half + 5:
            over += 1
        x0 = int(round(np.clip(cx - half, 0, STD - CROP))); y0 = int(round(np.clip(cy - half, 0, STD - CROP)))
        out.append(warped[y0:y0 + CROP, x0:x0 + CROP])
    return np.stack(out).astype(np.uint8), over

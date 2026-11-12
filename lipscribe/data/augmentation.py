"""Visual data augmentation for video clips (W3-04).

Applies training-only augmentations to video clips while keeping the motion pattern
intact by applying the identical spatial transform to every frame of a clip.
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
try:
    import torch
except ImportError:
    torch = None


def _sample_int(low: int, high: int, rng: random.Random | np.random.Generator | None = None) -> int:
    """Sample an integer in [low, high] inclusive."""
    if rng is None:
        return random.randint(low, high)
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(low, high + 1))
    if isinstance(rng, random.Random):
        return rng.randint(low, high)
    raise TypeError(f"Unsupported RNG type: {type(rng)}")


def _sample_float(rng: random.Random | np.random.Generator | None = None) -> float:
    """Sample a float in [0.0, 1.0)."""
    if rng is None:
        return random.random()
    if isinstance(rng, np.random.Generator):
        return float(rng.random())
    if isinstance(rng, random.Random):
        return rng.random()
    raise TypeError(f"Unsupported RNG type: {type(rng)}")


def random_crop_jitter(
    frames: np.ndarray | torch.Tensor,
    max_jitter: int = 4,
    rng: random.Random | np.random.Generator | None = None,
    pad_mode: str = "edge",
) -> np.ndarray | torch.Tensor:
    """Apply random crop jitter with the same spatial offset across all frames of a clip.

    Args:
        frames: Array or Tensor of shape [T, H, W] or [T, C, H, W] or [1, T, H, W].
        max_jitter: Maximum pixel displacement along height and width (default: 4).
        rng: Optional random.Random or np.random.Generator instance for reproducibility.
        pad_mode: Padding mode for boundary extension ("edge" / replicate).

    Returns:
        Augmented clip with the exact same shape and dtype as the input.
    """
    if max_jitter <= 0:
        return frames

    is_torch = torch is not None and isinstance(frames, torch.Tensor)
    arr = frames.detach().cpu().numpy() if is_torch else np.asarray(frames)

    orig_ndim = arr.ndim
    if orig_ndim == 3:
        # [T, H, W]
        t, h, w = arr.shape
        c_dim = False
    elif orig_ndim == 4:
        # [T, C, H, W] or [C, T, H, W]
        # In LipScribe convention, video items are [T, H, W] or batch [1, T, H, W]
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            # [1, T, H, W]
            arr = arr[0]
            t, h, w = arr.shape
            c_dim = "leading_1"
        else:
            t, c, h, w = arr.shape
            c_dim = True
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")

    # Sample a single vertical and horizontal offset for the entire clip
    offset_y = _sample_int(0, 2 * max_jitter, rng)
    offset_x = _sample_int(0, 2 * max_jitter, rng)

    if orig_ndim == 3 or c_dim == "leading_1":
        padded = np.pad(
            arr,
            ((0, 0), (max_jitter, max_jitter), (max_jitter, max_jitter)),
            mode=pad_mode,
        )
        cropped = padded[:, offset_y : offset_y + h, offset_x : offset_x + w]
        if c_dim == "leading_1":
            cropped = cropped[np.newaxis, ...]
    else:
        padded = np.pad(
            arr,
            ((0, 0), (0, 0), (max_jitter, max_jitter), (max_jitter, max_jitter)),
            mode=pad_mode,
        )
        cropped = padded[:, :, offset_y : offset_y + h, offset_x : offset_x + w]

    cropped = np.ascontiguousarray(cropped, dtype=arr.dtype)
    if is_torch:
        return torch.from_numpy(cropped).to(device=frames.device, dtype=frames.dtype)
    return cropped


def random_horizontal_flip(
    frames: np.ndarray | torch.Tensor,
    p: float = 0.5,
    rng: random.Random | np.random.Generator | None = None,
) -> np.ndarray | torch.Tensor:
    """Randomly horizontally flip all frames of a clip with probability p.

    The flip decision is sampled once per clip so all frames receive the identical transform.

    Args:
        frames: Array or Tensor of shape [T, H, W] or [T, C, H, W] or [1, T, H, W].
        p: Probability of applying horizontal flip (default: 0.5).
        rng: Optional random.Random or np.random.Generator instance for reproducibility.

    Returns:
        Augmented clip (either all frames flipped horizontally, or none).
    """
    if p <= 0.0:
        return frames

    if _sample_float(rng) >= p:
        return frames

    is_torch = torch is not None and isinstance(frames, torch.Tensor)
    if is_torch:
        return torch.flip(frames, dims=[-1])

    # Flip along horizontal axis (last dimension)
    flipped = np.ascontiguousarray(frames[..., ::-1])
    return flipped


def random_scale(
    frames: np.ndarray,
    max_scale: float = 0.1,
    rng: random.Random | np.random.Generator | None = None,
) -> np.ndarray:
    """Zoom the whole clip in or out by up to ``max_scale`` (one factor per clip).

    Zoom-in crops the centre and resizes back to the original size; zoom-out
    shrinks the clip and edge-pads it. Simulates speakers sitting nearer or
    further from the camera. Works on ``[T, H, W]`` uint8 or float arrays.
    """
    if max_scale <= 0:
        return frames
    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError(f"random_scale expects [T, H, W], got {arr.shape}")
    factor = 1.0 + (2.0 * _sample_float(rng) - 1.0) * max_scale
    t, h, w = arr.shape
    nh, nw = max(8, int(round(h * factor))), max(8, int(round(w * factor)))
    if (nh, nw) == (h, w):
        return frames
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(1)
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)[:, 0].numpy()
    if factor > 1.0:
        top, left = (nh - h) // 2, (nw - w) // 2
        out = x[:, top:top + h, left:left + w]
    else:
        ph, pw = h - nh, w - nw
        out = np.pad(x, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), mode="edge")
    if np.issubdtype(arr.dtype, np.integer):
        out = np.clip(np.rint(out), 0, 255)
    return np.ascontiguousarray(out, dtype=arr.dtype)


def random_time_mask(
    frames: np.ndarray,
    n_masks: int = 2,
    max_width: int = 6,
    rng: random.Random | np.random.Generator | None = None,
) -> np.ndarray:
    """Replace ``n_masks`` random runs of up to ``max_width`` frames by the clip's mean frame.

    The transcript is unchanged, so the model has to infer the hidden mouth
    shapes from context (Ma et al., "Towards practical lipreading", 2022).
    """
    arr = np.asarray(frames)
    t = arr.shape[0]
    if n_masks <= 0 or max_width <= 0 or t < 2:
        return frames
    out = arr.copy()
    mean_frame = arr.mean(axis=0).astype(arr.dtype)
    for _ in range(n_masks):
        width = _sample_int(0, min(max_width, t - 1), rng)
        if width == 0:
            continue
        start = _sample_int(0, t - width, rng)
        out[start:start + width] = mean_frame
    return out


def random_erase(
    frames: np.ndarray,
    p: float = 0.5,
    max_area: float = 0.2,
    rng: random.Random | np.random.Generator | None = None,
) -> np.ndarray:
    """Cutout: with probability ``p``, fill one rectangle (same for every frame) with the clip mean."""
    if p <= 0 or _sample_float(rng) >= p:
        return frames
    arr = np.asarray(frames)
    t, h, w = arr.shape[-3:]
    area = (0.02 + _sample_float(rng) * (max_area - 0.02)) * h * w
    aspect = 0.5 + _sample_float(rng) * 1.5
    eh = int(min(h, max(1, round((area * aspect) ** 0.5))))
    ew = int(min(w, max(1, round((area / aspect) ** 0.5))))
    top, left = _sample_int(0, h - eh, rng), _sample_int(0, w - ew, rng)
    out = arr.copy()
    out[..., top:top + eh, left:left + ew] = arr.mean().astype(arr.dtype)
    return out


def apply_visual_augmentation(
    frames: np.ndarray | torch.Tensor,
    max_jitter: int = 4,
    p_flip: float = 0.5,
    rng: random.Random | np.random.Generator | None = None,
    max_scale: float = 0.0,
    time_masks: int = 0,
    max_mask_width: int = 6,
    p_erase: float = 0.0,
    max_erase_area: float = 0.2,
) -> np.ndarray | torch.Tensor:
    """Apply the clip-level visual augmentations; every frame gets the same spatial transform.

    Order: scale jitter -> crop jitter -> horizontal flip -> cutout -> time masking.
    With the defaults only crop jitter and flip run (the original W3-04 behaviour).

    Args:
        frames: Clip array or Tensor [T, H, W] or [T, C, H, W].
        max_jitter: Maximum crop displacement in pixels (default: 4).
        p_flip: Probability of horizontal flip (default: 0.5).
        rng: Optional random generator for deterministic/worker-seeded transforms.
        max_scale: Maximum zoom in/out as a fraction (0.1 = +-10 %); 0 disables.
        time_masks: Number of time masks; 0 disables.
        max_mask_width: Maximum frames per time mask.
        p_erase: Probability of one cutout rectangle; 0 disables.
        max_erase_area: Maximum cutout area as a fraction of the frame.

    Returns:
        Augmented clip.
    """
    out = frames
    if max_scale > 0:
        out = random_scale(out, max_scale=max_scale, rng=rng)
    out = random_crop_jitter(out, max_jitter=max_jitter, rng=rng)
    out = random_horizontal_flip(out, p=p_flip, rng=rng)
    if p_erase > 0:
        out = random_erase(out, p=p_erase, max_area=max_erase_area, rng=rng)
    if time_masks > 0:
        out = random_time_mask(out, n_masks=time_masks, max_width=max_mask_width, rng=rng)
    return out

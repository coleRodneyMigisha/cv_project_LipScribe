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


def apply_visual_augmentation(
    frames: np.ndarray | torch.Tensor,
    max_jitter: int = 4,
    p_flip: float = 0.5,
    rng: random.Random | np.random.Generator | None = None,
) -> np.ndarray | torch.Tensor:
    """Apply both random crop jitter and horizontal flip to a clip with consistent transforms.

    Args:
        frames: Clip array or Tensor [T, H, W] or [T, C, H, W].
        max_jitter: Maximum crop displacement in pixels (default: 4).
        p_flip: Probability of horizontal flip (default: 0.5).
        rng: Optional random generator for deterministic/worker-seeded transforms.

    Returns:
        Augmented clip.
    """
    out = random_crop_jitter(frames, max_jitter=max_jitter, rng=rng)
    out = random_horizontal_flip(out, p=p_flip, rng=rng)
    return out


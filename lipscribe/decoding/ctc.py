"""CTC collapse and greedy (best-path) decoding.

CTC maps a frame-level path to text by
    1. merging consecutive repeated symbols, then
    2. removing blanks (index 0).
So "b b _ i _ n n" -> "bin", and a doubled letter needs a blank between
its copies: "e _ e" -> "ee" but "e e" -> "e" (implementation plan, Section 7).

Beam search (W3-07) will live next to this file and return the same type.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch

from lipscribe.data.tokenizer import BLANK_ID, decode


def collapse_ids(ids: Iterable[int], blank: int = BLANK_ID) -> list[int]:
    """Merge repeats, then drop blanks. Returns symbol ids."""
    out: list[int] = []
    prev = None
    for i in ids:
        i = int(i)
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def collapse(ids: Iterable[int], blank: int = BLANK_ID) -> str:
    """Frame-level path -> text, e.g. ids for [b,b,0,i,0,n,n] -> "bin"."""
    return decode(collapse_ids(ids, blank))


def greedy_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor | Sequence[int] | None = None,
    blank: int = BLANK_ID,
) -> list[str]:
    """Best-path decoding of a batch.

    Args:
        log_probs: [T, B, V] (time-first, as returned by CTCHead).
        lengths: true number of frames per item [B]; padded frames are ignored.
            If None, all T frames are used for every item.

    Returns:
        list of B strings.
    """
    if log_probs.dim() != 3:
        raise ValueError(f"greedy_decode expects [T, B, V], got {tuple(log_probs.shape)}")
    T, B, _ = log_probs.shape
    best = log_probs.argmax(dim=-1).transpose(0, 1).cpu()  # [B, T]
    if lengths is None:
        lens = [T] * B
    else:
        lens = [int(n) for n in (lengths.tolist() if torch.is_tensor(lengths) else lengths)]
    return [collapse(best[b, : lens[b]].tolist(), blank) for b in range(B)]

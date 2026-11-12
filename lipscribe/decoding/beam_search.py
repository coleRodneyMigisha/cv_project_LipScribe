"""CTC prefix beam search (W3-07, implementation document Section 7 / Algorithm 2).

Maintains blank and non-blank probabilities per prefix in log space.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch

from lipscribe.data.tokenizer import BLANK_ID, decode


def _logaddexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    diff = a - b
    if diff > 0:
        return a + math.log1p(math.exp(-diff))
    else:
        return b + math.log1p(math.exp(diff))


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    beam_width: int = 10,
    blank: int = BLANK_ID,
    length: int | None = None,
) -> str:
    """CTC prefix beam search in log space for a single utterance (Algorithm 2).

    Args:
        log_probs: [T, V] log-probabilities for one utterance (float tensor).
        beam_width: number of top prefixes to maintain at each time step (W >= 1).
        blank: blank token index (default 0).
        length: true unpadded length T_valid (if None, uses log_probs.shape[0]).

    Returns:
        Decoded text string corresponding to the highest-probability prefix.
    """
    if log_probs.dim() != 2:
        raise ValueError(f"Expected [T, V], got shape {tuple(log_probs.shape)}")

    T, V = log_probs.shape
    if length is not None:
        T = min(T, int(length))

    lp = log_probs[:T].detach().cpu().numpy()

    # Prefixes represented as tuples of token ids
    # p_b: log prob ending in blank; p_nb: log prob ending in non-blank
    p_b: dict[tuple[int, ...], float] = {(): 0.0}
    p_nb: dict[tuple[int, ...], float] = {(): -math.inf}

    for t in range(T):
        next_p_b: dict[tuple[int, ...], float] = defaultdict(lambda: -math.inf)
        next_p_nb: dict[tuple[int, ...], float] = defaultdict(lambda: -math.inf)

        log_y_blank = float(lp[t, blank])

        # Step 1: Blank emission (does not extend prefix)
        for prefix in p_b:
            p_total = _logaddexp(p_b[prefix], p_nb[prefix])
            next_p_b[prefix] = _logaddexp(next_p_b[prefix], p_total + log_y_blank)

        # Step 2: Non-blank emissions
        for k in range(V):
            if k == blank:
                continue
            log_y_k = float(lp[t, k])

            for prefix in p_b:
                if len(prefix) > 0 and prefix[-1] == k:
                    # Repeated symbol:
                    # If prior was non-blank, repeat collapses to the same prefix
                    next_p_nb[prefix] = _logaddexp(next_p_nb[prefix], p_nb[prefix] + log_y_k)
                    # If prior was blank, repeat extends the prefix
                    new_prefix = prefix + (k,)
                    next_p_nb[new_prefix] = _logaddexp(next_p_nb[new_prefix], p_b[prefix] + log_y_k)
                else:
                    new_prefix = prefix + (k,)
                    p_total = _logaddexp(p_b[prefix], p_nb[prefix])
                    next_p_nb[new_prefix] = _logaddexp(next_p_nb[new_prefix], p_total + log_y_k)

        candidates = set(next_p_b.keys()) | set(next_p_nb.keys())

        def score(pref: tuple[int, ...]) -> float:
            return _logaddexp(next_p_b[pref], next_p_nb[pref])

        # Keep top beam_width prefixes
        sorted_prefixes = sorted(candidates, key=score, reverse=True)[:beam_width]

        p_b = {pref: next_p_b[pref] for pref in sorted_prefixes}
        p_nb = {pref: next_p_nb[pref] for pref in sorted_prefixes}

    best_prefix = max(p_b.keys(), key=lambda pref: _logaddexp(p_b[pref], p_nb[pref]))
    return decode(list(best_prefix))


def beam_search_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor | Sequence[int] | None = None,
    beam_width: int = 10,
    blank: int = BLANK_ID,
) -> list[str]:
    """Batch CTC prefix beam search decoding.

    Args:
        log_probs: [T, B, V] (time-first layout).
        lengths: per-item lengths [B].
        beam_width: beam width (default 10).
        blank: blank token index.

    Returns:
        List of decoded strings of length B.
    """
    if log_probs.dim() != 3:
        raise ValueError(f"Expected [T, B, V], got shape {tuple(log_probs.shape)}")

    T, B, _ = log_probs.shape
    results = []
    for b in range(B):
        l = lengths[b].item() if lengths is not None and torch.is_tensor(lengths) else (lengths[b] if lengths is not None else None)
        hyp = ctc_prefix_beam_search(log_probs[:, b, :], beam_width=beam_width, blank=blank, length=l)
        results.append(hyp)
    return results

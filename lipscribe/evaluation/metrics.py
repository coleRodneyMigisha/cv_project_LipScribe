"""Word and character error rates (WER, CER) at corpus level.

WER = (S + D + I) / N, where N is the number of reference words.
Corpus level means errors and reference lengths are summed over all
utterances first and divided once at the end; it is NOT the mean of
per-utterance rates (implementation plan, Section 9; W2-12 checklist).
"""
from __future__ import annotations

from typing import Sequence

import jiwer


def _check(refs: Sequence[str], hyps: Sequence[str]) -> tuple[list[str], list[str]]:
    if isinstance(refs, str) or isinstance(hyps, str):
        refs, hyps = [refs], [hyps]
    refs, hyps = list(refs), list(hyps)
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    if not refs:
        raise ValueError("No utterances to score")
    if any(not r.strip() for r in refs):
        raise ValueError("Empty reference transcript; WER/CER are undefined")
    return refs, hyps


def error_counts(refs: Sequence[str], hyps: Sequence[str]) -> dict:
    """Corpus-level word and character error counts, for logging."""
    refs, hyps = _check(refs, hyps)
    w = jiwer.process_words(refs, hyps)
    c = jiwer.process_characters(refs, hyps)
    n_words = sum(len(r.split()) for r in refs)
    n_chars = sum(len(r) for r in refs)
    return {
        "word_sub": w.substitutions, "word_del": w.deletions, "word_ins": w.insertions,
        "n_words": n_words,
        "char_sub": c.substitutions, "char_del": c.deletions, "char_ins": c.insertions,
        "n_chars": n_chars,
    }


def wer(refs: Sequence[str] | str, hyps: Sequence[str] | str) -> float:
    """Corpus-level word error rate."""
    k = error_counts(refs, hyps)
    return (k["word_sub"] + k["word_del"] + k["word_ins"]) / k["n_words"]


def cer(refs: Sequence[str] | str, hyps: Sequence[str] | str) -> float:
    """Corpus-level character error rate (spaces count as characters)."""
    k = error_counts(refs, hyps)
    return (k["char_sub"] + k["char_del"] + k["char_ins"]) / k["n_chars"]

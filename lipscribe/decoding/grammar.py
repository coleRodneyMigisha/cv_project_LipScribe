"""GRID grammar-constrained CTC decoding.

Every GRID sentence has the fixed form

    command  color  preposition  letter  digit  adverb
    (4)      (4)    (4)          (25)    (10)   (4)       -> 64,000 sentences

Unconstrained greedy or prefix-beam decoding can output words that are not in
the vocabulary ("bim", "gren") or the wrong number of words. This decoder only
ever outputs a sentence the grammar allows: it runs Viterbi over a CTC graph
built from the grammar, so the result is the single best CTC alignment of any
valid GRID sentence.

The graph is small (about 130 character states plus their blanks), so decoding
is a dense max-plus recursion over ``[B, S, S]`` per frame and runs on the same
device as the model output.
"""
from __future__ import annotations

from typing import Sequence

import torch

from lipscribe.data.tokenizer import BLANK_ID, SPACE_ID, VOCAB

GRID_GRAMMAR: tuple[tuple[str, ...], ...] = (
    ("bin", "lay", "place", "set"),
    ("blue", "green", "red", "white"),
    ("at", "by", "in", "with"),
    tuple(c for c in "abcdefghijklmnopqrstuvxyz"),  # GRID has no "w"
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    ("again", "now", "please", "soon"),
)

NEG_INF = float("-inf")


class GrammarCTCDecoder:
    """Viterbi CTC decoder restricted to a slot grammar (default: GRID).

    State layout: state 0 is the leading blank; each label node ``i`` (one per
    character of every word, plus one space node between slots) has a label
    state ``1 + 2i`` and a trailing-blank state ``2 + 2i``.
    """

    def __init__(self, grammar: Sequence[Sequence[str]] = GRID_GRAMMAR) -> None:
        labels: list[int] = []         # token id of each label node
        chars: list[str] = []
        preds: list[list[int]] = []    # predecessor label nodes (-1 = sentence start)
        entry: list[int] = [-1]        # label nodes a new slot's words follow
        for s, words in enumerate(grammar):
            last_chars = []
            for word in words:
                prev = entry
                for ch in word:
                    labels.append(VOCAB[ch])
                    chars.append(ch)
                    preds.append(list(prev))
                    prev = [len(labels) - 1]
                last_chars.append(prev[0])
            if s < len(grammar) - 1:
                labels.append(SPACE_ID)
                chars.append(" ")
                preds.append(last_chars)
                entry = [len(labels) - 1]
            else:
                finals = last_chars

        n = len(labels)
        S = 1 + 2 * n
        # trans[dst, src] = 0 if src -> dst is allowed, else -inf.
        trans = torch.full((S, S), NEG_INF)
        trans[0, 0] = 0.0
        initial = torch.full((S,), NEG_INF)
        initial[0] = 0.0
        for i in range(n):
            lab, blk = 1 + 2 * i, 2 + 2 * i
            trans[lab, lab] = 0.0          # stay on the label
            trans[blk, blk] = 0.0          # stay on the blank
            trans[blk, lab] = 0.0          # label -> its trailing blank
            for p in preds[i]:
                if p == -1:
                    trans[lab, 0] = 0.0
                    initial[lab] = 0.0
                else:
                    trans[lab, 2 + 2 * p] = 0.0         # via a blank
                    if labels[p] != labels[i]:
                        trans[lab, 1 + 2 * p] = 0.0     # direct (no repeat)
        final = torch.zeros(S, dtype=torch.bool)
        for f in finals:
            final[1 + 2 * f] = True
            final[2 + 2 * f] = True

        self.trans = trans
        self.initial = initial
        self.final = final
        self.chars = chars
        emit = torch.full((S,), BLANK_ID, dtype=torch.long)
        emit[1::2] = torch.tensor(labels, dtype=torch.long)
        self.emit = emit

    @torch.no_grad()
    def __call__(self, log_probs: torch.Tensor,
                 lengths: torch.Tensor | Sequence[int] | None = None) -> list[str]:
        """Decode time-first ``[T, B, V]`` log-probabilities into ``B`` sentences."""
        if log_probs.dim() != 3:
            raise ValueError(f"Expected [T, B, V], got shape {tuple(log_probs.shape)}")
        T, B, _ = log_probs.shape
        device = log_probs.device
        lp = log_probs.float()
        if lengths is None:
            lens = torch.full((B,), T, dtype=torch.long, device=device)
        else:
            lens = torch.as_tensor(lengths, device=device).long().clamp(1, T)

        trans = self.trans.to(device)
        emit = self.emit.to(device)
        final = self.final.to(device)
        S = trans.shape[0]

        # Emission scores for every state: [T, B, S]
        em = lp.index_select(2, emit)
        score = self.initial.to(device).unsqueeze(0) + em[0]          # [B, S]
        back = torch.zeros(T, B, S, dtype=torch.long, device=device)
        back[0] = torch.arange(S, device=device)
        for t in range(1, T):
            cand = score.unsqueeze(1) + trans.unsqueeze(0)            # [B, S_dst, S_src]
            best, arg = cand.max(dim=2)
            new = best + em[t]
            active = (t < lens).unsqueeze(1)
            score = torch.where(active, new, score)
            back[t] = torch.where(active, arg, torch.arange(S, device=device).expand(B, S))

        end_score = score.masked_fill(~final.unsqueeze(0), NEG_INF)
        last = end_score.argmax(dim=1)
        ok = torch.isfinite(end_score.max(dim=1).values)

        back = back.cpu()
        last = last.cpu().tolist()
        ok = ok.cpu().tolist()
        out = []
        for b in range(B):
            if not ok[b]:
                out.append("")     # sequence too short for any valid sentence
                continue
            path = [last[b]]
            state = last[b]
            for t in range(T - 1, 0, -1):
                state = int(back[t, b, state])
                path.append(state)
            path.reverse()
            text, prev = [], None
            for st in path:
                if st % 2 == 1 and st != prev:
                    text.append(self.chars[(st - 1) // 2])
                prev = st
            out.append("".join(text))
        return out


_DEFAULT: GrammarCTCDecoder | None = None


def grammar_decode(log_probs: torch.Tensor,
                   lengths: torch.Tensor | Sequence[int] | None = None) -> list[str]:
    """Batch GRID grammar-constrained decoding with a cached decoder."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GrammarCTCDecoder()
    return _DEFAULT(log_probs, lengths)

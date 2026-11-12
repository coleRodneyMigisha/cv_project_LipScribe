"""Lexicon + word-bigram language-model CTC beam search (X6 'Flix', open vocabulary).

The GRID grammar decoder (grammar.py) forces six fixed slots; film dialogue has no such
pattern. This is its open-vocabulary counterpart:

* lexicon: a partial word may only grow along a trie of known words, so every output
  word is a real word from the lexicon (out-of-lexicon words cannot be produced);
* language model: each completed word adds ``alpha * log P(word | previous word)``
  plus a word-insertion bonus ``beta`` (counteracts the LM's preference for short output);
* acoustic part: the usual CTC prefix beam search with blank / non-blank probabilities.

The bigram LM uses interpolated absolute discounting (Kneser-Ney-style continuation
counts for the lower order), built from plain text: ``WordBigramLM.from_texts``.
Pure Python + NumPy, no external decoder libraries.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from lipscribe.data.tokenizer import BLANK_ID, ID_TO_CHAR, SPACE_ID  # torch-free, so the LM
# can be built on a machine without PyTorch (scripts/flix_lm.py loads this file directly)

BOS, EOS = "<s>", "</s>"
NEG = -math.inf


def _lae(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


class WordBigramLM:
    """Interpolated absolute-discount bigram LM over a closed vocabulary."""

    def __init__(self, unigram: dict[str, float], bigram: dict[str, dict[str, int]], discount: float = 0.75):
        self.p_uni = unigram                               # continuation probabilities (sum to 1)
        self.big = bigram                                  # counts c(prev, w)
        self.d = discount
        self.ctx_total = {p: sum(ws.values()) for p, ws in bigram.items()}
        self.ctx_types = {p: len(ws) for p, ws in bigram.items()}
        self.vocab = sorted(w for w in unigram if w != EOS)
        self._cache: dict[tuple[str, str], float] = {}

    @classmethod
    def from_texts(cls, texts: Iterable[str], discount: float = 0.75) -> "WordBigramLM":
        big: dict[str, Counter] = defaultdict(Counter)
        for t in texts:
            ws = [BOS] + t.split() + [EOS]
            for a, b in zip(ws, ws[1:]):
                big[a][b] += 1
        cont = Counter(w for ws in big.values() for w in ws)            # number of distinct left contexts
        tot = sum(cont.values())
        return cls({w: c / tot for w, c in cont.items()}, {p: dict(ws) for p, ws in big.items()}, discount)

    def logp(self, prev: str, w: str) -> float:
        key = (prev, w)
        if key not in self._cache:
            pu = self.p_uni.get(w, 1e-9)
            n = self.ctx_total.get(prev, 0)
            if n == 0:
                p = pu
            else:
                c = self.big[prev].get(w, 0)
                p = max(c - self.d, 0) / n + self.d * self.ctx_types[prev] / n * pu
            self._cache[key] = math.log(max(p, 1e-12))
        return self._cache[key]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"discount": self.d, "unigram": self.p_uni, "bigram": self.big}))

    @classmethod
    def load(cls, path: str | Path) -> "WordBigramLM":
        d = json.loads(Path(path).read_text())
        return cls(d["unigram"], d["bigram"], d["discount"])


class LexiconLMDecoder:
    def __init__(self, lm: WordBigramLM, alpha: float = 0.8, beta: float = 1.5,
                 beam_width: int = 32, prune: float = -8.0):
        self.lm, self.alpha, self.beta, self.W, self.prune = lm, alpha, beta, beam_width, prune
        self.words = set(lm.vocab)
        self.prefixes = {w[:i] for w in self.words for i in range(len(w) + 1)}

    def _word_score(self, prev: str, w: str) -> float:
        return self.alpha * self.lm.logp(prev, w) + self.beta

    def decode_one(self, lp: np.ndarray) -> str:
        # key = (completed words, partial word); value = [log p_blank, log p_nonblank, lm score]
        beams: dict[tuple[tuple[str, ...], str], list[float]] = {((), ""): [0.0, NEG, 0.0]}
        for t in range(lp.shape[0]):
            row = lp[t]
            top = row.max()
            ks = [k for k in range(row.shape[0]) if k != BLANK_ID and row[k] >= top + self.prune]
            nxt: dict[tuple[tuple[str, ...], str], list[float]] = {}

            def add(key, pb, pnb, lm):
                e = nxt.get(key)
                if e is None:
                    nxt[key] = [pb, pnb, lm]
                else:
                    e[0] = _lae(e[0], pb); e[1] = _lae(e[1], pnb)

            for (words, part), (pb, pnb, lm) in beams.items():
                tot = _lae(pb, pnb)
                add((words, part), tot + float(row[BLANK_ID]), NEG, lm)        # blank
                last = part[-1] if part else (" " if words else None)
                for k in ks:
                    y = float(row[k])
                    if k == SPACE_ID:
                        if part:
                            if part not in self.words:
                                continue
                            prev = words[-1] if words else BOS
                            add((words + (part,), ""), NEG, tot + y, lm + self._word_score(prev, part))
                        elif last == " ":
                            add((words, part), NEG, pnb + y, lm)                 # repeated space collapses
                        continue
                    c = ID_TO_CHAR[k]
                    if c == last:                                               # repeated character
                        add((words, part), NEG, pnb + y, lm)                    # collapses (no blank between)
                        if part + c in self.prefixes:
                            add((words, part + c), NEG, pb + y, lm)             # blank between -> new char
                        continue
                    if part + c in self.prefixes:
                        add((words, part + c), NEG, tot + y, lm)
            beams = dict(sorted(nxt.items(), key=lambda kv: _lae(kv[1][0], kv[1][1]) + kv[1][2],
                                reverse=True)[: self.W])
        best, best_s = "", NEG
        for (words, part), (pb, pnb, lm) in beams.items():
            s = _lae(pb, pnb) + lm
            ws = words
            if part:
                if part not in self.words:
                    continue
                s += self._word_score(words[-1] if words else BOS, part); ws = words + (part,)
            s += self.alpha * self.lm.logp(ws[-1] if ws else BOS, EOS)
            if s > best_s:
                best, best_s = " ".join(ws), s
        return best

    def __call__(self, log_probs: "torch.Tensor", lengths: "torch.Tensor | Sequence[int] | None" = None) -> list[str]:
        """log_probs [T, B, V] -> list of B strings."""
        lp = log_probs.detach().float().cpu().numpy()
        out = []
        for b in range(lp.shape[1]):
            n = lp.shape[0] if lengths is None else int(lengths[b])
            out.append(self.decode_one(lp[:n, b]))
        return out

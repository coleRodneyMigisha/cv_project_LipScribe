"""Flix v4: a larger word 4-gram LM, from LibriSpeech book text plus our own training transcripts.

Same estimate as scripts/flix_v4_lm.py (interpolated modified Kneser-Ney), written with numpy so it
scales to tens of millions of words:

* Words map to 16-bit ids. The vocabulary is the most frequent LibriSpeech words plus every word
  of our training transcripts, at most 65,532 words. Anything else becomes ``<unk>``. Each n-gram
  is packed into one uint64 (16 bits per word, left-padded), so counting is a sort
  (``np.unique``).
* **Text.** A regular sample of the LibriSpeech LM corpus (OpenSLR 11: public-domain books,
  normalised; no overlap with our TED talks or films), plus our training-partition transcripts
  repeated ``--in-domain-repeat`` times so the film / TED phrasing carries weight. No
  validation or test text is used.
* **Pruning.** 3- and 4-grams with a raw count of 1 are dropped (as ``lmplz --prune 0 0 1 1``).
  The prefixes and suffixes of every kept n-gram are always kept. The back-off weight of a context
  is ``1 - sum of the kept discounted mass``, so the pruned model still sums to one.

  .venv/bin/python -m scripts.flix_v4_lm_big --libri lm/librispeech/librispeech-lm-norm.txt.gz \
      --libri-words 40000000 --out lm/flix_libri_4gram.arpa
"""
from __future__ import annotations

import argparse
import gzip
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.flix_v4_lm import training_sentences

ORDER = 4
BITS = 16
MASK16 = np.uint64(0xFFFF)
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["", "<s>", "</s>", "<unk>"]


def read_libri(path: str, n_words: int, total_words: float = 8.0e8) -> list[list[str]]:
    """Every k-th line of the corpus, so the sample spans all books."""
    step = max(1, int(total_words // n_words))
    out, got = [], 0
    with gzip.open(path, "rt", encoding="utf8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i % step:
                continue
            w = line.lower().split()
            if w:
                out.append(w)
                got += len(w)
                if got >= n_words:
                    break
    return out


def pack(cols: list[np.ndarray]) -> np.ndarray:
    key = np.zeros(len(cols[0]), dtype=np.uint64)
    for c in cols:
        key = (key << np.uint64(BITS)) | c.astype(np.uint64)
    return key


def ngram_counts(ids: np.ndarray, sent: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unique packed n-grams and raw counts; n-grams never cross sentences; (<s>) alone skipped."""
    L = len(ids) - n + 1
    ok = sent[:L] == sent[n - 1:n - 1 + L]
    cols = [ids[k:k + L][ok] for k in range(n)]
    keys = pack(cols)
    if n == 1:
        keys = keys[keys != BOS]
    return np.unique(keys, return_counts=True)


def prefix(keys: np.ndarray) -> np.ndarray:          # drop the last word
    return keys >> np.uint64(BITS)


def suffix(keys: np.ndarray, n: int) -> np.ndarray:  # drop the first word of an n-gram
    if n == 1:
        return np.zeros_like(keys)
    return keys & np.uint64((1 << (BITS * (n - 1))) - 1)


def first_word(keys: np.ndarray, n: int) -> np.ndarray:
    return (keys >> np.uint64(BITS * (n - 1))) & MASK16


def lookup(table_keys: np.ndarray, table_vals: np.ndarray, q: np.ndarray, default=0.0) -> np.ndarray:
    pos = np.searchsorted(table_keys, q)
    pos = np.minimum(pos, len(table_keys) - 1)
    hit = table_keys[pos] == q
    out = np.full(len(q), default, dtype=table_vals.dtype)
    out[hit] = table_vals[pos[hit]]
    return out


def discounts(adj: np.ndarray) -> np.ndarray:
    t = np.bincount(np.minimum(adj, 5).astype(np.int64), minlength=6)
    t1, t2, t3, t4 = t[1], t[2], t[3], t[4]
    if min(t1, t2, t3, t4) == 0:
        return np.array([0.5, 1.0, 1.5])
    y = t1 / (t1 + 2 * t2)
    d = np.array([1 - 2 * y * t2 / t1, 2 - 3 * y * t3 / t2, 3 - 4 * y * t4 / t3])
    return np.clip(d, 0.05, np.array([0.95, 1.95, 2.95]))


def build(ids: np.ndarray, sent: np.ndarray, vocab_size: int, prune_min: dict[int, int]):
    keys, raw = {}, {}
    for n in range(1, ORDER + 1):
        keys[n], raw[n] = ngram_counts(ids, sent, n)
        print(f"order {n}: {len(keys[n])} distinct n-grams", flush=True)
    # adjusted (continuation) counts
    adj = {ORDER: raw[ORDER].astype(np.int64)}
    for n in range(ORDER - 1, 0, -1):
        sk, sc = np.unique(suffix(keys[n + 1], n + 1), return_counts=True)
        cont = lookup(sk, sc.astype(np.int64), keys[n], 0)
        a = np.where(first_word(keys[n], n) == BOS, raw[n], cont).astype(np.int64)
        adj[n] = a
    # keep set: prune raw-count-1 high orders, then force prefixes/suffixes of kept entries
    keep = {n: (raw[n] >= prune_min.get(n, 1)) & (adj[n] > 0) for n in range(1, ORDER + 1)}
    for n in range(ORDER, 1, -1):
        k = keys[n][keep[n]]
        need = np.unique(np.concatenate([prefix(k), suffix(k, n)]))
        keep[n - 1] |= np.isin(keys[n - 1], need, assume_unique=False)
    # probabilities bottom-up over kept entries
    probs, bows = {}, {}
    for n in range(1, ORDER + 1):
        D = discounts(adj[n])
        kk, aa = keys[n], adj[n]
        ctx = prefix(kk)                                             # context key (order n-1)
        # totals over ALL observed continuations of each context (unpruned counts)
        cu, inv = np.unique(ctx, return_inverse=True)
        tot = np.bincount(inv, weights=aa.astype(np.float64))
        dsc = D[np.minimum(aa, 3) - 1]
        u = (aa - dsc) / tot[inv]
        kept = keep[n]
        kept_u = np.bincount(inv[kept], weights=u[kept], minlength=len(cu))
        gamma = np.clip(1.0 - kept_u, 1e-12, 1.0)                     # back-off mass of each context
        if n == 1:
            lower = np.full(kept.sum(), 1.0 / vocab_size)
        else:
            pk, pv = probs[n - 1]
            lower = lookup(pk, pv, suffix(kk[kept], n), np.nan)
            if np.isnan(lower).any():
                raise RuntimeError(f"order {n}: suffix missing for {np.isnan(lower).sum()} n-grams")
        p = u[kept] + gamma[inv[kept]] * lower
        order_idx = np.argsort(kk[kept])
        probs[n] = (kk[kept][order_idx], p[order_idx])
        bows[n - 1] = (cu, gamma)                                    # contexts are (n-1)-grams
        print(f"order {n}: kept {kept.sum()} n-grams, discounts {np.round(D, 3).tolist()}", flush=True)
    unk_p = bows[0][1][0] / vocab_size                               # <unk> gets the uniform share
    return probs, bows, unk_p


def write_arpa(path: Path, probs, bows, unk_p: float, words: list[str]) -> None:
    def text(keys: np.ndarray, n: int) -> list[str]:
        cols = [((keys >> np.uint64(BITS * (n - 1 - k))) & MASK16).astype(np.int64) for k in range(n)]
        return [" ".join(words[c[i]] for c in cols) for i in range(len(keys))]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\\data\\\n")
        uni_has_unk = np.any(probs[1][0] == UNK)
        sizes = [len(probs[1][0]) + 1 + (0 if uni_has_unk else 1)] + [len(probs[n][0]) for n in range(2, ORDER + 1)]
        for n, k in enumerate(sizes, 1):
            f.write(f"ngram {n}={k}\n")
        for n in range(1, ORDER + 1):
            f.write(f"\n\\{n}-grams:\n")
            k, p = probs[n]
            lp = np.log10(p)
            bk, bv = bows[n] if n < ORDER else (np.array([], dtype=np.uint64), np.array([]))
            bo = lookup(bk, np.log10(bv), k, np.nan) if n < ORDER else np.full(len(k), np.nan)
            if n == 1:
                b_bos = lookup(bk, np.log10(bv), np.array([BOS], dtype=np.uint64), 0.0)[0]
                f.write(f"-99.000000\t<s>\t{b_bos:.6f}\n")
                if not uni_has_unk:
                    f.write(f"{math.log10(unk_p):.6f}\t<unk>\n")
            for s, l, b in zip(text(k, n), lp, bo):
                f.write(f"{l:.6f}\t{s}" + ("" if np.isnan(b) else f"\t{b:.6f}") + "\n")
        f.write("\n\\end\\\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", required=True)
    ap.add_argument("--libri-words", type=int, default=40_000_000)
    ap.add_argument("--manifest", default="data/flix_v4c/train_mix.csv")
    ap.add_argument("--splits", default="data/flix_v4c/splits.csv")
    ap.add_argument("--in-domain-repeat", type=int, default=5)
    ap.add_argument("--vocab", type=int, default=65532)
    ap.add_argument("--out", default="lm/flix_libri_4gram.arpa")
    a = ap.parse_args()

    ours = training_sentences(a.manifest, a.splits)
    libri = read_libri(a.libri, a.libri_words)
    print(f"LibriSpeech sample: {len(libri)} lines, {sum(map(len, libri))} words; "
          f"ours: {len(ours)} sentences x {a.in_domain_repeat}", flush=True)
    freq = Counter(w for s in libri for w in s)
    own = {w for s in ours for w in s}
    room = a.vocab - len(own)
    words = SPECIAL + sorted(own) + [w for w, _ in freq.most_common() if w not in own][:room]
    index = {w: i for i, w in enumerate(words)}
    if len(words) > 1 << BITS:                  # ids 0 .. 65535
        raise ValueError("vocabulary does not fit 16 bits")
    print(f"vocabulary {len(words) - len(SPECIAL)} words", flush=True)

    seqs = libri + ours * a.in_domain_repeat
    lens = np.array([len(s) + 2 for s in seqs], dtype=np.int64)
    ids = np.empty(lens.sum(), dtype=np.uint16)
    pos = 0
    for s in seqs:
        ids[pos] = BOS
        ids[pos + 1:pos + 1 + len(s)] = [index.get(w, UNK) for w in s]
        ids[pos + 1 + len(s)] = EOS
        pos += len(s) + 2
    sent = np.repeat(np.arange(len(seqs), dtype=np.int64), lens)
    del seqs, libri
    probs, bows, unk_p = build(ids, sent, len(words) - 1, prune_min={3: 2, 4: 2})
    write_arpa(Path(a.out), probs, bows, unk_p, words)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

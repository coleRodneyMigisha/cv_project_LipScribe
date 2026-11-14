"""Flix v4: build a word n-gram language model (ARPA) from our own training transcripts.

Interpolated modified Kneser-Ney, the same estimate as KenLM's ``lmplz``:

* The highest order uses raw counts. Each lower order uses continuation counts (how many
  distinct words precede the n-gram), except n-grams that start with ``<s>``, which keep raw counts.
* Three discounts per order (D1, D2, D3+) come from the count-of-counts (Chen & Goodman, 1998).
* ``p(w|h) = (a(hw) - D) / a(h.) + gamma(h) * p(w|h')``. Unigrams interpolate with a uniform
  distribution over the vocabulary plus ``<unk>``. ARPA stores the interpolated probabilities,
  with ``gamma`` as the back-off weight.

Text: transcripts of the training partition only (train_mix rows whose speaker/talk is "train"
in the splits file), so no validation or test sentence reaches the LM. Lower-case, as our
references are.

  .venv/bin/python -m scripts.flix_v4_lm --order 4 --out lm/flix_train_4gram.arpa
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def training_sentences(manifest: str, splits: str) -> list[list[str]]:
    part = {}
    with open(splits, newline="") as f:
        for r in csv.DictReader(f):
            part[r["speaker"]] = r["partition"]
    out = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("partition", "train") != "train" or part.get(r["speaker"], "train") != "train":
                continue
            words = r["transcript"].lower().split()
            if words:
                out.append(words)
    return out


def discounts(counts: dict) -> tuple[float, float, float]:
    t = Counter(c for c in counts.values() if c <= 4)      # exact count-of-counts n_1..n_4
    t1, t2, t3, t4 = (t.get(k, 0) for k in (1, 2, 3, 4))
    if min(t1, t2, t3, t4) == 0:
        return 0.5, 1.0, 1.5
    y = t1 / (t1 + 2 * t2)
    d = (1 - 2 * y * t2 / t1, 2 - 3 * y * t3 / t2, 3 - 4 * y * t4 / t3)
    return tuple(min(max(x, 0.05), k + 1 - 0.05) for k, x in enumerate(d))  # keep 0 < D_k < k+1


def build(sentences: list[list[str]], order: int):
    raw = [Counter() for _ in range(order + 1)]          # raw[n][ngram tuple]
    for s in sentences:
        toks = [BOS] + s + [EOS]
        for n in range(1, order + 1):
            for i in range(len(toks) - n + 1):
                g = tuple(toks[i:i + n])
                if g == (BOS,):
                    continue
                raw[n][g] += 1
    # adjusted counts
    adj = [None] * (order + 1)
    adj[order] = dict(raw[order])
    for n in range(order - 1, 0, -1):
        cont = Counter(g[1:] for g in raw[n + 1])        # distinct left extensions
        a = {}
        for g, c in raw[n].items():
            a[g] = c if g[0] == BOS else cont.get(g, 0)
        adj[n] = {g: c for g, c in a.items() if c > 0}
    vocab = sorted({g[0] for g in adj[1]} | {UNK})

    probs = [None] * (order + 1)
    bows = [None] * (order + 1)
    for n in range(1, order + 1):
        D = discounts(adj[n])
        tot = defaultdict(float)
        nk = defaultdict(lambda: [0, 0, 0])
        for g, c in adj[n].items():
            h = g[:-1]
            tot[h] += c
            nk[h][min(c, 3) - 1] += 1
        gamma = {h: (D[0] * k[0] + D[1] * k[1] + D[2] * k[2]) / tot[h] for h, k in nk.items()}
        p = {}
        for g, c in adj[n].items():
            h = g[:-1]
            u = (c - D[min(c, 3) - 1]) / tot[h]
            lower = (1.0 / len(vocab)) if n == 1 else probs[n - 1][g[1:]]
            p[g] = u + gamma[h] * lower
        if n == 1:
            p[(UNK,)] = gamma[()] / len(vocab)
        probs[n] = p
        bows[n - 1] = gamma                               # back-off of the (n-1)-gram contexts
        print(f"order {n}: {len(p)} n-grams, discounts {tuple(round(x, 3) for x in D)}", flush=True)
    return probs, bows, vocab


def write_arpa(path: Path, probs, bows, order: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\\data\\\n")
        sizes = [len(probs[1]) + 1] + [len(probs[n]) for n in range(2, order + 1)]   # +1 for <s>
        for n, k in enumerate(sizes, 1):
            f.write(f"ngram {n}={k}\n")
        for n in range(1, order + 1):
            f.write(f"\n\\{n}-grams:\n")
            entries = dict(probs[n])
            if n == 1:
                entries[(BOS,)] = None
            for g in sorted(entries):
                lp = -99.0 if entries[g] is None else math.log10(entries[g])
                line = f"{lp:.6f}\t{' '.join(g)}"
                if n < order and g in bows[n]:
                    line += f"\t{math.log10(bows[n][g]):.6f}"
                f.write(line + "\n")
        f.write("\n\\end\\\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/flix_v4c/train_mix.csv")
    ap.add_argument("--splits", default="data/flix_v4c/splits.csv")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--out", default="lm/flix_train_4gram.arpa")
    a = ap.parse_args()
    sents = training_sentences(a.manifest, a.splits)
    print(f"{len(sents)} training sentences, {sum(map(len, sents))} words", flush=True)
    probs, bows, vocab = build(sents, a.order)
    write_arpa(Path(a.out), probs, bows, a.order)
    print(f"wrote {a.out} (vocabulary {len(vocab)})")


if __name__ == "__main__":
    main()

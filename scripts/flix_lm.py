"""X6 'Flix': build the word-bigram LM + lexicon for lexicon_lm decoding (run on the laptop).

Text = every subtitle line (any length) of the TRAINING titles only, cleaned exactly like
the clip transcripts (scripts/mine_flix.clean_text). Val titles (incl. --extra-val) and
test titles are excluded, so their words are only known if they also occur in training.

Usage: python -m scripts.flix_lm --out flix_lm.json [--extra-val i_robot skyfall]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import importlib.util
import sys

import scripts.mine_flix as mf

# load lexicon_lm.py directly: lipscribe.decoding/__init__ imports torch, absent on the laptop
_spec = importlib.util.spec_from_file_location(
    "lexicon_lm", Path(__file__).resolve().parents[1] / "lipscribe/decoding/lexicon_lm.py")
_mod = importlib.util.module_from_spec(_spec); sys.modules["lexicon_lm"] = _mod; _spec.loader.exec_module(_mod)
WordBigramLM = _mod.WordBigramLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subs-cache", type=Path, default=Path.home() / "flix_mine" / "subs")
    ap.add_argument("--extra-val", nargs="*", default=["i_robot", "skyfall"])
    ap.add_argument("--videos", type=Path, default=mf.V)
    a = ap.parse_args()
    mf.V = a.videos
    held = mf.VAL | mf.TEST | set(a.extra_val)
    texts, used = [], Counter()
    for key, video, sub in mf._titles():
        g = mf.group_of(key)
        if g in held or not Path(video).exists():
            continue
        for _, _, lines in mf.load_cues(Path(video), sub, a.subs_cache):
            t = mf.clean_text(lines)
            if t:
                texts.append(t); used[g] += 1
    lm = WordBigramLM.from_texts(texts)
    lm.save(a.out)
    words = Counter(w for t in texts for w in t.split())
    stats = {"lines": len(texts), "word_tokens": sum(words.values()), "vocab": len(lm.vocab),
             "titles": dict(used), "held_out": sorted(held)}
    a.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

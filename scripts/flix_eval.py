"""X6 'Flix': score a checkpoint on any manifest (Flix test or the GRID test subset).

Reuses scripts.evaluate.score_checkpoint but swaps the checkpoint's data section for
the given manifest/splits/SNRs, so the stored model config stays untouched. Greedy
decoding only: the grammar decoder forces GRID's six-word pattern and would hide
every film word.

Writes <out>/<tag>.json (WER/CER per SNR) and <out>/<tag>_hyps.csv.

Usage: python -m scripts.flix_eval --checkpoint checkpoints/flix_base_p4.pt \
           --manifest data/flix/flix_test.csv --tag base_flix [--drop-modality audio]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

import scripts.evaluate as ev
from scripts.train import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="data/flix/splits.csv")
    ap.add_argument("--snrs", nargs="+", default=["clean", "0"])
    ap.add_argument("--drop-modality", choices=["audio", "visual"], default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/flix"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    ckpt = {**ckpt, "config": copy.deepcopy(ckpt["config"])}
    d = ckpt["config"]["data"]
    d.update(manifest=args.manifest, splits=args.splits, eval_snrs=list(args.snrs))
    d.pop("noisy_dir", None)          # no fixed noisy files: mix deterministically from the test pool
    ckpt["config"]["decode"] = {"method": "greedy"}
    ev.load_checkpoint = lambda *a, **k: ckpt

    results, _, hyps = ev.score_checkpoint(args.checkpoint, torch.device(args.device),
                                           drop_modality=args.drop_modality, decode="greedy")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "manifest": args.manifest,
         "drop_modality": args.drop_modality, "results": results}, indent=2))
    ev.save_hypotheses(args.out / f"{args.tag}_hyps.csv", hyps)
    print(f"[{args.tag}] " + ", ".join(f"{s}: WER {r['wer']:.3f} CER {r['cer']:.3f}"
                                       for s, r in results.items()), flush=True)


if __name__ == "__main__":
    main()

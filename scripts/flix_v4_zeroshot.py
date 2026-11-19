"""Flix v4 diagnostic: score the *unmodified* Auto-AVSR lip reader on our mouth crops.

This is not a result we report as our model. It checks that our 96x96 crops (GRID crop
pipeline, MediaPipe face mesh) are framed closely enough to Auto-AVSR's own crops for its
weights to be a useful starting point, and gives the "before fine-tuning" baseline for the
learning-curve log. Greedy CTC with Auto-AVSR's own 5,000-piece subword CTC layer
(no decoder, no beam search), so it is a lower bound on what the original model can do.

  .venv/bin/python -m scripts.flix_v4_zeroshot --manifests data/flix_v3/film_val.csv data/flix_v3/ted_val.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lipscribe.data.dataset import GridDataset, collate
from lipscribe.evaluation import cer, wer
from lipscribe.models.pretrained import AutoAVSRVisualEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--splits", default="data/flix_v3/splits.csv")
    ap.add_argument("--ckpt", default="pretrained/vsr_trlrs3vox2_base.pth")
    ap.add_argument("--spm", default="pretrained/unigram5000.model")
    ap.add_argument("--units", default="pretrained/unigram5000_units.txt")
    ap.add_argument("--out", default="reports/flix_v4/zeroshot_autoavsr.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--zooms", type=float, nargs="+", default=[1.0])
    ap.add_argument("--limit", type=int, default=0, help="first N clips per manifest (quick scans)")
    a = ap.parse_args()

    import sentencepiece  # noqa: F401  (only to fail early if missing)
    units = [ln.split()[0] for ln in Path(a.units).read_text(encoding="utf8").splitlines()]
    token_list = ["<blank>"] + units + ["<eos>"]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = AutoAVSRVisualEncoder(checkpoint=a.ckpt).to(dev).eval()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    ctc = torch.nn.Linear(768, sd["ctc.ctc_lo.weight"].shape[0]).to(dev)
    ctc.load_state_dict({"weight": sd["ctc.ctc_lo.weight"], "bias": sd["ctc.ctc_lo.bias"]})

    def decode(ids: list[int]) -> str:
        out, prev = [], None
        for i in ids:
            if i != prev and i != 0 and i < len(token_list) - 1:
                out.append(token_list[i])
            prev = i
        return "".join(out).replace("▁", " ").strip().lower()

    results = {}
    for zoom, m in [(z, m) for z in a.zooms for m in a.manifests]:
        enc.zoom = zoom
        ds = GridDataset(m, a.splits, "val", visual_augment=False)
        if a.limit:
            ds = torch.utils.data.Subset(ds, range(min(a.limit, len(ds))))
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, collate_fn=collate, num_workers=4)
        refs, hyps = [], []
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            for b in dl:
                v, L = b["video"].to(dev), b["lengths"].to(dev).clamp(max=b["video"].size(2))
                best = ctc(enc(v, L)).argmax(-1).cpu()
                hyps += [decode(best[i, : int(L[i])].tolist()) for i in range(len(L))]
                refs += b["texts"]
        name = f"{Path(m).stem}@zoom{zoom:g}"
        results[name] = {"n": len(refs), "wer": wer(refs, hyps), "cer": cer(refs, hyps)}   # no per-clip text in reports
        print(f"{name}: n={len(refs)}  WER {results[name]['wer']:.3f}  CER {results[name]['cer']:.3f}")
        for r, h in list(zip(refs, hyps))[:2]:
            print(f"   ref: {r}\n   hyp: {h}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

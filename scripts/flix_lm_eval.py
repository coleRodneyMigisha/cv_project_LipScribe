"""X6 'Flix': decode with the lexicon + word-bigram LM (the film counterpart of GRID's grammar).

For every checkpoint: run the model once over Flix val and test, keep the CTC
log-probabilities, tune the LM weight (alpha) and word bonus (beta) on VAL (mean WER over
clean and 0 dB), then decode the test titles once with the chosen pair. Greedy results from
the same log-probabilities are written too, for a like-for-like comparison.

Writes <out>/<label>_{greedy,lm}_flix[_lips_only].json + _hyps.csv (flix_eval format),
<out>/tuning.csv and <out>/lm_summary.json.

Usage: python -m scripts.flix_lm_eval --lm data/flix_lm.json --ckpt base=checkpoints/flix_base_p4.pt \
         --ckpt sweep_s1=runs/flix_sweep/<run>/best.pt
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lipscribe.data.dataset import GridDataset, collate
from lipscribe.decoding import greedy_decode
from lipscribe.decoding.lexicon_lm import LexiconLMDecoder, WordBigramLM
from lipscribe.evaluation import cer, wer
from lipscribe.models import build_model
from scripts.train import build_feature_step, load_checkpoint, snr_value, to_device


def logits(ckpt_path: str, manifest: str, splits: str, part: str, snr: str, drop: str | None, dev):
    ck = load_checkpoint(ckpt_path, map_location="cpu"); cfg = ck["config"]
    model = build_model(cfg["model"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
    prep = build_feature_step(cfg, dev)
    ds = GridDataset(manifest, splits, part, snr_db=snr_value(snr), noise_splits=cfg["data"].get("noise_splits"))
    out = []
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=4):
            batch = prep(to_device(batch, dev))
            if drop:
                batch["drop_modality"] = drop
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                o = model(batch)
            lp = o["log_probs"].float().cpu(); ln = o["lengths"].cpu()
            for b, (u, ref) in enumerate(zip(batch["utt_ids"], batch["texts"])):
                out.append((u, ref, lp[: int(ln[b]), b].numpy()))
    return out


def decode_all(items, dec) -> list[tuple[str, str, str]]:
    return [(u, ref, dec(lp)) for u, ref, lp in items]


def greedy(lp) -> str:
    return greedy_decode(torch.from_numpy(lp).unsqueeze(1), torch.tensor([lp.shape[0]]))[0]


def save(out: Path, tag: str, rows_by_snr: dict, ck: str, extra: dict) -> None:
    res, hyps = {}, []
    for snr, rows in rows_by_snr.items():
        refs = [r for _, r, _ in rows]; hs = [h for _, _, h in rows]
        res[snr] = {"n_utts": len(rows), "wer": wer(refs, hs), "cer": cer(refs, hs)}
        hyps += [{"utt_id": u, "snr": snr, "speaker": u.split("_", 1)[0], "ref": r, "hyp": h,
                  "wer": f"{wer([r], [h]):.4f}", "cer": f"{cer([r], [h]):.4f}", "gate": ""} for u, r, h in rows]
    (out / f"{tag}.json").write_text(json.dumps({"checkpoint": ck, **extra, "results": res}, indent=2))
    with open(out / f"{tag}_hyps.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["utt_id", "snr", "speaker", "ref", "hyp", "wer", "cer", "gate"])
        w.writeheader(); w.writerows(hyps)
    print(f"[{tag}] " + ", ".join(f"{s}: WER {v['wer']:.3f} CER {v['cer']:.3f}" for s, v in res.items()), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="label=path")
    ap.add_argument("--data", type=Path, default=Path("data/flix_only"))
    ap.add_argument("--out", type=Path, default=Path("reports/flix_lm"))
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.5, 2.0])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--beam", type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = WordBigramLM.load(a.lm)
    sp = str(a.data / "splits.csv")
    summary, tune_rows = {}, []
    for spec in a.ckpt:
        label, ck = spec.split("=", 1)
        val = {s: logits(ck, str(a.data / "train_mix.csv"), sp, "val", s, None, dev) for s in ("clean", "0")}
        test = {s: logits(ck, str(a.data / "flix_test.csv"), sp, "test", s, None, dev) for s in ("clean", "0")}
        lips = {"clean": logits(ck, str(a.data / "flix_test.csv"), sp, "test", "clean", "audio", dev)}
        best = None
        for al, be in itertools.product(a.alphas, a.betas):
            d = LexiconLMDecoder(lm, alpha=al, beta=be, beam_width=a.beam)
            scores = {s: decode_all(val[s], d.decode_one) for s in val}
            vw = sum(wer([r for _, r, _ in v], [h for _, _, h in v]) for v in scores.values()) / len(scores)
            vc = sum(cer([r for _, r, _ in v], [h for _, _, h in v]) for v in scores.values()) / len(scores)
            tune_rows.append({"model": label, "alpha": al, "beta": be, "val_wer": f"{vw:.4f}", "val_cer": f"{vc:.4f}"})
            if best is None or vw < best[0] or (vw == best[0] and vc < best[1]):
                best = (vw, vc, al, be)
        vw, vc, al, be = best
        print(f"{label}: best alpha {al} beta {be} (val WER {vw:.3f}, CER {vc:.3f})", flush=True)
        d = LexiconLMDecoder(lm, alpha=al, beta=be, beam_width=a.beam)
        extra = {"alpha": al, "beta": be, "val_wer": vw, "val_cer": vc}
        save(a.out, f"{label}_lm_flix", {s: decode_all(test[s], d.decode_one) for s in test}, ck, extra)
        save(a.out, f"{label}_lm_flix_lips_only", {s: decode_all(lips[s], d.decode_one) for s in lips}, ck, extra)
        save(a.out, f"{label}_greedy_flix", {s: decode_all(test[s], greedy) for s in test}, ck, {})
        save(a.out, f"{label}_greedy_flix_lips_only", {s: decode_all(lips[s], greedy) for s in lips}, ck, {})
        summary[label] = {"checkpoint": ck, "alpha": al, "beta": be, "val_wer": vw, "val_cer": vc}
    with open(a.out / "tuning.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tune_rows[0])); w.writeheader(); w.writerows(tune_rows)
    refs = [r for _, r, _ in test["clean"]]
    toks = [w for r in refs for w in r.split()]
    summary["test_words_in_lexicon"] = {"in": sum(w in set(lm.vocab) for w in toks), "of": len(toks)}
    (a.out / "lm_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["test_words_in_lexicon"]), flush=True)


if __name__ == "__main__":
    main()

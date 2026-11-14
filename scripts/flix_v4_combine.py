"""Flix v4: combine two word-piece models frame by frame (no training).

fusion_wp and its audio-only twin (fusion_wp_audio) emit the same 5,049 word pieces on the same
25 fps frames, so their per-frame posteriors can be mixed:

  log-linear  : w * log p_av + (1 - w) * log p_audio
  linear      : log(w * p_av + (1 - w) * p_audio)

The weight (and rule) is chosen on the validation sets only (film_val + ted_val, clean / 0 / -5 /
-10 dB, mean WER with each set weighted equally), then the test sets are scored once with it.
The best weight per SNR on validation is reported too, as the ceiling of an SNR-aware mixer.
Greedy CTC decoding.

  PYTHONPATH=. .venv/bin/python scripts/flix_v4_combine.py --av RUN --audio RUN --out reports/flix_v4/combine.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lipscribe.data.dataset import GridDataset, collate
from lipscribe.decoding import greedy_decode
from lipscribe.evaluation import wer
from lipscribe.models import build_model

SNRS = {"clean": float("inf"), "0": 0.0, "-5": -5.0, "-10": -10.0}
VAL = {"film_val": "data/flix_v4c/film_val.csv", "ted_val": "data/flix_v4c/ted_val.csv"}
TEST = {"film_test": "data/flix_v4c/flix_test.csv", "ted_test": "data/flix_v4c/ted_test.csv"}
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]
RULES = ("loglinear", "linear")


def load(run: str, dev):
    ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    cfg = dict(ck["config"]["model"])
    cfg.pop("init_heads_from", None); cfg.pop("init_model_from", None)
    m = build_model(cfg)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval()


def mix(la: torch.Tensor, lb: torch.Tensor, w: float, rule: str) -> torch.Tensor:
    if rule == "loglinear":
        return w * la + (1 - w) * lb
    if w in (0.0, 1.0):
        return la if w == 1.0 else lb
    return torch.logaddexp(la + torch.log(torch.tensor(w)), lb + torch.log(torch.tensor(1 - w)))


def score_set(av, au, manifest, split, splits, snr, dev, combos):
    ds = GridDataset(manifest, splits, split, snr_db=snr, noise_splits="Data/Noise_splits/splits.csv",
                     visual_augment=False)
    dl = DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=6)
    refs, hyps = [], {c: [] for c in combos}
    with torch.no_grad(), torch.autocast("cuda"):
        for b in dl:
            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
            oa, ob = av(b), au(b)
            assert torch.equal(oa["lengths"], ob["lengths"]), "the two models disagree on frame counts"
            la, lb = oa["log_probs"].float(), ob["log_probs"].float()
            for c in combos:
                hyps[c] += greedy_decode(mix(la, lb, c[1], c[0]), oa["lengths"])
            refs += b["texts"]
    return {f"{r}@{w}": round(100 * wer(refs, hyps[(r, w)]), 2) for r, w in combos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--av", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="reports/flix_v4/combine.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    av, au = load(a.av, dev), load(a.audio, dev)
    res = {"val": {}, "test": {}}

    combos = [(r, w) for r in RULES for w in WEIGHTS]
    for name, man in VAL.items():
        for s, snr in SNRS.items():
            res["val"][f"{name}|{s}"] = score_set(av, au, man, "val", "data/flix_v4c/splits.csv", snr, dev, combos)
            print("val", name, s, min(res["val"][f"{name}|{s}"].items(), key=lambda kv: kv[1]), flush=True)
    keys = list(res["val"])
    mean = {c: sum(res["val"][k][c] for k in keys) / len(keys) for c in res["val"][keys[0]]}
    best = min(mean, key=mean.get)
    res["val_mean"] = mean
    res["chosen"] = best
    res["best_per_snr_val"] = {s: min(mean_s := {c: sum(res["val"][f"{n}|{s}"][c] for n in VAL) / len(VAL)
                                                 for c in mean}, key=mean_s.get) for s in SNRS}
    print("chosen on val:", best, f"(mean val WER {mean[best]:.2f})", "| per SNR:", res["best_per_snr_val"], flush=True)

    rule, w = best.split("@")
    test_combos = sorted({(rule, float(w)), ("loglinear", 0.0), ("loglinear", 1.0)})
    for name, man in TEST.items():
        for s, snr in SNRS.items():
            res["test"][f"{name}|{s}"] = score_set(av, au, man, "test", "data/flix_v4/splits.csv", snr, dev, test_combos)
            print("test", name, s, res["test"][f"{name}|{s}"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))

    lines = [f"Frame-level combination of fusion_wp (lips + audio) and fusion_wp_audio, chosen on validation: "
             f"**{best}** (weight on the audio-visual model). Test WER %:", "",
             "| test | SNR | audio-only twin | fusion_wp | combined |", "|---|---|---:|---:|---:|"]
    for k, v in res["test"].items():
        t, s = k.split("|")
        lines.append(f"| {t} | {s} | {v['loglinear@0.0']} | {v['loglinear@1.0']} | **{v[best]}** |")
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

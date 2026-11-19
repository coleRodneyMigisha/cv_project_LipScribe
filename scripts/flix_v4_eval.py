"""Flix v4: score the stage checkpoints ONCE on the untouched test sets.

Film test (flix_test, 3 unseen films, human-checked references) and TED test (held-out talks,
Whisper references from data/flix_v4) at clean, 0 dB and -5 dB (MUSAN test pool, the same
fixed mixture for every model). The fused model is also scored with each stream removed
(``drop_modality``), as in the Model P ablations. Greedy CTC decoding.

  .venv/bin/python -m scripts.flix_v4_eval --run audio=RUN_DIR visual=RUN_DIR fusion=RUN_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lipscribe.data.dataset import GridDataset, collate
from lipscribe.decoding import greedy_decode
from lipscribe.evaluation import cer, wer
from lipscribe.models import build_model

TESTS = {"film_test": "data/flix_v4/flix_test.csv", "ted_test": "data/flix_v4/ted_test.csv"}
SNRS = {"clean": float("inf"), "0": 0.0, "-5": -5.0, "-10": -10.0}


def score(model, manifest, splits, snr, noise_splits, dev, drop=None, bs=16):
    ds = GridDataset(manifest, splits, "test", snr_db=snr, noise_splits=noise_splits, visual_augment=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=6)
    refs, hyps = [], []
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        for b in dl:
            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
            if drop:
                b["drop_modality"] = drop
            out = model(b)
            hyps += greedy_decode(out["log_probs"].float(), out["lengths"])
            refs += b["texts"]
    return {"n": len(refs), "wer": wer(refs, hyps), "cer": cer(refs, hyps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="name=run_dir")
    ap.add_argument("--splits", default="data/flix_v4/splits.csv")
    ap.add_argument("--noise-splits", default="Data/Noise_splits/splits.csv")
    ap.add_argument("--out", default="reports/flix_v4/test_results.json")
    ap.add_argument("--tests", nargs="*", help="name=manifest pairs instead of the default test sets")
    a = ap.parse_args()
    tests = dict(t.split("=", 1) for t in a.tests) if a.tests else TESTS
    dev = torch.device("cuda")
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for spec in a.run:
        name, run = spec.split("=", 1)
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
        mcfg = dict(ck["config"]["model"])
        model = build_model(mcfg)
        model.load_state_dict(ck["model"])
        model.to(dev).eval()
        modes = [None] if not name.startswith("fusion") or name.endswith("_audio") else [None, "visual", "audio"]   # drop nothing / lips / sound
        for test, man in tests.items():
            for drop in modes:
                snrs = ["clean"] if drop == "audio" or name == "visual" else list(SNRS)
                for s in snrs:
                    key = f"{name}{'' if drop is None else '-no_' + drop}|{test}|{s}"
                    if key in res:
                        continue
                    res[key] = score(model, man, a.splits, SNRS[s], a.noise_splits, dev, drop)
                    res[key]["epoch"] = int(ck["epoch"])
                    print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[key].items()},
                          flush=True)
                    Path(a.out).write_text(json.dumps(res, indent=2))
        del model; torch.cuda.empty_cache()

    lines = ["| model | test | SNR | WER % | CER % | n |", "|---|---|---|---:|---:|---:|"]
    for k, v in res.items():
        m, t, s = k.split("|")
        lines.append(f"| {m} | {t} | {s} | {100 * v['wer']:.1f} | {100 * v['cer']:.1f} | {v['n']} |")
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

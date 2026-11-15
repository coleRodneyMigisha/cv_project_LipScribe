"""Flix v6: score the models ONCE on the v5 test sets, with ordinary noise and with audio that does
not belong to the face on screen.

Conditions (GridDataset): clean, 0, -5, -10 dB MUSAN; silent (the speaker only mouths: zeros);
talkover (another test clip's speech at 0 dB over the speaker); replace (the speaker only mouths
while someone else talks: the other speech alone). The reference is always the on-screen
speaker's words. "lips_v5" is the v5 warm start with its audio stream switched off: the lip-only
level a model that ignores wrong audio should reach.

  PYTHONPATH=. .venv/bin/python scripts/flix_v6_eval.py --run twin_v5=RUN ws_v5=RUN ws_v6=RUN
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.flix_v5_eval import align_errors, group_of

SETS = {"film": "data/flix_v5/flix_test.csv", "ted": "data/flix_v5/ted_test.csv"}
CONDS = {"clean": (float("inf"), "none"), "0": (0.0, "none"), "-5": (-5.0, "none"), "-10": (-10.0, "none"),
         "silent": (float("inf"), "silent"), "talkover": (float("inf"), "talkover"),
         "replace": (float("inf"), "replace")}
OUT = Path("reports/flix_v6")


def decode(runs: dict[str, str]) -> dict:
    import torch
    from lipscribe.data.dataset import GridDataset, collate
    from lipscribe.decoding import greedy_decode
    from lipscribe.models import build_model
    dev = torch.device("cuda")
    cache = OUT / "hyps.json"
    res = json.loads(cache.read_text()) if cache.exists() else {}
    jobs = [(n, r, None) for n, r in runs.items()] + [("lips_v5", runs["ws_v5"], "audio")]
    for name, run, drop in jobs:
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
        cfg = dict(ck["config"]["model"])
        for k in ("init_heads_from", "init_model_from", "init_streams_from"):
            cfg.pop(k, None)
        model = build_model(cfg); model.load_state_dict(ck["model"]); model.to(dev).eval()
        for set_name, man in SETS.items():
            for c, (snr, cond) in CONDS.items():
                key = f"{name}|{set_name}|{c}"
                if key in res or (drop == "audio" and c != "clean"):
                    continue
                ds = GridDataset(man, str(Path(man).parent / "splits.csv"), "test", snr_db=snr, condition=cond,
                                 noise_splits="Data/Noise_splits/splits.csv", visual_augment=False)
                dl = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=6)
                hyps, refs = [], []
                with torch.no_grad(), torch.autocast("cuda"):
                    for b in dl:
                        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
                        if drop:
                            b["drop_modality"] = drop
                        o = model(b)
                        hyps += greedy_decode(o["log_probs"].float(), o["lengths"]); refs += b["texts"]
                res[key] = {"ids": [r["utt_id"] for r in ds.rows], "refs": refs, "hyps": hyps}
                cache.write_text(json.dumps(res))
                print("decoded", key, len(refs), flush=True)
        del model; torch.cuda.empty_cache()
    return res


def summarise(res: dict) -> str:
    rng = np.random.default_rng(0)
    per = {}
    for key, v in res.items():
        per[key] = [(u, sum(e) + ins, len(r.split()))
                    for u, r, h in zip(v["ids"], v["refs"], v["hyps"])
                    for e, ins in [align_errors(r.split(), h.split())]]
    names = ["twin_v5", "ws_v5", "ws_v6", "lips_v5"]
    lines = ["WER % (greedy). Reference = the on-screen speaker's words.", "",
             "| test set | condition | " + " | ".join(names) + " |", "|---|---|" + "---:|" * len(names)]
    for set_name in SETS:
        for c in CONDS:
            cells = []
            for n in names:
                rows = per.get(f"{n}|{set_name}|{c}")
                cells.append(f"{100 * sum(x[1] for x in rows) / sum(x[2] for x in rows):.1f}" if rows else "–")
            lines.append(f"| {set_name} | {c} | " + " | ".join(cells) + " |")
    lines += ["", "Paired bootstrap ws_v6 − ws_v5 (WER points, 95 % CI; whole talks resampled for TED):", "",
              "| test set | condition | diff | CI |", "|---|---|---:|---|"]
    for set_name in SETS:
        for c in CONDS:
            ra, rb = per.get(f"ws_v6|{set_name}|{c}"), per.get(f"ws_v5|{set_name}|{c}")
            if not ra or not rb:
                continue
            groups = sorted({group_of(x[0]) for x in ra})
            gi = np.array([groups.index(group_of(x[0])) for x in ra])
            A = np.bincount(gi, [x[1] for x in ra], len(groups)); B = np.bincount(gi, [x[1] for x in rb], len(groups))
            W = np.bincount(gi, [x[2] for x in ra], len(groups))
            idx = rng.integers(0, len(groups), (10000, len(groups)))
            d = 100 * (A[idx].sum(1) - B[idx].sum(1)) / W[idx].sum(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            lines.append(f"| {set_name} | {c} | {100 * (A.sum() - B.sum()) / W.sum():+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="name=run_dir (twin_v5, ws_v5, ws_v6)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = decode(dict(x.split("=", 1) for x in a.run))
    md = summarise(res)
    (OUT / "test_results.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()

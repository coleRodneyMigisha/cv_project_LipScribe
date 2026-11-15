"""Flix v5: score v4 and v5 models ONCE on the old (v4c) and the re-cut (v5) test sets.

Greedy CTC, clean / 0 / -5 / -10 dB (the fixed MUSAN test mixture of GridDataset). Per clip
hypotheses are kept, so the report also gives edge-word vs inner-word error (first / last
reference word vs the rest) and paired bootstrap intervals for v5 minus v4 (whole talks are
resampled for TED, clips for films).

  PYTHONPATH=. .venv/bin/python scripts/flix_v5_eval.py --run twin_v4=RUN ws_v4=RUN twin_v5=RUN ws_v5=RUN
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SETS = {"film_v4": "data/flix_v4c/flix_test.csv", "ted_v4": "data/flix_v4c/ted_test.csv",
        "film_v5": "data/flix_v5/flix_test.csv", "ted_v5": "data/flix_v5/ted_test.csv"}
SNRS = {"clean": float("inf"), "0": 0.0, "-5": -5.0, "-10": -10.0}
OUT = Path("reports/flix_v5")


def align_errors(r: list[str], h: list[str]) -> tuple[list[int], int]:
    """Errors charged to each reference word (substitution / deletion) and the insertion count."""
    n, m = len(r), len(h)
    D = np.zeros((n + 1, m + 1), int); D[:, 0] = range(n + 1); D[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    i, j, err, ins = n, m, [0] * n, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            err[i - 1] += r[i - 1] != h[j - 1]; i -= 1; j -= 1
        elif j > 0 and D[i, j] == D[i, j - 1] + 1:
            ins += 1; j -= 1
        else:
            err[i - 1] += 1; i -= 1
    return err, ins


def decode(runs: dict[str, str]) -> dict:
    import torch
    from lipscribe.data.dataset import GridDataset, collate
    from lipscribe.decoding import greedy_decode
    from lipscribe.models import build_model
    dev = torch.device("cuda")
    cache = OUT / "hyps.json"
    res = json.loads(cache.read_text()) if cache.exists() else {}
    for name, run in runs.items():
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
        cfg = dict(ck["config"]["model"])
        for k in ("init_heads_from", "init_model_from", "init_streams_from"):
            cfg.pop(k, None)
        model = build_model(cfg); model.load_state_dict(ck["model"]); model.to(dev).eval()
        for set_name, man in SETS.items():
            for s, snr in SNRS.items():
                key = f"{name}|{set_name}|{s}"
                if key in res:
                    continue
                ds = GridDataset(man, str(Path(man).parent / "splits.csv"), "test", snr_db=snr,
                                 noise_splits="Data/Noise_splits/splits.csv", visual_augment=False)
                dl = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=6)
                hyps, refs = [], []
                with torch.no_grad(), torch.autocast("cuda"):
                    for b in dl:
                        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
                        o = model(b)
                        hyps += greedy_decode(o["log_probs"].float(), o["lengths"]); refs += b["texts"]
                res[key] = {"ids": [r["utt_id"] for r in ds.rows], "refs": refs, "hyps": hyps}
                cache.write_text(json.dumps(res))
                print("decoded", key, len(refs), flush=True)
        del model; torch.cuda.empty_cache()
    return res


def group_of(uid: str) -> str:
    return re.sub(r"_t\d+$", "", uid) if "_ted_" in uid else uid


def summarise(res: dict, names: list[str]) -> str:
    rng = np.random.default_rng(0)
    per = {}                     # key -> (ids, err, words, edge_err, edge_words)
    for key, v in res.items():
        rows = []
        for uid, r, h in zip(v["ids"], v["refs"], v["hyps"]):
            r, h = r.split(), h.split()
            e, ins = align_errors(r, h)
            edge = {0, len(r) - 1} if r else set()
            rows.append((uid, sum(e) + ins, len(r), sum(e[p] for p in edge), len(edge), ins))
        per[key] = rows
    lines = ["| model | test set | SNR | WER % | edge-word err % | inner-word err % | n |", "|---|---|---|---:|---:|---:|---:|"]
    table = {}
    for name in names:
        for set_name in SETS:
            for s in SNRS:
                rows = per.get(f"{name}|{set_name}|{s}")
                if not rows:
                    continue
                E = sum(x[1] for x in rows); W = sum(x[2] for x in rows)
                Ee = sum(x[3] for x in rows); We = sum(x[4] for x in rows)
                Ei = E - Ee - sum(x[5] for x in rows)      # substitutions + deletions on inner words
                table[(name, set_name, s)] = 100 * E / W
                lines.append(f"| {name} | {set_name} | {s} | {100 * E / W:.2f} | {100 * Ee / We:.2f} | "
                             f"{100 * Ei / max(1, W - We):.2f} | {len(rows)} |")
    lines += ["", "Paired bootstrap, v5 minus v4 model on the same set (WER points, 95 % CI, P(v5 better)):", "",
              "| pair | test set | SNR | diff | CI | P |", "|---|---|---|---:|---|---:|"]
    for a, b in (("twin_v5", "twin_v4"), ("ws_v5", "ws_v4"), ("ws_v5", "twin_v5")):
        for set_name in SETS:
            for s in SNRS:
                ra, rb = per.get(f"{a}|{set_name}|{s}"), per.get(f"{b}|{set_name}|{s}")
                if not ra or not rb:
                    continue
                groups = sorted({group_of(x[0]) for x in ra})
                gi = np.array([groups.index(group_of(x[0])) for x in ra])
                A = np.bincount(gi, [x[1] for x in ra], len(groups)); B = np.bincount(gi, [x[1] for x in rb], len(groups))
                W = np.bincount(gi, [x[2] for x in ra], len(groups))
                idx = rng.integers(0, len(groups), (10000, len(groups)))
                d = 100 * (A[idx].sum(1) - B[idx].sum(1)) / W[idx].sum(1)
                pt = 100 * (A.sum() - B.sum()) / W.sum(); lo, hi = np.percentile(d, [2.5, 97.5])
                lines.append(f"| {a} − {b} | {set_name} | {s} | {pt:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {np.mean(d < 0):.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="name=run_dir")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = dict(x.split("=", 1) for x in a.run)
    res = decode(runs)
    md = summarise(res, list(runs))
    (OUT / "test_results.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()

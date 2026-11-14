"""Flix v4: CTC beam search with a word n-gram LM for the word-piece models (no retraining).

Two steps, so CUDA is never forked into the decoding workers:

  dump    (GPU)  run each model once on every set / SNR; keep the top-K word pieces per frame
                 (runs/flix_v4/beam_cache/<name>/<set>|<snr>.npz)
  decode  (CPU)  pyctcdecode beam search + KenLM (lm/flix_train_4gram.arpa, built from training
                 transcripts only by scripts/flix_v4_lm.py). For each model, the LM weight alpha
                 and word bonus beta are chosen on the validation sets only (film_val + ted_val,
                 every SNR, mean WER, each set weighted equally); the test sets are then decoded
                 once. Greedy and beam-without-LM results are reported alongside as controls.

  PYTHONPATH=. .venv/bin/python scripts/flix_v4_beam.py dump --run twin=RUN av=RUN ...
  PYTHONPATH=. .venv/bin/python scripts/flix_v4_beam.py decode --names twin av ... --out reports/flix_v4/beam
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

CACHE = Path("runs/flix_v4/beam_cache")
SNRS = {"clean": float("inf"), "0": 0.0, "-5": -5.0, "-10": -10.0}
VAL = {"film_val": "data/flix_v4c/film_val.csv", "ted_val": "data/flix_v4c/ted_val.csv"}
TEST = {"film_test": "data/flix_v4c/flix_test.csv", "ted_test": "data/flix_v4c/ted_test.csv"}
TOPK = 48
FLOOR = -30.0
ALPHAS = [0.0, 0.3, 0.5, 0.8, 1.2, 1.6]
BETAS = [0.0, 1.0, 2.0, 3.0]


# ----------------------------------------------------------------------------- dump (GPU)

def dump(runs: list[str]) -> None:
    import torch
    from torch.utils.data import DataLoader
    from lipscribe.data.dataset import GridDataset, collate
    from lipscribe.models import build_model

    dev = torch.device("cuda")
    for spec in runs:
        name, run = spec.split("=", 1)
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
        cfg = dict(ck["config"]["model"])
        for k in ("init_heads_from", "init_model_from", "init_streams_from"):
            cfg.pop(k, None)
        model = build_model(cfg)
        model.load_state_dict(ck["model"])
        model.to(dev).eval()
        visual_only = cfg["name"] == "v4_visual"
        out_dir = CACHE / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, sets in (("val", VAL), ("test", TEST)):
            splits = "data/flix_v4c/splits.csv" if split == "val" else "data/flix_v4/splits.csv"
            for set_name, man in sets.items():
                for s, snr in (({"clean": SNRS["clean"]}) if visual_only else SNRS).items():
                    path = out_dir / f"{set_name}|{s}.npz"
                    if path.exists():
                        continue
                    ds = GridDataset(man, splits, split, snr_db=snr, noise_splits="Data/Noise_splits/splits.csv",
                                     visual_augment=False)
                    dl = DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=6)
                    vals, idxs, lens, refs, greedy = [], [], [], [], []
                    from lipscribe.decoding import greedy_decode
                    with torch.no_grad(), torch.autocast("cuda"):
                        for b in dl:
                            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
                            o = model(b)
                            lp = o["log_probs"].float()                      # [T, B, V]
                            greedy += greedy_decode(lp, o["lengths"])
                            top = lp.topk(TOPK, dim=-1)
                            for i, n in enumerate(o["lengths"].tolist()):
                                vals.append(top.values[:n, i].half().cpu().numpy())
                                idxs.append(top.indices[:n, i].short().cpu().numpy())
                                lens.append(n)
                            refs += b["texts"]
                    np.savez(path, vals=np.concatenate(vals), idxs=np.concatenate(idxs), lens=np.array(lens),
                             refs=np.array(refs, dtype=object), greedy=np.array(greedy, dtype=object))
                    print(f"dumped {name} {set_name} {s} ({len(refs)} clips)", flush=True)
        del model
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- decode (CPU)

_DEC = None


@functools.lru_cache(maxsize=1)
def _labels() -> tuple[str, ...]:
    from lipscribe.data import subword
    tl = subword._tables()[1]
    return tuple([""] + [t.lower() for t in tl[1:-1]])   # blank = "", drop <eos> (last column)


def _init(lm: str) -> None:
    global _DEC
    from pyctcdecode import build_ctcdecoder
    _DEC = build_ctcdecoder(list(_labels()), kenlm_model_path=lm, alpha=0.5, beta=1.0)


def _dense(vals: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    m = np.full((vals.shape[0], len(_labels())), FLOOR, dtype=np.float32)
    keep = idxs < m.shape[1]                         # drop <eos>
    rows = np.repeat(np.arange(vals.shape[0])[:, None], vals.shape[1], 1)
    m[rows[keep], idxs[keep].astype(np.int64)] = vals[keep].astype(np.float32)
    return m


def _decode_job(args) -> list[str]:
    vals_list, idxs_list, alpha, beta, beam, use_lm = args
    _DEC.reset_params(alpha=alpha if use_lm else 0.0, beta=beta if use_lm else 0.0)
    out = []
    for v, i in zip(vals_list, idxs_list):
        text = _DEC.decode(_dense(v, i), beam_width=beam)
        out.append(" ".join(text.split()))
    return out


def load_set(name: str, key: str):
    z = np.load(CACHE / name / f"{key}.npz", allow_pickle=True)
    cuts = np.cumsum(z["lens"])[:-1]
    return (np.split(z["vals"], cuts), np.split(z["idxs"], cuts), list(z["refs"]), list(z["greedy"]))


def decode_all(pool, vals, idxs, alpha, beta, beam, use_lm, chunk=24) -> list[str]:
    jobs = [(vals[i:i + chunk], idxs[i:i + chunk], alpha, beta, beam, use_lm) for i in range(0, len(vals), chunk)]
    return [h for part in pool.map(_decode_job, jobs) for h in part]


def decode(names: list[str], lm: str, out: str, beam: int, workers: int) -> None:
    from lipscribe.evaluation import wer

    ctx = mp.get_context("fork")
    res = json.loads(Path(out + ".json").read_text()) if Path(out + ".json").exists() else {}
    _init(lm)                     # load the LM once; forked workers share it copy-on-write
    with ctx.Pool(workers) as pool:
        for name in names:
            if name in res and "test" in res[name]:
                continue
            keys = sorted(p.stem for p in (CACHE / name).glob("*.npz"))
            val_keys = [k for k in keys if k.split("|")[0] in VAL]
            test_keys = [k for k in keys if k.split("|")[0] in TEST]
            data = {k: load_set(name, k) for k in val_keys}
            grid = {}
            for a, b in itertools.product(ALPHAS, BETAS):
                if a == 0.0 and b != 0.0:
                    continue
                w = [100 * wer(data[k][2], decode_all(pool, data[k][0], data[k][1], a, b, beam, True)) for k in val_keys]
                grid[f"{a}|{b}"] = round(float(np.mean(w)), 2)
                print(f"{name} val alpha={a} beta={b}: mean WER {grid[f'{a}|{b}']:.2f}", flush=True)
            best = min(grid, key=grid.get)
            a, b = (float(x) for x in best.split("|"))
            r = {"val_grid": grid, "alpha": a, "beta": b, "test": {}}
            for k in test_keys:
                vals, idxs, refs, greedy = load_set(name, k)
                r["test"][k] = {
                    "greedy": round(100 * wer(refs, greedy), 2),
                    "beam_no_lm": round(100 * wer(refs, decode_all(pool, vals, idxs, 0.0, 0.0, beam, False)), 2),
                    "beam_lm": round(100 * wer(refs, decode_all(pool, vals, idxs, a, b, beam, True)), 2),
                    "n": len(refs)}
                print(name, k, r["test"][k], flush=True)
            res[name] = r
            Path(out + ".json").write_text(json.dumps(res, indent=1))

    lines = [f"Greedy vs beam search (width {beam}) without and with the 4-gram LM `{lm}`. "
             "alpha/beta chosen per model on validation. Test WER %:", "",
             "| model | alpha | beta | test | SNR | greedy | beam, no LM | beam + LM |",
             "|---|---:|---:|---|---|---:|---:|---:|"]
    for name, r in res.items():
        for k, v in r["test"].items():
            t, s = k.split("|")
            lines.append(f"| {name} | {r['alpha']} | {r['beta']} | {t} | {s} | {v['greedy']} | "
                         f"{v['beam_no_lm']} | **{v['beam_lm']}** |")
    Path(out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--run", nargs="+", required=True)
    e = sub.add_parser("decode")
    e.add_argument("--names", nargs="+", required=True)
    e.add_argument("--lm", default="lm/flix_train_4gram.arpa")
    e.add_argument("--out", default="reports/flix_v4/beam")
    e.add_argument("--beam", type=int, default=32)
    e.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.run)
    else:
        decode(a.names, a.lm, a.out, a.beam, a.workers)


if __name__ == "__main__":
    main()

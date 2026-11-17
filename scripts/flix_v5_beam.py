"""Flix v5 (F-04): pick the real-world model's decoder on validation, then score the v5 test sets once.

Same two steps as scripts/flix_v4_beam.py, whose decoder code this reuses:

  dump    (GPU)  run each model once on every set and condition; keep the top-K word pieces per frame
                 (runs/flix_v5/beam_cache/<name>/<set>|<cond>.npz), plus the greedy hypotheses
  decode  (CPU)  pyctcdecode beam search + a KenLM 4-gram. For each model, the LM (own training
                 transcripts, or LibriSpeech + own), alpha and beta are chosen at beam width 32 on
                 validation only (film_val + ted_val at clean, 0, -5, -10 dB; mean WER, each set
                 weighted equally); the beam width is then chosen the same way. The test sets are
                 decoded once with that setting.

Test conditions are those of scripts/flix_v6_eval.py: MUSAN noise at clean / 0 / -5 / -10 dB, and
audio that does not belong to the face (silent, talkover at 0 dB, replace). "lips_v5" is ws_v5 with
its audio stream off; it is decoded with ws_v5's setting.

  PYTHONPATH=. .venv/bin/python scripts/flix_v5_beam.py dump --run twin_v5=RUN ws_v5=RUN ws_v6=RUN
  PYTHONPATH=. .venv/bin/python scripts/flix_v5_beam.py decode
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from scripts import flix_v4_beam as v4b
from scripts.flix_v5_eval import align_errors, group_of

CACHE = Path("runs/flix_v5/beam_cache")
OUT = Path("reports/flix_v5")
VAL = {"film_val": "data/flix_v5/film_val.csv", "ted_val": "data/flix_v5/ted_val.csv"}
TEST = {"film": "data/flix_v5/flix_test.csv", "ted": "data/flix_v5/ted_test.csv"}
NOISE = {"clean": float("inf"), "0": 0.0, "-5": -5.0, "-10": -10.0}
WRONG = {"silent": "silent", "talkover": "talkover", "replace": "replace"}
LMS = {"own": "lm/flix_train_4gram.arpa", "libri": "lm/flix_libri_4gram.arpa"}
ALPHAS = [0.0, 0.2, 0.3, 0.5, 0.8, 1.2]
BETAS = [0.0, 0.5, 1.0, 2.0, 3.0]
WIDTHS = [16, 32, 64, 128]
MODELS = ["twin_v5", "ws_v5", "ws_v6"]


# ----------------------------------------------------------------------------- dump (GPU)

def jobs_for(name: str):
    """(split, set name, manifest, condition key, snr, condition, drop_modality) for one model."""
    if name == "lips_v5":
        return [("test", s, m, "clean", float("inf"), "none", "audio") for s, m in TEST.items()]
    out = [("val", s, m, c, snr, "none", None) for s, m in VAL.items() for c, snr in NOISE.items()]
    out += [("test", s, m, c, snr, "none", None) for s, m in TEST.items() for c, snr in NOISE.items()]
    out += [("test", s, m, c, float("inf"), cond, None) for s, m in TEST.items() for c, cond in WRONG.items()]
    return out


def dump(runs: dict[str, str]) -> None:
    import torch
    from torch.utils.data import DataLoader

    from lipscribe.data.dataset import GridDataset, collate
    from lipscribe.decoding import greedy_decode
    from lipscribe.models import build_model

    dev = torch.device("cuda")
    todo = [(n, runs[n]) for n in runs] + ([("lips_v5", runs["ws_v5"])] if "ws_v5" in runs else [])
    for name, run in todo:
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
        cfg = dict(ck["config"]["model"])
        for k in ("init_heads_from", "init_model_from", "init_streams_from"):
            cfg.pop(k, None)
        model = build_model(cfg); model.load_state_dict(ck["model"]); model.to(dev).eval()
        out_dir = CACHE / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run.txt").write_text(run + "\n")
        for split, set_name, man, ckey, snr, cond, drop in jobs_for(name):
            path = out_dir / f"{set_name}|{ckey}.npz"
            if path.exists():
                continue
            ds = GridDataset(man, str(Path(man).parent / "splits.csv"), split, snr_db=snr, condition=cond,
                             noise_splits="Data/Noise_splits/splits.csv", visual_augment=False)
            dl = DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=6)
            vals, idxs, lens, refs, greedy = [], [], [], [], []
            with torch.no_grad(), torch.autocast("cuda"):
                for b in dl:
                    b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
                    if drop:
                        b["drop_modality"] = drop
                    o = model(b)
                    lp = o["log_probs"].float()                      # [T, B, V]
                    greedy += greedy_decode(lp, o["lengths"])
                    top = lp.topk(v4b.TOPK, dim=-1)
                    for i, n in enumerate(o["lengths"].tolist()):
                        vals.append(top.values[:n, i].half().cpu().numpy())
                        idxs.append(top.indices[:n, i].short().cpu().numpy())
                        lens.append(n)
                    refs += b["texts"]
            ids = [r["utt_id"] for r in ds.rows]
            assert len(ids) == len(refs)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, vals=np.concatenate(vals), idxs=np.concatenate(idxs), lens=np.array(lens),
                     refs=np.array(refs, dtype=object), greedy=np.array(greedy, dtype=object),
                     ids=np.array(ids, dtype=object))
            tmp.rename(path)
            print(f"dumped {name} {set_name} {ckey} ({len(refs)} clips)", flush=True)
        del model
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- decode (CPU)

def load(name: str, key: str):
    z = np.load(CACHE / name / f"{key}.npz", allow_pickle=True)
    cuts = np.cumsum(z["lens"])[:-1]
    return {"vals": np.split(z["vals"], cuts), "idxs": np.split(z["idxs"], cuts), "refs": list(z["refs"]),
            "greedy": list(z["greedy"]), "ids": list(z["ids"])}


def val_keys() -> list[str]:
    return [f"{s}|{c}" for s in VAL for c in NOISE]


def test_keys(name: str) -> list[str]:
    if name == "lips_v5":
        return [f"{s}|clean" for s in TEST]
    return [f"{s}|{c}" for s in TEST for c in [*NOISE, *WRONG]]


def tune(workers: int, res: dict) -> None:
    """Fill res[name]['val'] (grid) and res[name]['choice'] for every model, from validation only."""
    from lipscribe.evaluation import wer
    ctx = mp.get_context("fork")
    data = {n: {k: load(n, k) for k in val_keys()} for n in MODELS if n not in res or "choice" not in res[n]}
    if not data:
        return
    for n, sets in data.items():
        res.setdefault(n, {})["val"] = {"greedy": round(float(np.mean(
            [100 * wer(d["refs"], d["greedy"]) for d in sets.values()])), 2)}

    def score(pool, n, a, b, w, use_lm=True):
        return round(float(np.mean([100 * wer(d["refs"], v4b.decode_all(pool, d["vals"], d["idxs"], a, b, w, use_lm))
                                    for d in data[n].values()])), 2)

    for lm_name, lm in LMS.items():
        v4b._init(lm)                          # load the LM once; forked workers share it
        with ctx.Pool(workers) as pool:
            for n in data:
                grid = res[n]["val"]
                if lm_name == "own":
                    grid["beam32_no_lm"] = score(pool, n, 0.0, 0.0, 32, use_lm=False)
                for a, b in itertools.product(ALPHAS, BETAS):
                    if a == 0.0 and b != 0.0:
                        continue
                    grid[f"{lm_name}|{a}|{b}|32"] = score(pool, n, a, b, 32)
                    print(f"{n} val lm={lm_name} alpha={a} beta={b}: {grid[f'{lm_name}|{a}|{b}|32']:.2f}", flush=True)
    for n in data:                             # best LM / alpha / beta at width 32, then the width
        grid = res[n]["val"]
        lm_name, a, b, _ = min((k for k in grid if k.count("|") == 3), key=grid.get).split("|")
        v4b._init(LMS[lm_name])
        with ctx.Pool(workers) as pool:
            for w in WIDTHS:
                key = f"{lm_name}|{a}|{b}|{w}"
                if key not in grid:
                    grid[key] = score(pool, n, float(a), float(b), w)
                    print(f"{n} val width={w}: {grid[key]:.2f}", flush=True)
        w = min(WIDTHS, key=lambda x: (grid[f"{lm_name}|{a}|{b}|{x}"], x))
        res[n]["choice"] = {"lm": lm_name, "alpha": float(a), "beta": float(b), "width": w,
                            "val_wer": grid[f"{lm_name}|{a}|{b}|{w}"]}
        print(n, "choice", res[n]["choice"], flush=True)


def test(workers: int, res: dict, hyps: dict) -> None:
    """Decode every test set once with each model's chosen setting (greedy is kept as the control)."""
    ctx = mp.get_context("fork")
    res["lips_v5"] = {"choice": dict(res["ws_v5"]["choice"], note="ws_v5's setting")}
    for n in [*MODELS, "lips_v5"]:
        ch = res[n]["choice"]
        todo = [k for k in test_keys(n) if f"{n}|{k}" not in hyps]
        if not todo:
            continue
        v4b._init(LMS[ch["lm"]])
        with ctx.Pool(workers) as pool:
            for k in todo:
                d = load(n, k)
                beam = v4b.decode_all(pool, d["vals"], d["idxs"], ch["alpha"], ch["beta"], ch["width"], True)
                hyps[f"{n}|{k}"] = {"ids": d["ids"], "refs": d["refs"], "greedy": d["greedy"], "beam_lm": beam}
                print("decoded test", n, k, flush=True)
        (OUT / "beam_hyps.json").write_text(json.dumps(hyps))


# ----------------------------------------------------------------------------- report

def per_clip(v: dict, which: str):
    rows = []
    for u, r, h in zip(v["ids"], v["refs"], v[which]):
        e, ins = align_errors(r.split(), h.split())
        rows.append((group_of(u), sum(e) + ins, len(r.split())))
    return rows


def wer_of(rows) -> float:
    return 100 * sum(x[1] for x in rows) / sum(x[2] for x in rows)


def boot(ra, rb, rng, n=10000):
    """Paired bootstrap of WER(a) - WER(b), resampling groups (whole talks for TED, clips for films)."""
    groups = sorted({x[0] for x in ra})
    gi = np.array([groups.index(x[0]) for x in ra])
    A = np.bincount(gi, [x[1] for x in ra], len(groups)); B = np.bincount(gi, [x[1] for x in rb], len(groups))
    W = np.bincount(gi, [x[2] for x in ra], len(groups))
    idx = rng.integers(0, len(groups), (n, len(groups)))
    d = 100 * (A[idx].sum(1) - B[idx].sum(1)) / W[idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return 100 * (A.sum() - B.sum()) / W.sum(), lo, hi


def report(res: dict, hyps: dict) -> str:
    rng = np.random.default_rng(0)
    cell = lambda n, k, w: f"{wer_of(per_clip(hyps[f'{n}|{k}'], w)):.1f}" if f"{n}|{k}" in hyps else "–"
    L = ["# Flix v5: the real-world model's decoder and final test numbers (F-04)", "",
         "Generated by `scripts/flix_v5_beam.py`. Models are frozen; nothing here retrains.", "",
         "## Decoder chosen on validation", "",
         ("Validation = `film_val` + `ted_val` (v5 re-cut) at clean, 0, −5 and −10 dB; mean WER %, each set weighted "
         f"equally. Grid: LM ∈ {{own, libri}} × alpha ∈ {ALPHAS} × beta ∈ {BETAS} at beam width 32, then width ∈ {WIDTHS}. "
         "`own` = 4-gram on our training transcripts only (`lm/flix_train_4gram.arpa`); `libri` = 40 M LibriSpeech "
         "words + our training transcripts ×5 (`lm/flix_libri_4gram.arpa`). No validation or test talk is in either LM."), "",
         "| model | LM | alpha | beta | width | val greedy | val beam, no LM | val beam + LM |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for n in MODELS:
        c, v = res[n]["choice"], res[n]["val"]
        L.append(f"| {n} | {c['lm']} | {c['alpha']} | {c['beta']} | {c['width']} | {v['greedy']:.2f} | "
                 f"{v['beam32_no_lm']:.2f} | **{c['val_wer']:.2f}** |")
    L += ["", "## Test WER % with ordinary noise (MUSAN), scored once", "",
          "| test set | SNR | twin_v5 greedy | twin_v5 beam + LM | ws_v5 greedy | ws_v5 beam + LM | lips gain (beam + LM), 95 % CI |",
          "|---|---|---:|---:|---:|---:|---|"]
    for s in TEST:
        for c in NOISE:
            k = f"{s}|{c}"
            pt, lo, hi = boot(per_clip(hyps[f"ws_v5|{k}"], "beam_lm"), per_clip(hyps[f"twin_v5|{k}"], "beam_lm"), rng)
            L.append(f"| {s} | {c} | {cell('twin_v5', k, 'greedy')} | {cell('twin_v5', k, 'beam_lm')} | "
                     f"{cell('ws_v5', k, 'greedy')} | **{cell('ws_v5', k, 'beam_lm')}** | {pt:+.1f} [{lo:+.1f}, {hi:+.1f}] |")
    L += ["", ("Lips gain = WER(ws_v5) − WER(twin_v5); negative means the lips help. Paired bootstrap, 10,000 "
          "resamples of whole talks (TED) or clips (films)."), "",
          "## Audio that does not belong to the face (limitations), beam + LM", "",
          ("Reference is always the on-screen speaker's words. silent: the speaker only mouths. talkover: another "
          "test clip's speech at 0 dB over the speaker. replace: the speaker mouths while someone else talks. "
          "lips_v5 = ws_v5 with its audio stream off (clean row only). Greedy in brackets."), "",
          "| test set | condition | twin_v5 | ws_v5 | ws_v6 | lips_v5 |", "|---|---|---:|---:|---:|---:|"]
    for s in TEST:
        for c in ["clean", *WRONG]:
            k = f"{s}|{c}"
            cells = [f"{cell(n, k, 'beam_lm')} ({cell(n, k, 'greedy')})" if f"{n}|{k}" in hyps else "–"
                     for n in ["twin_v5", "ws_v5", "ws_v6", "lips_v5"]]
            L.append(f"| {s} | {c} | " + " | ".join(cells) + " |")
    L += ["", "## Checkpoints", ""]
    for n in MODELS:
        L.append(f"- {n}: `{(CACHE / n / 'run.txt').read_text().strip()}/best.pt`")
    return "\n".join(L) + "\n"


def decode(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rp, hp = OUT / "beam.json", OUT / "beam_hyps.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    hyps = json.loads(hp.read_text()) if hp.exists() else {}
    tune(workers, res)
    rp.write_text(json.dumps(res, indent=1))
    test(workers, res, hyps)
    rp.write_text(json.dumps(res, indent=1))
    md = report(res, hyps)
    (OUT / "final_decoder.md").write_text(md)
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--run", nargs="+", required=True, help="name=run_dir")
    e = sub.add_parser("decode"); e.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(dict(x.split("=", 1) for x in a.run))
    else:
        decode(a.workers)


if __name__ == "__main__":
    main()

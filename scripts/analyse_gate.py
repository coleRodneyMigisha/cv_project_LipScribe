"""Gate and attention analysis for the proposed model P (card W4-06).

Usage::

    python -m scripts.analyse_gate --checkpoint checkpoints/p_proposed_best.pt \
        --out reports/gate_analysis [--snrs clean 10 5 0 -5] [--examples 3] [--limit N]

Runs the model over the GRID test set at each SNR and writes to ``--out``:

* ``per_utterance_gate.csv``        utt_id, speaker, snr, mean_gate, n_frames
* ``summary.json`` / ``summary.md`` per-SNR n, mean, median, IQR, monotonicity, Spearman rho
* ``gate_boxplot.png``              distribution of the per-utterance mean gate at each SNR
* ``gate_per_frame_examples.png``   g_t for a few utterances, highest vs lowest SNR overlaid
* ``attention_heatmap.png``         audio->video and video->audio attention (mean over heads)

The gate g_t is the audio weight (near 1 = relies on audio). The per-utterance mean
is taken over real frames only (batch mask), as in ``train._masked_gate_mean``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from scripts.train import snr_value, to_device

DEFAULT_SNRS = ["clean", "10", "5", "0", "-5"]
CSV_FIELDS = ("utt_id", "speaker", "snr", "mean_gate", "n_frames")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def order_snrs(labels: Iterable[str]) -> list[str]:
    """Highest SNR first (clean, 10, 5, 0, -5)."""
    return sorted((str(s) for s in labels), key=snr_value, reverse=True)


def _real_frames(batch: dict, gate: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Number of real frames per utterance, from the batch mask (lengths as a fallback)."""
    t = gate.shape[1]
    mask = batch.get("mask")
    if mask is not None:
        return mask[:, :t].bool().sum(dim=1).clamp(min=1).cpu()
    if lengths is not None:
        return lengths.clamp(min=1, max=t).cpu()
    return torch.full((gate.shape[0],), t, dtype=torch.long)


@torch.no_grad()
def collect(model, loaders: Mapping[str, Iterable[dict]], prepare_batch, device: torch.device,
            *, use_amp: bool = False, attention_snr: str | None = None) -> dict:
    """Run ``model`` over every loader and gather per-utterance and per-frame gates.

    Returns ``{"rows": [...], "frames": {snr: {utt_id: ndarray}}, "texts": {utt_id: str},
    "attention": {utt_id: {"a2v": ndarray, "v2a": ndarray}}}``. Attention is kept for
    ``attention_snr`` only (default: the highest SNR), cropped to the real frames.
    """
    model.eval()
    labels = order_snrs(loaders)
    attention_snr = attention_snr or labels[0]
    rows, frames, texts, attention = [], {}, {}, {}
    for label in labels:
        frames[label] = {}
        for batch in loaders[label]:
            batch = prepare_batch(to_device(batch, device))
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(batch)
            if "gate" not in out:
                raise ValueError("model output has no 'gate'; this analysis needs the fusion model P")
            gate = out["gate"].float().cpu()
            n_real = _real_frames(batch, out["gate"], out.get("lengths")).tolist()
            for i, (utt_id, text) in enumerate(zip(batch["utt_ids"], batch["texts"])):
                n = int(n_real[i])
                g = gate[i, :n].numpy()
                frames[label][utt_id] = g
                texts[utt_id] = text
                rows.append({"utt_id": utt_id, "speaker": utt_id.split("_", 1)[0], "snr": label,
                             "mean_gate": float(g.mean()), "n_frames": n})
                if label == attention_snr and "attn_a2v" in out:
                    attention[utt_id] = {
                        "a2v": out["attn_a2v"][i, :n, :n].float().cpu().numpy(),
                        "v2a": out["attn_v2a"][i, :n, :n].float().cpu().numpy(),
                    }
    return {"rows": rows, "frames": frames, "texts": texts, "attention": attention,
            "attention_snr": attention_snr}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        tied = x == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x, y) -> float:
    rx, ry = _rank(np.asarray(x, dtype=float)), _rank(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def summarise(rows: list[dict]) -> dict:
    """Per-SNR statistics, monotonicity check (g falls as SNR falls) and Spearman rho."""
    labels = order_snrs({r["snr"] for r in rows})
    per_snr = {}
    for label in labels:
        g = np.array([r["mean_gate"] for r in rows if r["snr"] == label])
        q1, q3 = np.percentile(g, [25, 75])
        per_snr[label] = {"n": int(len(g)), "mean": float(g.mean()), "median": float(np.median(g)),
                          "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}
    medians = [per_snr[s]["median"] for s in labels]
    means = [per_snr[s]["mean"] for s in labels]
    # labels run from high to low SNR, so the gate must be non-increasing along them.
    mono = lambda v: bool(all(a >= b for a, b in zip(v, v[1:])))
    snr_x = [snr_value(r["snr"]) for r in rows]
    rho = spearman(snr_x, [r["mean_gate"] for r in rows])
    return {"snr_order": labels, "per_snr": per_snr,
            "monotonic_median": mono(medians), "monotonic_mean": mono(means),
            "spearman_rho_snr_vs_gate": rho,
            "expectation": "mean gate decreases as SNR falls (positive rho between SNR and gate)"}


def write_summary(out_dir: Path, summary: dict) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    yes = lambda b: "yes" if b else "no"
    lines = ["# Gate analysis (per-utterance mean gate)", "",
             "| SNR (dB) | n | mean | median | Q1 | Q3 | IQR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label in summary["snr_order"]:
        s = summary["per_snr"][label]
        lines.append(f"| {label} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
                     f"{s['q1']:.4f} | {s['q3']:.4f} | {s['iqr']:.4f} |")
    lines += ["", f"Median gate non-increasing as SNR falls: {yes(summary['monotonic_median'])}",
              f"Mean gate non-increasing as SNR falls: {yes(summary['monotonic_mean'])}",
              f"Spearman rho (SNR vs per-utterance gate): {summary['spearman_rho_snr_vs_gate']:.4f}", ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def pick_examples(frames: dict, n: int) -> list[str]:
    """Deterministic: utterances present at every SNR, sorted by id, evenly spaced."""
    common = sorted(set.intersection(*(set(v) for v in frames.values()))) if frames else []
    if not common or n <= 0:
        return []
    idx = np.unique(np.linspace(0, len(common) - 1, num=min(n, len(common))).round().astype(int))
    return [common[i] for i in idx]


def plot_boxplot(path: Path, rows: list[dict], labels: list[str]) -> None:
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    data = [[r["mean_gate"] for r in rows if r["snr"] == s] for s in labels]
    axis.boxplot(data, tick_labels=labels, showmeans=True)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Mean gate per utterance (audio weight)")
    axis.set_title("Fusion gate against noise level")
    axis.set_ylim(0, 1)
    axis.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_examples(path: Path, frames: dict, texts: dict, examples: list[str], labels: list[str],
                  fps: float = 25.0) -> None:
    n = max(1, len(examples))
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.8 * n), squeeze=False, constrained_layout=True)
    hi, lo = labels[0], labels[-1]
    for axis, utt in zip(axes[:, 0], examples):
        for label, colour in ((hi, "tab:blue"), (lo, "tab:red")):
            g = frames[label][utt]
            axis.plot(np.arange(len(g)) / fps, g, color=colour, label=f"{label} dB")
        axis.set_ylim(0, 1)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Gate g_t (audio weight)")
        axis.set_title(f"{texts.get(utt, utt)}  [{utt}]", fontsize=10)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="lower right")
    if not examples:
        axes[0, 0].text(0.5, 0.5, "no example utterances", ha="center")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_attention(path: Path, attention: dict, utt: str | None, text: str, snr: str,
                   fps: float = 25.0) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    if utt is None or utt not in attention:
        for axis in axes:
            axis.text(0.5, 0.5, "no attention weights", ha="center")
    else:
        for axis, key, title in ((axes[0], "a2v", "Audio queries, video keys"),
                                 (axes[1], "v2a", "Video queries, audio keys")):
            w = attention[utt][key]
            extent = (0, w.shape[1] / fps, w.shape[0] / fps, 0)
            image = axis.imshow(w, aspect="auto", cmap="viridis", extent=extent)
            axis.set_xlabel("Key time (s)")
            axis.set_ylabel("Query time (s)")
            axis.set_title(title)
            fig.colorbar(image, ax=axis, label="Attention weight (mean over heads)")
        fig.suptitle(f"{text}  [{utt}, {snr} dB]")
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_analysis(model, loaders: Mapping[str, Iterable[dict]], prepare_batch, device: torch.device,
                 out_dir: Path, *, n_examples: int = 3, use_amp: bool = False) -> dict:
    """Collect gates, then write every output file. Returns the summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = collect(model, loaders, prepare_batch, device, use_amp=use_amp)
    rows, frames = result["rows"], result["frames"]
    labels = order_snrs(frames)

    with (out_dir / "per_utterance_gate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({**r, "mean_gate": f"{r['mean_gate']:.6f}"} for r in rows)

    summary = summarise(rows)
    write_summary(out_dir, summary)
    plot_boxplot(out_dir / "gate_boxplot.png", rows, labels)
    examples = pick_examples(frames, n_examples)
    plot_examples(out_dir / "gate_per_frame_examples.png", frames, result["texts"], examples, labels)
    att_utt = examples[0] if examples else None
    plot_attention(out_dir / "attention_heatmap.png", result["attention"], att_utt,
                   result["texts"].get(att_utt, ""), result["attention_snr"])

    print(f"{'SNR':>6} {'n':>5} {'mean':>7} {'median':>7} {'IQR':>7}")
    for label in summary["snr_order"]:
        s = summary["per_snr"][label]
        print(f"{label:>6} {s['n']:>5} {s['mean']:7.4f} {s['median']:7.4f} {s['iqr']:7.4f}")
    print(f"Monotonic (median): {summary['monotonic_median']}, (mean): {summary['monotonic_mean']}, "
          f"Spearman rho: {summary['spearman_rho_snr_vs_gate']:.4f}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/gate_analysis"))
    parser.add_argument("--snrs", nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="first N test utterances per SNR")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    from torch.utils.data import DataLoader, Subset

    from lipscribe.data.dataset import GridDataset, collate
    from lipscribe.models import build_model
    from scripts.train import build_feature_step, load_checkpoint

    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    prepare_batch = build_feature_step(cfg, device)
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"

    loaders = {}
    for label in (str(s) for s in args.snrs):
        dataset = GridDataset(
            data_cfg["manifest"], data_cfg["splits"], "test",
            snr_db=snr_value(label), noise_splits=data_cfg.get("noise_splits"),
            noisy_dir=data_cfg.get("noisy_dir", "data/processed/noisy_test"),
        )
        if args.limit:
            dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        loaders[label] = DataLoader(
            dataset, batch_size=int(train_cfg["batch_size"]), shuffle=False,
            collate_fn=collate, num_workers=int(train_cfg["num_workers"]),
            pin_memory=device.type == "cuda",
        )
    run_analysis(model, loaders, prepare_batch, device, args.out,
                 n_examples=args.examples, use_amp=use_amp)
    print(f"Saved gate analysis in {args.out}")


if __name__ == "__main__":
    main()

"""Evaluate a saved checkpoint on the held-out test speakers at every SNR.

Usage::

    python -m scripts.evaluate --checkpoint runs/<run>/best.pt
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from lipscribe.data.dataset import GridDataset, collate
from lipscribe.decoding import greedy_decode
from lipscribe.evaluation import cer, wer
from lipscribe.models import build_model
from scripts.train import (
    build_feature_step, load_checkpoint, snr_value, to_device,
)


def score_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict, list[dict]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    prepare_batch = build_feature_step(cfg, device)
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"
    results, details = {}, []

    for label in data_cfg["eval_snrs"]:
        label = str(label)
        dataset = GridDataset(
            data_cfg["manifest"], data_cfg["splits"], "test",
            snr_db=snr_value(label), noise_splits=data_cfg.get("noise_splits"),
            noisy_dir=data_cfg.get("noisy_dir", "data/processed/noisy_test"),
        )
        loader = DataLoader(
            dataset, batch_size=int(train_cfg["batch_size"]), shuffle=False,
            collate_fn=collate, num_workers=int(train_cfg["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        refs, hyps = [], []
        by_speaker: dict[str, tuple[list[str], list[str]]] = {}
        with torch.inference_mode():
            for batch in loader:
                batch = prepare_batch(to_device(batch, device))
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    output = model(batch)
                decoded = greedy_decode(output["log_probs"].float(), output["lengths"])
                refs.extend(batch["texts"])
                hyps.extend(decoded)
                for utt_id, ref, hyp in zip(batch["utt_ids"], batch["texts"], decoded):
                    speaker = utt_id.split("_", 1)[0]
                    speaker_refs, speaker_hyps = by_speaker.setdefault(speaker, ([], []))
                    speaker_refs.append(ref)
                    speaker_hyps.append(hyp)

        results[label] = {"n_utts": len(refs), "wer": wer(refs, hyps), "cer": cer(refs, hyps)}
        details.append({"snr": label, "scope": "all", **results[label]})
        for speaker, (speaker_refs, speaker_hyps) in sorted(by_speaker.items()):
            details.append({"snr": label, "scope": speaker, "n_utts": len(speaker_refs),
                            "wer": wer(speaker_refs, speaker_hyps),
                            "cer": cer(speaker_refs, speaker_hyps)})
        print(f"{label:>5} dB: {len(refs)} utterances, WER {results[label]['wer']:.4f}, "
              f"CER {results[label]['cer']:.4f}", flush=True)
    return results, details


def save_metrics(path: Path, details: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("snr", "scope", "n_utts", "wer", "cer"))
        writer.writeheader()
        writer.writerows(details)


def save_test_curve(path: Path, results: dict, order: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = list(range(len(order)))
    for axis, metric, title in zip(axes, ("wer", "cer"), ("Test WER", "Test CER")):
        axis.plot(x, [results[label][metric] for label in order], marker="o")
        axis.set_xticks(x, order)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("Error rate")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_training_curves(run_dir: Path, path: Path) -> Path:
    accumulator = EventAccumulator(str(run_dir / "tensorboard"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    series = {}
    for tag in ("epoch/train_loss", "val/mean_loss", "val/mean_wer", "val/mean_cer"):
        if tag in tags:
            events = accumulator.Scalars(tag)
            series[tag] = ([event.step for event in events], [event.value for event in events])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for tag, axis in (("epoch/train_loss", axes[0]), ("val/mean_loss", axes[0]),
                      ("val/mean_wer", axes[1]), ("val/mean_cer", axes[1])):
        if tag in series:
            axis.plot(*series[tag], marker="o", label=tag.split("/")[-1])
    axes[0].set_title("Training and validation loss")
    axes[1].set_title("Validation error rates")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def update_results_csv(results_path: Path, run_dir: Path, metrics: dict) -> None:
    if not results_path.exists():
        raise FileNotFoundError(f"Training results not found: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    added_fields = ["config_path", "checkpoint_path", "test_wer_by_snr", "test_cer_by_snr"]
    fieldnames.extend(field for field in added_fields if field not in fieldnames)
    matching = [row for row in rows if Path(row.get("run_dir", "")).resolve() == run_dir.resolve()]
    if not matching:
        raise ValueError(f"No results.csv row for run {run_dir}")
    row = matching[-1]
    row.update({
        "config_path": str(run_dir / "config.yaml"),
        "checkpoint_path": str(run_dir / "best.pt"),
        "test_wer_by_snr": ";".join(f"{snr}={metrics[snr]['wer']:.4f}" for snr in metrics),
        "test_cer_by_snr": ";".join(f"{snr}={metrics[snr]['cer']:.4f}" for snr in metrics),
    })
    tmp = results_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(results_path)


def write_report(path: Path, checkpoint_path: Path, results: dict) -> None:
    try:
        ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
        title = ckpt.get("config", {}).get("experiment", "Model").replace("_", " ").title()
    except Exception:
        title = "Model"
    rows = ["| SNR | Test utterances | WER (%) | CER (%) |", "|---|---:|---:|---:|"]
    rows.extend(f"| {snr} | {values['n_utts']} | {100 * values['wer']:.2f} | "
                f"{100 * values['cer']:.2f} |"
                for snr, values in results.items())
    path.write_text(
        f"# {title} Test Results\n\n"
        f"Config: `{checkpoint_path.parent / 'config.yaml'}`\n\n"
        f"Checkpoint: `{checkpoint_path}`\n\n"
        + "\n".join(rows)
        + "\n\nCurves: `training_curves.png` and `test_snr_curves.png`.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    run_dir = checkpoint_path.parent
    device = torch.device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    results, details = score_checkpoint(checkpoint_path, device)
    order = [str(snr) for snr in checkpoint["config"]["data"]["eval_snrs"]]
    report_dir = Path(__file__).resolve().parents[1] / "reports" / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(report_dir / "test_metrics.csv", details)
    save_test_curve(report_dir / "test_snr_curves.png", results, order)
    save_training_curves(run_dir, report_dir / "training_curves.png")
    update_results_csv(run_dir.parent.parent / "experiments" / "results.csv", run_dir, results)
    write_report(report_dir / "evaluation_report.md", checkpoint_path, results)
    print(f"Saved evaluation artifacts in {report_dir}")


if __name__ == "__main__":
    main()
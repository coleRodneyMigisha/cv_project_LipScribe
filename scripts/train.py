"""Train a LipScribe model from one YAML config (implementation document, Section 8).

Usage::

    python -m scripts.train --config configs/b1_audio_only.yaml
    python -m scripts.train --config configs/b1_audio_only.yaml --limit 64 --max-epochs 2
    python -m scripts.train --resume runs/b1_audio_only_20261031-120000/latest.pt

Each run gets its own directory ``runs/<experiment>_<timestamp>/`` holding the
config, the Git commit, ``latest.pt``, ``best.pt`` (by mean validation WER) and
TensorBoard logs. A finished run appends one row to ``experiments/results.csv``.
Resuming continues in the run directory of the checkpoint.

Batch keys and model outputs follow ``docs/INTERFACES.md``. The training loop
(:func:`fit`) takes the model, loaders, feature step and decoder as arguments, so
it can be tested with stand-ins; :func:`main` wires in the real modules.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from lipscribe.data import tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "experiments" / "results.csv"
RESULTS_FIELDS = (
    "finished_at", "experiment", "run_dir", "commit", "seed", "epochs_run",
    "best_epoch", "best_val_wer_mean", "best_val_cer_mean", "val_wer_by_snr",
    "nonfinite_steps",
)

# Starting values from Table "Initial training hyperparameters"; a config overrides them.
TRAIN_DEFAULTS = {
    "optimiser": "adamw",
    "lr": 3.0e-4,
    "weight_decay": 1.0e-2,
    "batch_size": 16,
    "grad_accum": 1,
    "max_epochs": 60,
    "warmup_epochs": 1,
    "patience": 8,
    "grad_clip": 5.0,
    "amp": True,
    "num_workers": 4,
}


# ---------------------------------------------------------------------------
# Set-up
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["train"] = {**TRAIN_DEFAULTS, **(cfg.get("train") or {})}
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    """Current commit hash, marked ``-dirty`` if there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def make_run_dir(cfg: dict, runs_root: str | Path) -> Path:
    """Create a new run directory and save the config and commit into it.

    Run directories are never overwritten: ``mkdir`` fails if the name exists.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_root) / f"{cfg['experiment']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    (run_dir / "commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
    return run_dir


def vocabulary() -> dict:
    """Vocabulary saved in every checkpoint so decoding never depends on code version."""
    return {"blank": tokenizer.BLANK_ID,
            "symbols": [tokenizer.ID_TO_CHAR.get(i, "<blank>") for i in range(tokenizer.VOCAB_SIZE)]}


# ---------------------------------------------------------------------------
# Model, optimiser and schedule
# ---------------------------------------------------------------------------

def init_lstm_forget_bias(model: nn.Module, value: float = 1.0) -> None:
    """Set every LSTM forget-gate bias to ``value``.

    PyTorch orders gate biases (input, forget, cell, output) and adds
    ``bias_ih`` and ``bias_hh``, so the forget slice of ``bias_ih`` is set to
    ``value`` and that of ``bias_hh`` to 0.
    """
    for module in model.modules():
        if not isinstance(module, nn.LSTM):
            continue
        for name, param in module.named_parameters():
            if not name.startswith("bias_"):
                continue
            h = param.shape[0] // 4
            with torch.no_grad():
                param[h:2 * h].fill_(value if name.startswith("bias_ih") else 0.0)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on weight matrices only; none on biases or norm parameters."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(optimiser: torch.optim.Optimizer, warmup_steps: int,
                  total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up to the peak learning rate, then cosine decay to zero."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimiser, factor)


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def ctc_loss_fn() -> nn.CTCLoss:
    # Default reduction "mean" divides each loss by its target length, then averages.
    return nn.CTCLoss(blank=tokenizer.BLANK_ID, zero_infinity=True)


def compute_loss(criterion: nn.CTCLoss, out: dict, batch: dict) -> torch.Tensor:
    # CTC runs in float32 even under AMP.
    return criterion(out["log_probs"].float(), batch["targets"],
                     out["lengths"], batch["target_lengths"])


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimiser, scheduler, scaler, *,
                    device, prepare_batch, train_cfg, writer, state) -> float:
    """One pass over ``loader``. Updates ``state`` (global step, non-finite count)."""
    model.train()
    use_amp = scaler.is_enabled()
    accum = max(1, int(train_cfg["grad_accum"]))
    total, count = 0.0, 0
    optimiser.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        batch = prepare_batch(to_device(batch, device))
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(batch)
        loss = compute_loss(criterion, out, batch)

        if not torch.isfinite(loss):
            # Skip and log; more than a handful points at a data or length bug.
            state["nonfinite_steps"] += 1
            writer.add_scalar("train/nonfinite_steps", state["nonfinite_steps"], state["step"])
            print(f"  warning: non-finite loss at step {state['step']} "
                  f"(utts {batch.get('utt_ids', ['?'])[:4]}...), skipped", file=sys.stderr)
            optimiser.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss / accum).backward()
        total += loss.item()
        count += 1
        if (i + 1) % accum != 0 and (i + 1) != len(loader):
            continue

        scaler.unscale_(optimiser)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        scaler.step(optimiser)  # skipped by the scaler if gradients are inf/NaN
        scaler.update()
        optimiser.zero_grad(set_to_none=True)
        scheduler.step()
        state["step"] += 1

        writer.add_scalar("train/loss", loss.item(), state["step"])
        writer.add_scalar("train/grad_norm", float(grad_norm), state["step"])
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], state["step"])
        if "gate" in out:
            writer.add_scalar("train/gate_mean", _masked_gate_mean(out["gate"], batch), state["step"])

    return total / max(1, count)


@torch.no_grad()
def validate(model, val_loaders: dict, criterion, *, device, prepare_batch,
             decode_fn, wer_fn, cer_fn, use_amp) -> dict:
    """Validation loss and greedy WER/CER at every SNR, plus their means."""
    model.eval()
    metrics = {}
    for snr, loader in val_loaders.items():
        losses, refs, hyps, gates = [], [], [], []
        for batch in loader:
            batch = prepare_batch(to_device(batch, device))
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(batch)
            loss = compute_loss(criterion, out, batch)
            if torch.isfinite(loss):
                losses.append(loss.item())
            hyps.extend(decode_fn(out["log_probs"].float(), out["lengths"]))
            refs.extend(batch["texts"])
            if "gate" in out:
                gates.append(_masked_gate_mean(out["gate"], batch))
        metrics[snr] = {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "wer": float(wer_fn(refs, hyps)),
            "cer": float(cer_fn(refs, hyps)),
        }
        if gates:
            metrics[snr]["gate_mean"] = float(np.mean(gates))
    for key in ("loss", "wer", "cer"):
        metrics[f"mean_{key}"] = float(np.mean([metrics[s][key] for s in val_loaders]))
    return metrics


def _masked_gate_mean(gate: torch.Tensor, batch: dict) -> float:
    mask = batch.get("mask")
    if mask is None or mask.shape != gate.shape:
        return gate.float().mean().item()
    return gate.float()[mask].mean().item()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, *, model, optimiser, scheduler, scaler, epoch,
                    cfg, state) -> None:
    payload = {
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "config": cfg,
        "vocab": vocabulary(),
        "state": state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint behind


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    # Our own checkpoints hold optimiser and RNG state, not just tensors.
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_rng(rng: dict) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def fit(cfg: dict, model: nn.Module, train_loader: Iterable, val_loaders: dict,
        run_dir: Path, *, device: torch.device,
        prepare_batch: Callable[[dict], dict] = lambda b: b,
        decode_fn: Callable, wer_fn: Callable, cer_fn: Callable,
        resume: dict | None = None,
        results_csv: Path | None = RESULTS_CSV) -> dict:
    """Train until early stopping or ``max_epochs``; return the final ``state``.

    ``resume`` is a checkpoint dict from :func:`load_checkpoint`. If it is None
    the model is freshly initialised (LSTM forget-gate biases set to 1).
    """
    tc = cfg["train"]
    model.to(device)
    if resume is None:
        init_lstm_forget_bias(model)

    optimiser = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"])
    steps_per_epoch = math.ceil(len(train_loader) / max(1, int(tc["grad_accum"])))
    scheduler = warmup_cosine(optimiser, warmup_steps=int(tc["warmup_epochs"] * steps_per_epoch),
                              total_steps=int(tc["max_epochs"] * steps_per_epoch))
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = ctc_loss_fn()

    state = {"step": 0, "best_wer": float("inf"), "best_cer": float("inf"),
             "best_epoch": -1, "best_by_snr": {}, "bad_epochs": 0, "nonfinite_steps": 0}
    start_epoch = 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimiser.load_state_dict(resume["optimiser"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        state.update(resume["state"])
        restore_rng(resume["rng"])
        start_epoch = resume["epoch"] + 1
        print(f"Resumed from epoch {resume['epoch']} (best mean WER {state['best_wer']:.4f})")

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    ckpt = dict(model=model, optimiser=optimiser, scheduler=scheduler, scaler=scaler, cfg=cfg)
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, tc["max_epochs"]):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimiser, scheduler, scaler,
                device=device, prepare_batch=prepare_batch, train_cfg=tc,
                writer=writer, state=state)
            val = validate(model, val_loaders, criterion, device=device,
                           prepare_batch=prepare_batch, decode_fn=decode_fn,
                           wer_fn=wer_fn, cer_fn=cer_fn, use_amp=use_amp)
            _log_epoch(writer, epoch, train_loss, val, val_loaders)

            improved = val["mean_wer"] < state["best_wer"]
            if improved:
                state.update(best_wer=val["mean_wer"], best_cer=val["mean_cer"],
                             best_epoch=epoch, bad_epochs=0,
                             best_by_snr={str(s): val[s]["wer"] for s in val_loaders})
            else:
                state["bad_epochs"] += 1

            save_checkpoint(run_dir / "latest.pt", epoch=epoch, state=state, **ckpt)
            if improved:
                save_checkpoint(run_dir / "best.pt", epoch=epoch, state=state, **ckpt)

            per_snr = "  ".join(f"{s}:{val[s]['wer']:.3f}" for s in val_loaders)
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val['mean_loss']:.4f}  "
                  f"WER {val['mean_wer']:.4f}  CER {val['mean_cer']:.4f}  [{per_snr}]  "
                  f"{time.time() - t0:.0f}s{'  *best*' if improved else ''}")

            if state["bad_epochs"] >= tc["patience"]:
                print(f"Early stopping: no improvement for {tc['patience']} epochs.")
                break
    finally:
        writer.close()

    state["epochs_run"] = epoch + 1
    if results_csv is not None:
        append_results(results_csv, cfg, run_dir, state)
    return state


def _log_epoch(writer, epoch, train_loss, val, val_loaders) -> None:
    writer.add_scalar("epoch/train_loss", train_loss, epoch)
    for key in ("loss", "wer", "cer"):
        writer.add_scalar(f"val/mean_{key}", val[f"mean_{key}"], epoch)
    for snr in val_loaders:
        for key, value in val[snr].items():
            writer.add_scalar(f"val_{key}/{snr}", value, epoch)


def append_results(path: Path, cfg: dict, run_dir: Path, state: dict) -> None:
    """Append one summary row to ``experiments/results.csv``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    commit_file = run_dir / "commit.txt"
    row = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": cfg["experiment"],
        "run_dir": str(run_dir),
        "commit": commit_file.read_text().strip() if commit_file.exists() else git_commit(),
        "seed": cfg.get("seed"),
        "epochs_run": state["epochs_run"],
        "best_epoch": state["best_epoch"],
        "best_val_wer_mean": f"{state['best_wer']:.4f}",
        "best_val_cer_mean": f"{state['best_cer']:.4f}",
        "val_wer_by_snr": ";".join(f"{s}={w:.4f}" for s, w in state["best_by_snr"].items()),
        "nonfinite_steps": state["nonfinite_steps"],
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Wiring to the real LipScribe modules
# ---------------------------------------------------------------------------

def snr_value(snr) -> float:
    """Config SNR label -> dB value. ``clean`` is +inf (see INTERFACES.md item ``snr_db``)."""
    return float("inf") if str(snr).lower() == "clean" else float(snr)


def build_loaders(cfg: dict, limit: int | None = None):
    """Training loader and one validation loader per SNR (W2-09 dataset)."""
    from torch.utils.data import DataLoader, Subset

    from lipscribe.data.dataset import GridDataset, collate

    d, tc = cfg["data"], cfg["train"]
    train_ds = GridDataset(d["manifest"], d["splits"], "train",
                           snrs=[snr_value(s) for s in d["train_snrs"]])
    val_sets = {str(s): GridDataset(d["manifest"], d["splits"], "val", snr_db=snr_value(s))
                for s in d["eval_snrs"]}
    if limit:
        train_ds = Subset(train_ds, range(min(limit, len(train_ds))))
        val_sets = {s: Subset(ds, range(min(limit, len(ds)))) for s, ds in val_sets.items()}

    common = dict(batch_size=tc["batch_size"], collate_fn=collate,
                  num_workers=tc["num_workers"], pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loaders = {s: DataLoader(ds, shuffle=False, **common) for s, ds in val_sets.items()}
    return train_loader, val_loaders


def build_feature_step(cfg: dict, device: torch.device) -> Callable[[dict], dict]:
    """Log-Mel + normalisation on the GPU, added to the batch as ``features`` (W2-07, W2-08)."""
    if "audio" not in cfg["model"].get("streams", []):
        return lambda batch: batch

    from lipscribe.preprocessing.features import logmel, normalise

    stats = dict(np.load(cfg["data"].get("feature_stats", "data/stats/feature_norm.npz")))
    stats = {k: torch.as_tensor(v, device=device) for k, v in stats.items()}

    def prepare(batch: dict) -> dict:
        batch["features"] = normalise(logmel(batch["audio"]), stats)
        return batch
    return prepare


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="experiment YAML in configs/")
    p.add_argument("--resume", help="checkpoint to resume (continues in its run directory)")
    p.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    p.add_argument("--limit", type=int, help="use only the first N utterances per split (smoke runs)")
    p.add_argument("--max-epochs", type=int, help="override train.max_epochs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if not args.config and not args.resume:
        p.error("give --config or --resume")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)

    resume = None
    if args.resume:
        resume = load_checkpoint(args.resume)
        cfg = resume["config"]
        run_dir = Path(args.resume).resolve().parent
    else:
        cfg = load_config(args.config)
        set_seed(int(cfg.get("seed", 1)))
    if args.max_epochs is not None:
        cfg = copy.deepcopy(cfg)
        cfg["train"]["max_epochs"] = args.max_epochs
    if resume is None:
        run_dir = make_run_dir(cfg, args.runs_root)
    print(f"Run directory: {run_dir}")

    from lipscribe.decoding import greedy_decode
    from lipscribe.evaluation import cer, wer
    from lipscribe.models import build_model

    train_loader, val_loaders = build_loaders(cfg, limit=args.limit)
    model = build_model(cfg["model"])
    fit(cfg, model, train_loader, val_loaders, run_dir, device=device,
        prepare_batch=build_feature_step(cfg, device),
        decode_fn=greedy_decode, wer_fn=wer, cer_fn=cer, resume=resume)


if __name__ == "__main__":
    main()

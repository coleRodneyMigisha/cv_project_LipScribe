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
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from lipscribe.data import tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "experiments" / "results.csv"
RESULTS_FIELDS = (
    "finished_at", "experiment", "run_dir", "commit", "seed", "epochs_run",
    "best_epoch", "best_val_wer_mean", "best_val_cer_mean", "val_wer_by_snr",
    "nonfinite_steps", "config_path", "checkpoint_path", "test_wer_by_snr",
    "test_cer_by_snr",
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


def resolve_init_from(cfg: dict) -> dict:
    """Resolve pretrained encoder checkpoint paths from ``cfg`` (W4-05).

    Supports both ``model.init_from: {audio_encoder: ..., visual_encoder: ...}``
    and the shorthand ``init_visual_from`` / ``init_audio_from`` keys (under
    ``model``, ``train`` or top-level).
    """
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("train") or {}
    init_from = dict(model_cfg.get("init_from") or cfg.get("init_from") or {})
    vis_ckpt = (
        model_cfg.get("init_visual_from")
        or train_cfg.get("init_visual_from")
        or cfg.get("init_visual_from")
    )
    if vis_ckpt:
        init_from["visual_encoder"] = vis_ckpt
    aud_ckpt = (
        model_cfg.get("init_audio_from")
        or train_cfg.get("init_audio_from")
        or cfg.get("init_audio_from")
    )
    if aud_ckpt:
        init_from["audio_encoder"] = aud_ckpt
    return init_from


def load_pretrained(model: nn.Module, init_from: dict, map_location="cpu") -> dict:
    """Copy sub-module weights from other runs' checkpoints into ``model``.

    ``init_from`` maps a sub-module name to a checkpoint, e.g.
    ``{"audio_encoder": "checkpoints/b1_audio_only_best.pt",
       "visual_encoder": "checkpoints/b2_visual_only_v2_best.pt"}``.
    Every parameter of that sub-module must be present with the same shape in
    the source checkpoint (B1/B2/B3 all name their encoders the same way), so a
    mismatched encoder type fails loudly instead of silently training from scratch.
    Returns ``{module: number_of_tensors_loaded}``.
    """
    loaded = {}
    for module_name, ckpt_path in (init_from or {}).items():
        path = Path(ckpt_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"init_from.{module_name}: checkpoint {path} not found")
        source = torch.load(path, map_location=map_location, weights_only=False)
        source = source.get("model", source)
        target = getattr(model, module_name, None)
        if target is None:
            raise ValueError(f"init_from: model has no sub-module {module_name!r}")
        prefix = module_name + "."
        sub = {k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}
        own = target.state_dict()
        missing = [k for k in own if k not in sub]
        bad_shape = [k for k in own if k in sub and sub[k].shape != own[k].shape]
        if missing or bad_shape:
            raise ValueError(
                f"init_from.{module_name}: {path} does not match this model "
                f"(missing {missing[:3]}, shape mismatch {bad_shape[:3]}). "
                "Check that model.visual_encoder matches the run being loaded.")
        target.load_state_dict({k: sub[k] for k in own})
        loaded[module_name] = len(own)
        print(f"Initialised {module_name} from {path} ({len(own)} tensors)")
    return loaded


def param_groups(model: nn.Module, weight_decay: float, lr: float | None = None,
                 lr_mult: dict | None = None) -> list[dict]:
    """Weight decay on weight matrices only; none on biases or norm parameters.

    ``lr_mult`` (optional, ``train.lr_mult``) maps a parameter-name prefix such as
    ``visual_encoder`` or ``visual_encoder.trunk.6`` to a multiplier of ``lr``; the longest
    matching prefix wins. Without it the two groups are exactly as before.
    """
    lr_mult = dict(lr_mult or {})
    groups: dict[tuple[bool, float], list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = 1.0
        for prefix in sorted(lr_mult, key=len, reverse=True):
            if name == prefix or name.startswith(prefix + "."):
                mult = float(lr_mult[prefix]); break
        nd = param.ndim <= 1 or name.endswith(".bias")
        groups.setdefault((nd, mult), []).append(param)
    out = []
    for (nd, mult), params in sorted(groups.items(), key=lambda kv: (kv[0][1] != 1.0, kv[0][0])):
        g = {"params": params, "weight_decay": 0.0 if nd else weight_decay}
        if mult != 1.0 and lr is not None:
            g["lr"] = lr * mult
        out.append(g)
    if not lr_mult:          # keep the original [decay, no_decay] order (and empty groups)
        dec = groups.get((False, 1.0), []); nod = groups.get((True, 1.0), [])
        return [{"params": dec, "weight_decay": weight_decay}, {"params": nod, "weight_decay": 0.0}]
    return out


def warmup_cosine(optimiser: torch.optim.Optimizer, warmup_steps: int,
                  total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up to the peak learning rate, then cosine decay to zero.

    ``scheduler.total_ref["total"]`` can be lowered during training (time budget)
    so the cosine still reaches zero at the new final step.
    """
    ref = {"total": total_steps}

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, ref["total"] - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, factor)
    scheduler.total_ref = ref
    return scheduler


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

    ``train.time_budget_min`` (optional) caps wall-clock training time: after
    every epoch the mean epoch time is measured and, if ``max_epochs`` would not
    fit in the remaining budget, the run is shortened and the cosine schedule is
    compressed to end at the new last epoch (so the learning rate still anneals).

    ``resume`` is a checkpoint dict from :func:`load_checkpoint`. If it is None
    the model is freshly initialised (LSTM forget-gate biases set to 1).
    """
    tc = cfg["train"]
    model.to(device)
    if resume is None:
        init_lstm_forget_bias(model)
        # After the forget-bias init, so pretrained LSTM biases are not overwritten.
        init_from = resolve_init_from(cfg)
        if init_from:
            load_pretrained(model, init_from)
    # Optional model.freeze: [sub-module, ...] keeps those weights fixed (param_groups
    # skips parameters with requires_grad=False). Used for small-data fine-tuning (X6 Flix).
    for module_name in (cfg.get("model") or {}).get("freeze") or []:
        try:                                  # dotted paths allowed, e.g. visual_encoder.trunk.0
            sub = model.get_submodule(module_name)
        except AttributeError:
            raise ValueError(f"model.freeze: model has no sub-module {module_name!r}") from None
        sub.requires_grad_(False)
        print(f"Frozen {module_name}")
    # train.select_metric: "wer" (default) or "cer" picks best.pt / early stopping.
    # CER is the useful signal when WER saturates near 100% (open-vocabulary film clips).
    select_key = "mean_cer" if str(tc.get("select_metric", "wer")).lower() == "cer" else "mean_wer"

    optimiser = torch.optim.AdamW(param_groups(model, tc["weight_decay"], lr=tc["lr"],
                                               lr_mult=tc.get("lr_mult")), lr=tc["lr"])
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
        if state.get("budget_max_epochs"):
            scheduler.total_ref["total"] = int(state["budget_max_epochs"] * steps_per_epoch)
        start_epoch = resume["epoch"] + 1
        print(f"Resumed from epoch {resume['epoch']} (best mean WER {state['best_wer']:.4f})")

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    ckpt = dict(model=model, optimiser=optimiser, scheduler=scheduler, scaler=scaler, cfg=cfg)
    epoch = start_epoch - 1
    budget_s = float(tc.get("time_budget_min") or 0) * 60.0
    max_epochs = int(state.get("budget_max_epochs") or tc["max_epochs"])
    session_t0 = time.time()
    try:
        for epoch in range(start_epoch, min(int(tc["max_epochs"]), max_epochs)):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimiser, scheduler, scaler,
                device=device, prepare_batch=prepare_batch, train_cfg=tc,
                writer=writer, state=state)
            val = validate(model, val_loaders, criterion, device=device,
                           prepare_batch=prepare_batch, decode_fn=decode_fn,
                           wer_fn=wer_fn, cer_fn=cer_fn, use_amp=use_amp)
            _log_epoch(writer, epoch, train_loss, val, val_loaders)
            log_epoch_history(run_dir / "history.csv", epoch, train_loss, val, val_loaders,
                              lr=scheduler.get_last_lr()[0], elapsed=time.time() - t0)
            plot_history(run_dir / "history.csv", run_dir / "learning_curves.png")

            improved = val[select_key] < state["best_cer" if select_key == "mean_cer" else "best_wer"]
            if improved:
                state.update(best_wer=val["mean_wer"], best_cer=val["mean_cer"],
                             best_epoch=epoch, bad_epochs=0,
                             best_by_snr={str(s): val[s]["wer"] for s in val_loaders})
            else:
                state["bad_epochs"] += 1

            save_checkpoint(run_dir / "latest.pt", epoch=epoch, state=state, **ckpt)
            if improved:
                save_checkpoint(run_dir / "best.pt", epoch=epoch, state=state, **ckpt)
            if improved and results_csv is not None:   # no convenience copy for --no-results smoke runs
                try:
                    top_ckpt_dir = REPO_ROOT / "checkpoints"
                    top_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(run_dir / "best.pt", top_ckpt_dir / f"{cfg.get('experiment', 'model')}_best.pt")
                except Exception:
                    pass

            per_snr = "  ".join(f"{s}:{val[s]['wer']:.3f}" for s in val_loaders)
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val['mean_loss']:.4f}  "
                  f"WER {val['mean_wer']:.4f}  CER {val['mean_cer']:.4f}  [{per_snr}]  "
                  f"{time.time() - t0:.0f}s{'  *best (saved best.pt)*' if improved else ''}")

            if state["bad_epochs"] >= tc["patience"]:
                print(f"Early stopping: no improvement for {tc['patience']} epochs.")
                break

            if budget_s > 0:
                spent = time.time() - session_t0
                per_epoch = spent / (epoch - start_epoch + 1)
                fit_epochs = epoch + 1 + int(max(0.0, budget_s - spent) // per_epoch)
                if fit_epochs < max_epochs:
                    max_epochs = max(epoch + 1, fit_epochs)
                    state["budget_max_epochs"] = max_epochs
                    scheduler.total_ref["total"] = int(max_epochs * steps_per_epoch)
                    print(f"  time budget {budget_s / 60:.0f} min, {per_epoch / 60:.1f} min/epoch: "
                          f"training shortened to {max_epochs} epochs (cosine compressed)")
                if epoch + 1 >= max_epochs:
                    print("Time budget reached.")
                    break
    finally:
        writer.close()

    state["epochs_run"] = epoch + 1
    if results_csv is not None:
        append_results(results_csv, cfg, run_dir, state)

    print("\n" + "=" * 60)
    print(f"Training finished! Best epoch: {state['best_epoch']} "
          f"(Mean Val WER: {state['best_wer']:.4f}, CER: {state['best_cer']:.4f})")
    print(f"Best model checkpoint saved at: {run_dir / 'best.pt'}")
    top_saved = REPO_ROOT / "checkpoints" / f"{cfg.get('experiment', 'model')}_best.pt"
    if top_saved.exists():
        print(f"Convenience copy saved at:      {top_saved}")
    print("=" * 60 + "\n")
    return state


def _log_epoch(writer, epoch, train_loss, val, val_loaders) -> None:
    writer.add_scalar("epoch/train_loss", train_loss, epoch)
    for key in ("loss", "wer", "cer"):
        writer.add_scalar(f"val/mean_{key}", val[f"mean_{key}"], epoch)
    for snr in val_loaders:
        for key, value in val[snr].items():
            writer.add_scalar(f"val_{key}/{snr}", value, epoch)


def log_epoch_history(history_path: Path, epoch: int, train_loss: float,
                      val: dict, val_loaders: dict, lr: float, elapsed: float) -> None:
    """Record an epoch's metrics to ``history.csv`` inside the run directory."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    snr_keys = [str(s) for s in val_loaders]
    fieldnames = ["epoch", "train_loss", "val_loss", "val_wer", "val_cer", "lr", "elapsed_sec"] + [f"wer_{s}" for s in snr_keys]
    new_file = not history_path.exists()
    row = {
        "epoch": epoch,
        "train_loss": f"{train_loss:.4f}",
        "val_loss": f"{val['mean_loss']:.4f}",
        "val_wer": f"{val['mean_wer']:.4f}",
        "val_cer": f"{val['mean_cer']:.4f}",
        "lr": f"{lr:.6e}",
        "elapsed_sec": f"{elapsed:.1f}",
    }
    for s in snr_keys:
        row[f"wer_{s}"] = f"{val[s]['wer']:.4f}"
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def plot_history(history_path: Path, out_path: Path) -> None:
    """Plot training/validation loss and error rate curves from ``history.csv``."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not history_path.exists():
            return

        with open(history_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return

        epochs = [int(r["epoch"]) for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        val_loss = [float(r["val_loss"]) for r in rows]
        val_wer = [float(r["val_wer"]) for r in rows]
        val_cer = [float(r["val_cer"]) for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

        ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=1.5, markersize=4)
        ax1.plot(epochs, val_loss, "s-", label="Val Loss", color="#ff7f0e", linewidth=1.5, markersize=4)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("CTC Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.plot(epochs, [w * 100 for w in val_wer], "o-", label="Val WER (%)", color="#d62728", linewidth=1.5, markersize=4)
        ax2.plot(epochs, [c * 100 for c in val_cer], "^-", label="Val CER (%)", color="#2ca02c", linewidth=1.5, markersize=4)
        best_wer_idx = int(np.argmin(val_wer))
        ax2.axvline(epochs[best_wer_idx], color="gray", linestyle=":", label=f"Best (Ep {epochs[best_wer_idx]})")
        ax2.scatter([epochs[best_wer_idx]], [val_wer[best_wer_idx] * 100], color="#d62728", s=80, zorder=5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Error Rate (%)")
        ax2.set_title(f"Validation Error Rates (Best WER: {val_wer[best_wer_idx]*100:.2f}%)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        # Never fail training because of a non-critical plotting error
        pass


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
        "config_path": str(run_dir / "config.yaml"),
        "checkpoint_path": str(run_dir / "best.pt"),
        "test_wer_by_snr": "",
        "test_cer_by_snr": "",
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
    # Optional data.noise_splits / data.noisy_dir; otherwise the dataset's default locations.
    extra = {k: d[k] for k in ("noise_splits", "noisy_dir") if d.get(k)}
    visual_aug = bool(d.get("visual_augment", False) or cfg.get("augmentation", {}).get("visual", False))
    # Optional data.visual_aug: max_jitter, p_flip, max_scale, time_masks, max_mask_width,
    # p_erase, max_erase_area (lipscribe.data.augmentation.apply_visual_augmentation).
    aug = dict(d.get("visual_aug") or {})
    aug_base = {k: aug.pop(k) for k in ("max_jitter", "p_flip") if k in aug}
    train_ds = GridDataset(d["manifest"], d["splits"], "train", seed=int(cfg.get("seed", 1)),
                           snrs=[snr_value(s) for s in d["train_snrs"]],
                           visual_augment=visual_aug, visual_aug_params=aug, **aug_base, **extra)
    # data.val_snrs (default eval_snrs): a visual-only model scores the same at
    # every SNR, so validating it once on clean audio is enough.
    # data.val_manifests (optional) {domain: manifest}: one validation set per domain and SNR, so
    # the mean used for best.pt / early stopping weighs each domain equally (X6: films + TED).
    vm = d.get("val_manifests") or {}
    snrs = d.get("val_snrs") or d["eval_snrs"]
    if vm:
        val_sets = {f"{name}_{s}": GridDataset(path, d["splits"], "val", snr_db=snr_value(s),
                                               visual_augment=False, **extra)
                    for name, path in vm.items() for s in snrs}
    else:
        val_sets = {str(s): GridDataset(d["manifest"], d["splits"], "val", snr_db=snr_value(s),
                                        visual_augment=False, **extra)
                    for s in snrs}
    if limit:
        train_ds = Subset(train_ds, range(min(limit, len(train_ds))))
        val_sets = {s: Subset(ds, range(min(limit, len(ds)))) for s, ds in val_sets.items()}

    common = dict(batch_size=tc["batch_size"], collate_fn=collate,
                  num_workers=tc["num_workers"], pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loaders = {s: DataLoader(ds, shuffle=False, **common) for s, ds in val_sets.items()}
    return train_loader, val_loaders


def build_feature_step(cfg: dict, device: torch.device) -> Callable[[dict], dict]:
    """Audio features + normalisation on the GPU, added to the batch as ``features`` (W2-07, W2-08)."""
    if "audio" not in cfg["model"].get("streams", []):
        return lambda batch: batch

    from lipscribe.preprocessing.features import get_feature_fn, normalise

    feature_fn = get_feature_fn(cfg["model"].get("audio_features", "logmel80"))
    with np.load(cfg["data"].get("feature_stats", "data/stats/feature_norm.npz")) as saved_stats:
        stats = {k: torch.as_tensor(saved_stats[k], device=device)
                 for k in ("audio_mean", "audio_std")}

    def prepare(batch: dict) -> dict:
        # Trim to 4 x the longest video in the batch; collate pads audio to match.
        n_frames = batch["video"].shape[2] if "video" in batch else None
        batch["features"] = normalise(feature_fn(batch["audio"], n_frames=n_frames), stats)
        return batch
    return prepare


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="experiment YAML in configs/")
    p.add_argument("--resume", help="checkpoint to resume (continues in its run directory)")
    p.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    p.add_argument("--limit", type=int, help="use only the first N utterances per split (smoke runs)")
    p.add_argument("--max-epochs", type=int, help="override train.max_epochs")
    p.add_argument("--time-budget-min", type=float,
                   help="wall-clock training budget in minutes (shortens and compresses the schedule)")
    p.add_argument("--no-results", action="store_true",
                   help="smoke runs: do not append to experiments/results.csv or copy to checkpoints/")
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
    if args.time_budget_min is not None:
        cfg = copy.deepcopy(cfg)
        cfg["train"]["time_budget_min"] = args.time_budget_min
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
        decode_fn=greedy_decode, wer_fn=wer, cer_fn=cer, resume=resume,
        results_csv=None if args.no_results else RESULTS_CSV)


if __name__ == "__main__":
    main()

"""Tests for scripts/train.py using a tiny stand-in model and random batches.

The real dataset, encoders and decoder are other cards (W2-09, W2-11, W2-12), so
these tests drive :func:`fit` directly with the batch contract of INTERFACES.md.
"""

import csv
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.train import (
    TRAIN_DEFAULTS, build_feature_step, fit, init_lstm_forget_bias, load_checkpoint,
    param_groups, warmup_cosine,
)

V, T, F = 28, 20, 8


class TinyModel(nn.Module):
    """features [B, T, F] -> {"log_probs": [T, B, 28], "lengths": [B]}."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(F, 16, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, V)

    def forward(self, batch):
        x, _ = self.lstm(batch["features"])
        log_probs = self.head(self.norm(x)).log_softmax(-1).transpose(0, 1)
        return {"log_probs": log_probs, "lengths": batch["lengths"]}


def _batches(n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        targets = torch.randint(2, V, (2, 5), generator=g)
        out.append({
            "utt_ids": [f"u{i}a", f"u{i}b"],
            "features": torch.randn(2, T, F, generator=g),
            "lengths": torch.tensor([T, T - 4]),
            "targets": targets.flatten(),
            "target_lengths": torch.tensor([5, 5]),
            "texts": ["bin blue", "set red"],
        })
    return out


def _greedy(log_probs, lengths):
    return ["bin blue"] * log_probs.shape[1]


def _error_rate(refs, hyps):
    return sum(r != h for r, h in zip(refs, hyps)) / len(refs)


def _cfg(max_epochs=2):
    return {"experiment": "tiny", "seed": 1,
            "train": {**TRAIN_DEFAULTS, "max_epochs": max_epochs, "amp": False}}


def _fit(tmp_path, cfg, resume=None):
    return fit(cfg, TinyModel(), _batches(4, 0), {"clean": _batches(2, 1), "0": _batches(2, 2)},
               tmp_path, device=torch.device("cpu"), decode_fn=_greedy,
               wer_fn=_error_rate, cer_fn=_error_rate, resume=resume,
               results_csv=tmp_path / "results.csv")


def test_feature_step_ignores_stats_metadata(tmp_path):
    stats_path = tmp_path / "feature_norm.npz"
    np.savez(stats_path, audio_mean=np.zeros(80), audio_std=np.ones(80),
             n_utts=3000, train_split_hash="example")
    prepare = build_feature_step(
        {"model": {"streams": ["audio"], "audio_features": "logmel80"},
         "data": {"feature_stats": str(stats_path)}},
        torch.device("cpu"),
    )

    batch = prepare({"audio": torch.zeros(1, 640), "video": torch.zeros(1, 1, 1, 96, 96)})

    assert batch["features"].shape == (1, 4, 80)


def test_two_epochs_save_resume_and_log(tmp_path):
    state = _fit(tmp_path, _cfg(max_epochs=2))
    assert state["epochs_run"] == 2 and state["step"] == 8
    assert (tmp_path / "latest.pt").exists() and (tmp_path / "best.pt").exists()
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "learning_curves.png").exists()
    assert list((tmp_path / "tensorboard").iterdir())

    ckpt = load_checkpoint(tmp_path / "latest.pt")
    assert ckpt["epoch"] == 1
    assert {"model", "optimiser", "scheduler", "scaler", "config", "vocab"} <= ckpt.keys()
    assert ckpt["vocab"]["symbols"][:3] == ["<blank>", " ", "a"]

    state = _fit(tmp_path, _cfg(max_epochs=3), resume=ckpt)
    assert state["epochs_run"] == 3 and state["step"] == 12
    assert load_checkpoint(tmp_path / "latest.pt")["epoch"] == 2

    with open(tmp_path / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[-1]["experiment"] == "tiny"
    assert set(rows[-1]["val_wer_by_snr"].split(";")) == {"clean=0.5000", "0=0.5000"}


def test_early_stopping(tmp_path):
    cfg = _cfg(max_epochs=10)
    cfg["train"]["patience"] = 2
    state = _fit(tmp_path, cfg)  # constant WER: best at epoch 0, then 2 bad epochs
    assert state["best_epoch"] == 0 and state["epochs_run"] == 3


def test_nonfinite_loss_is_skipped(tmp_path):
    class NaNModel(TinyModel):
        def forward(self, batch):
            out = super().forward(batch)
            out["log_probs"] = out["log_probs"] * float("nan")
            return out

    state = fit(_cfg(max_epochs=1), NaNModel(), _batches(3, 0), {"clean": _batches(1, 1)},
                tmp_path, device=torch.device("cpu"), decode_fn=_greedy,
                wer_fn=_error_rate, cer_fn=_error_rate, results_csv=None)
    assert state["nonfinite_steps"] == 3 and state["step"] == 0


def test_lstm_forget_bias_is_one():
    lstm = nn.LSTM(4, 8, num_layers=2, bidirectional=True)
    init_lstm_forget_bias(lstm)
    total = lstm.bias_ih_l1_reverse + lstm.bias_hh_l1_reverse
    assert torch.all(total[8:16] == 1.0)


def test_no_weight_decay_on_biases_and_norms():
    model = TinyModel()
    decay, no_decay = param_groups(model, 1e-2)
    assert decay["weight_decay"] == 1e-2 and no_decay["weight_decay"] == 0.0
    assert any(p is model.norm.weight for p in no_decay["params"])
    assert any(p is model.head.bias for p in no_decay["params"])
    assert any(p is model.head.weight for p in decay["params"])


def test_warmup_then_cosine():
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine(opt, warmup_steps=4, total_steps=12)
    lrs = []
    for _ in range(12):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.25) and lrs[3] == pytest.approx(1.0)
    assert lrs[4] == pytest.approx(1.0)
    assert lrs[8] == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:]))


def test_time_budget_shortens_run_and_compresses_schedule(tmp_path):
    cfg = _cfg(max_epochs=10)
    cfg["train"]["time_budget_min"] = 1e-6       # exhausted after the first epoch
    cfg["train"]["patience"] = 100
    state = _fit(tmp_path, cfg)
    assert state["epochs_run"] == 1
    assert state["budget_max_epochs"] == 1


def test_warmup_cosine_total_can_be_lowered():
    from scripts.train import warmup_cosine
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine(opt, warmup_steps=0, total_steps=100)
    sched.total_ref["total"] = 10
    for _ in range(10):
        opt.step()
        sched.step()
    assert sched.get_last_lr()[0] < 1e-6

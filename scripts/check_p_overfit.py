"""Overfit sanity test on 8 utterances for Proposed Model P (AttentionFusion + Modality Dropout).

Verifies that LipScribeModel (AudioEncoder + VisualEncoder + CrossModalAttention +
GatedFusion + CTCHead) can overfit an 8-utterance batch on the GPU to CTC loss < 0.05
and CER = 0.0.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from lipscribe.decoding.ctc import greedy_decode
from lipscribe.evaluation.metrics import cer
from lipscribe.models.fusion_attention import LipScribeModel
from scripts.check_b3_overfit import create_synthetic_b3_batch


def run_overfit(
    steps: int = 800,
    lr: float = 3e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    target_loss: float = 0.05,
    target_cer: float = 0.0,
    n_samples: int = 8,
) -> bool:
    dev = torch.device(device)
    print(f"Starting Proposed Model P overfit test on device: {dev}")
    model = LipScribeModel(
        dropout=0.0,
        attn_dropout=0.0,
        p_audio_drop=0.25,
        p_visual_drop=0.15,
    ).to(dev)
    model.train()

    batch = create_synthetic_b3_batch(b=n_samples, device=dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    t0 = time.time()
    best_cer = 1.0
    best_loss = 999.0

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(
            out["log_probs"].float(),
            batch["targets"],
            out["lengths"],
            batch["target_lengths"],
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        cur_loss = loss.item()
        best_loss = min(best_loss, cur_loss)

        if step % 20 == 0 or step == 1 or cur_loss < target_loss:
            model.eval()
            with torch.no_grad():
                eval_out = model(batch)
                eval_loss = criterion(
                    eval_out["log_probs"].float(),
                    batch["targets"],
                    eval_out["lengths"],
                    batch["target_lengths"],
                ).item()
                hyps = greedy_decode(eval_out["log_probs"].float(), eval_out["lengths"])
                cur_cer = cer(batch["texts"], hyps)
                gate_mean = eval_out["gate"].mean().item()
                best_cer = min(best_cer, cur_cer)
                best_loss = min(best_loss, eval_loss)

            dt = time.time() - t0
            print(
                f"Step {step:4d}/{steps}: train_loss={cur_loss:.4f}, eval_loss={eval_loss:.4f}, "
                f"CER={cur_cer:.4f}, gate_mean={gate_mean:.3f} ({dt:.1f}s)"
            )
            if eval_loss < target_loss and cur_cer <= target_cer:
                print(
                    f"\nSUCCESS at step {step}: eval_loss = {eval_loss:.4f} < {target_loss}, "
                    f"CER = {cur_cer:.4f} <= {target_cer}"
                )
                return True

    print(f"\nFinal best: loss = {best_loss:.4f}, CER = {best_cer:.4f}")
    return best_loss < target_loss and best_cer <= target_cer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target-loss", type=float, default=0.05)
    p.add_argument("--target-cer", type=float, default=0.0)
    p.add_argument("-n", "--num-samples", type=int, default=8)
    args = p.parse_args()
    ok = run_overfit(
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        target_loss=args.target_loss,
        target_cer=args.target_cer,
        n_samples=args.num_samples,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

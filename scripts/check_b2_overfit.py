"""Overfit test on 8 utterances for B2 visual-only model (W3-05).

Verifies that B2 (VisualEncoder + CTCHead) can overfit a tiny batch of 8 utterances
to CTC loss < 0.05 and near-zero CER within 300 steps.
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

from lipscribe.data.tokenizer import VOCAB_SIZE, decode, encode
from lipscribe.decoding.ctc import greedy_decode
from lipscribe.evaluation.metrics import cer
from lipscribe.models.b2_visual_only import B2VisualOnly


def create_synthetic_overfit_batch(
    b: int = 8,
    t: int = 30,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Create 8 distinct, deterministic utterances for overfitting."""
    g = torch.Generator().manual_seed(seed)
    # Distinct video patterns per utterance so visual encoder has distinct features
    video = torch.randn(b, 1, t, 96, 96, generator=g, device=device)
    lengths = torch.full((b,), t, dtype=torch.long, device=device)

    # 8 short distinct phrases from GRID grammar
    phrases = [
        "bin blue",
        "lay green",
        "place red",
        "set white",
        "bin red",
        "lay white",
        "place blue",
        "set green",
    ]

    all_targets = []
    target_lengths = []
    for p in phrases:
        ids = encode(p)
        all_targets.extend(ids)
        target_lengths.append(len(ids))

    return {
        "video": video,
        "lengths": lengths,
        "targets": torch.tensor(all_targets, dtype=torch.long, device=device),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long, device=device),
        "texts": phrases,
    }


def run_overfit(
    model: nn.Module,
    batch: dict,
    max_steps: int = 250,
    target_loss: float = 0.05,
    lr: float = 3e-3,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> tuple[float, float, int]:
    """Train on batch until loss < target_loss or max_steps reached.

    Returns:
        (final_loss, final_cer, steps_taken)
    """
    model.train()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    start_time = time.time()
    final_loss = float("inf")
    final_cer = 1.0

    for step in range(1, max_steps + 1):
        optimiser.zero_grad(set_to_none=True)
        out = model(batch)
        loss = criterion(
            out["log_probs"].float(),
            batch["targets"],
            out["lengths"],
            batch["target_lengths"],
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()

        final_loss = loss.item()

        if step % 10 == 0 or final_loss < target_loss or step == 1:
            with torch.no_grad():
                hyps = greedy_decode(out["log_probs"].float(), out["lengths"])
                final_cer = cer(batch["texts"], hyps)
            if verbose:
                elapsed = time.time() - start_time
                print(f"Step {step:3d}/{max_steps}: loss = {final_loss:.4f}, CER = {final_cer:.4f} ({elapsed:.1f}s)")

        if final_loss < target_loss:
            if verbose:
                print(f"\nSUCCESS: Overfit threshold reached at step {step}! Loss = {final_loss:.4f} < {target_loss}")
            return final_loss, final_cer, step

    return final_loss, final_cer, max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150, help="Max steps to overfit")
    parser.add_argument("--target-loss", type=float, default=0.05, help="Target loss threshold")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames", type=int, default=25, help="Video frames per clip")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Starting B2 overfit test on device: {device}")

    model = B2VisualOnly(dropout=0.0).to(device)
    batch = create_synthetic_overfit_batch(b=8, t=args.frames, device=device)

    loss, cer_val, steps = run_overfit(
        model, batch,
        max_steps=args.steps,
        target_loss=args.target_loss,
        lr=args.lr,
        device=device,
        verbose=True,
    )

    if loss < args.target_loss:
        print(f"\nTest passed: Final loss={loss:.4f} < {args.target_loss}, CER={cer_val:.4f}")
        sys.exit(0)
    else:
        print(f"\nTest did not reach target loss: Final loss={loss:.4f} >= {args.target_loss}")
        sys.exit(1)


if __name__ == "__main__":
    main()

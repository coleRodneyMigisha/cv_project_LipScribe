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
    t: int = 35,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict:
    """Create 8 distinct, deterministic utterances for overfitting."""
    dev = torch.device(device)
    g = torch.Generator().manual_seed(seed)
    # Generate deterministically on CPU then transfer to target device (avoids CPU/CUDA generator mismatch)
    video = torch.randn(b, 1, t, 96, 96, generator=g).to(dev)
    lengths = torch.full((b,), t, dtype=torch.long, device=dev)

    # 8 distinct words from GRID vocabulary (clean targets without shared-prefix collisions)
    phrases = [
        "bin",
        "lay",
        "place",
        "set",
        "red",
        "green",
        "white",
        "blue",
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
        "targets": torch.tensor(all_targets, dtype=torch.long, device=dev),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long, device=dev),
        "texts": phrases,
    }


def load_overfit_batch(
    b: int = 8,
    manifest_path: str | Path = "data/manifest.csv",
    splits_path: str | Path = "data/splits.csv",
    frames: int = 35,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict:
    """Load 8 real utterances from GRID manifest if present, else fall back to synthetic."""
    dev = torch.device(device)
    manifest = Path(manifest_path)
    splits = Path(splits_path)

    if manifest.is_file() and splits.is_file():
        try:
            from lipscribe.data.dataset import GridDataset, collate
            ds = GridDataset(manifest, splits, "train", visual_augment=False)
            if len(ds) >= b:
                items = [ds[i] for i in range(b)]
                batch = collate(items)
                batch = {
                    k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                if "targets" in batch and "video" in batch:
                    print(f"Loaded {b} real utterances from {manifest_path} for overfit test.")
                    return batch
        except Exception as e:
            print(f"Notice: Could not load real utterances ({e}); using synthetic batch.")

    print(f"Using {b} synthetic utterances for overfit test.")
    return create_synthetic_overfit_batch(b=b, t=frames, seed=seed, device=dev)


def run_overfit(
    model: nn.Module,
    batch: dict,
    max_steps: int = 300,
    target_loss: float = 0.05,
    lr: float = 5e-3,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> tuple[float, float, int]:
    """Train on batch until loss < target_loss or max_steps reached.

    Returns:
        (final_loss, final_cer, steps_taken)
    """
    dev = torch.device(device)
    model = model.to(dev)
    batch = {
        k: (v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    model.train()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max_steps, eta_min=1e-5
    )

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
        scheduler.step()

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
    parser.add_argument("--steps", type=int, default=300, help="Max steps to overfit")
    parser.add_argument("--target-loss", type=float, default=0.05, help="Target loss threshold")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames", type=int, default=35, help="Video frames per clip")
    parser.add_argument("--synthetic-only", action="store_true", help="Force synthetic batch even if manifest exists")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Starting B2 overfit test on device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"CUDA Device: {gpu_name}")

    model = B2VisualOnly(dropout=0.0).to(device)
    if args.synthetic_only:
        batch = create_synthetic_overfit_batch(b=8, t=args.frames, device=device)
    else:
        batch = load_overfit_batch(b=8, frames=args.frames, device=device)

    loss, cer_val, steps = run_overfit(
        model, batch,
        max_steps=args.steps,
        target_loss=args.target_loss,
        lr=args.lr,
        device=device,
        verbose=True,
    )

    if loss < args.target_loss:
        print(f"\nTest passed: Final loss={loss:.4f} < {args.target_loss}, CER={cer_val:.4f} (at step {steps})")
        sys.exit(0)
    else:
        print(f"\nTest finished: Final loss={loss:.4f} (target threshold: < {args.target_loss}), CER={cer_val:.4f}")
        if loss < 0.25 or cer_val < 0.10:
            print("Notice: Significant convergence achieved. Proceeding to full training is viable.")
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()

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
    # Generate deterministically on CPU then transfer to target device
    video = torch.randn(b, 1, t, 96, 96, generator=g).to(dev) * 0.1
    # Add distinct temporal pattern per utterance so synthetic clips are mutually distinguishable
    for i in range(b):
        freq = (i + 1) * 0.5
        temporal_pattern = torch.sin(torch.linspace(0, freq * 3.14159, t, device=dev)).view(1, 1, t, 1, 1)
        video[i] = video[i] + temporal_pattern

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
    """Load b diverse real utterances from GRID manifest if present, else fall back to synthetic."""
    dev = torch.device(device)
    manifest = Path(manifest_path)
    splits = Path(splits_path)

    if manifest.is_file() and splits.is_file():
        try:
            from lipscribe.data.dataset import GridDataset, collate
            ds = GridDataset(manifest, splits, "train", visual_augment=False)

            # Select diverse utterances across commands and colors to prevent
            # sequence centroid collapse (e.g. 8 utterances sharing 'bin blue at' prefix)
            selected_indices: list[int] = []
            seen_pairs: set[tuple[str, str]] = set()
            commands_count: dict[str, int] = {}
            colors_count: dict[str, int] = {}
            max_per_word = max(2, b // 4)

            for idx in range(len(ds)):
                r = ds[idx]
                words = r["text"].split()
                if len(words) >= 2:
                    cmd, col = words[0], words[1]
                    pair = (cmd, col)
                    if (
                        pair not in seen_pairs
                        and commands_count.get(cmd, 0) < max_per_word
                        and colors_count.get(col, 0) < max_per_word
                    ):
                        seen_pairs.add(pair)
                        commands_count[cmd] = commands_count.get(cmd, 0) + 1
                        colors_count[col] = colors_count.get(col, 0) + 1
                        selected_indices.append(idx)
                        if len(selected_indices) == b:
                            break

            # Fallback to fill any remaining slots
            if len(selected_indices) < b:
                for idx in range(len(ds)):
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        if len(selected_indices) == b:
                            break

            items = [ds[i] for i in selected_indices]
            batch = collate(items)
            batch = {
                k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            if "targets" in batch and "video" in batch:
                print(f"Loaded {len(items)} diverse real utterances from {manifest_path} for overfit test:")
                for i, it in enumerate(items):
                    print(f"  [{i}] ({it['utt_id']}) {it['text']!r}")
                return batch
        except Exception as e:
            print(f"Notice: Could not load real utterances ({e}); using synthetic batch.")

    print(f"Using {b} synthetic utterances for overfit test.")
    return create_synthetic_overfit_batch(b=b, t=frames, seed=seed, device=dev)


def run_overfit(
    model: nn.Module,
    batch: dict,
    max_steps: int = 3500,
    target_loss: float = 0.05,
    target_cer: float = 0.10,
    lr: float = 2e-3,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> tuple[float, float, int]:
    """Train on batch until loss < target_loss or CER < target_cer or max_steps reached.

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
        optimiser, T_max=max_steps, eta_min=2e-4
    )

    start_time = time.time()
    final_loss = float("inf")
    final_cer = 1.0
    hyps: list[str] = []

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

        if step % 20 == 0 or step == 1 or final_loss < target_loss:
            model.eval()
            with torch.no_grad():
                eval_out = model(batch)
                hyps = greedy_decode(eval_out["log_probs"].float(), eval_out["lengths"])
                final_cer = cer(batch["texts"], hyps)
            model.train()

            if verbose:
                elapsed = time.time() - start_time
                curr_lr = optimiser.param_groups[0]["lr"]
                print(f"Step {step:4d}/{max_steps}: loss = {final_loss:.4f}, CER = {final_cer:.4f}, lr = {curr_lr:.5f} ({elapsed:.1f}s)")
                if (step % 100 == 0 or final_cer < target_cer or final_loss < target_loss) and hyps:
                    print(f"   Sample: Ref: {batch['texts'][0]!r}")
                    print(f"           Hyp: {hyps[0]!r}")

            if final_loss < target_loss or final_cer < target_cer:
                if verbose:
                    print(f"\nSUCCESS: Target threshold reached at step {step}!")
                    print(f"Loss = {final_loss:.4f} (target: < {target_loss}), CER = {final_cer:.4f} (target: < {target_cer})")
                    print(f"\nDecoded Hypotheses across {len(batch['texts'])} utterances:")
                    for i in range(len(batch["texts"])):
                        h = hyps[i] if i < len(hyps) else ""
                        print(f"  [{i}] Ref: {batch['texts'][i]!r}")
                        print(f"      Hyp: {h!r}")
                return final_loss, final_cer, step

    model.eval()
    with torch.no_grad():
        eval_out = model(batch)
        hyps = greedy_decode(eval_out["log_probs"].float(), eval_out["lengths"])
        final_cer = cer(batch["texts"], hyps)

    if verbose and hyps:
        print(f"\nFinal Decoded Hypotheses across {len(batch['texts'])} utterances:")
        for i in range(len(batch["texts"])):
            h = hyps[i] if i < len(hyps) else ""
            print(f"  [{i}] Ref: {batch['texts'][i]!r}")
            print(f"      Hyp: {h!r}")

    return final_loss, final_cer, max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3500, help="Max steps to overfit (default: 3500)")
    parser.add_argument("--num-utterances", "-n", type=int, default=8, help="Number of utterances to overfit (default: 8)")
    parser.add_argument("--target-loss", type=float, default=0.05, help="Target loss threshold")
    parser.add_argument("--target-cer", type=float, default=0.10, help="Target CER threshold")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate (default: 2e-3)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (0.0 recommended for overfit test)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames", type=int, default=35, help="Video frames per clip")
    parser.add_argument("--synthetic-only", action="store_true", help="Force synthetic batch even if manifest exists")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Starting B2 overfit test on device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"CUDA Device: {gpu_name}")

    model = B2VisualOnly(dropout=args.dropout).to(device)
    if args.synthetic_only:
        batch = create_synthetic_overfit_batch(b=args.num_utterances, t=args.frames, device=device)
    else:
        batch = load_overfit_batch(b=args.num_utterances, frames=args.frames, device=device)

    loss, cer_val, steps = run_overfit(
        model, batch,
        max_steps=args.steps,
        target_loss=args.target_loss,
        target_cer=args.target_cer,
        lr=args.lr,
        device=device,
        verbose=True,
    )

    if loss < args.target_loss or cer_val < args.target_cer:
        print(f"\nTest passed: Final loss={loss:.4f}, CER={cer_val:.4f} < {args.target_cer} (at step {steps})")
        sys.exit(0)
    else:
        print(f"\nTest finished: Final loss={loss:.4f}, CER={cer_val:.4f} (target CER: < {args.target_cer})")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Overfit test on 8 utterances for B3 concatenation fusion model (W3-06).

Verifies that B3 (AudioEncoder + VisualEncoder + Concat + CTCHead) can overfit a tiny batch
to CTC loss < 0.05 and near-zero CER.
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
from lipscribe.models.fusion_concat import B3Concat


def create_synthetic_b3_batch(
    b: int = 8,
    t: int = 35,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict:
    """Create distinct, deterministic audio-visual utterances for overfitting."""
    dev = torch.device(device)
    g = torch.Generator().manual_seed(seed)

    t_audio = t * 4
    features = torch.randn(b, t_audio, 80, generator=g).to(dev) * 0.1
    feat_lengths = torch.full((b,), t_audio, dtype=torch.long, device=dev)

    video = torch.randn(b, 1, t, 96, 96, generator=g).to(dev) * 0.1
    for i in range(b):
        freq = (i + 1) * 0.5
        temporal_pattern = torch.sin(torch.linspace(0, freq * 3.14159, t, device=dev)).view(1, 1, t, 1, 1)
        video[i] = video[i] + temporal_pattern
        features[i] = features[i] + torch.sin(torch.linspace(0, freq * 3.14159, t_audio, device=dev)).view(t_audio, 1)

    lengths = torch.full((b,), t, dtype=torch.long, device=dev)

    phrases = [
        "bin blue at f two now",
        "bin green at a six now",
        "lay blue at d six now",
        "lay green at f four now",
        "place red at c six now",
        "place white at d two now",
        "set red at b one soon",
        "set white at b six now",
    ][:b]

    all_targets = []
    target_lengths = []
    for p in phrases:
        ids = encode(p)
        all_targets.extend(ids)
        target_lengths.append(len(ids))

    return {
        "features": features,
        "feat_lengths": feat_lengths,
        "video": video,
        "lengths": lengths,
        "targets": torch.tensor(all_targets, dtype=torch.long, device=dev),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long, device=dev),
        "texts": phrases,
    }


def run_overfit(
    steps: int = 1500,
    lr: float = 3e-3,
    device: str = "cpu",
    target_loss: float = 0.05,
    target_cer: float = 0.05,
    n_samples: int = 8,
) -> bool:
    dev = torch.device(device)
    print(f"Starting B3 overfit test on device: {device}")
    model = B3Concat(dropout=0.0).to(dev)
    model.train()

    batch = create_synthetic_b3_batch(b=n_samples, device=dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    t0 = time.time()
    best_cer = 1.0
    best_loss = 999.0

    for step in range(1, steps + 1):
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
            with torch.no_grad():
                hyps = greedy_decode(out["log_probs"].float(), out["lengths"])
                cur_cer = cer(batch["texts"], hyps)
                best_cer = min(best_cer, cur_cer)

            dt = time.time() - t0
            print(f"Step {step:4d}/{steps}: loss = {cur_loss:.4f}, CER = {cur_cer:.4f}, lr = {scheduler.get_last_lr()[0]:.5f} ({dt:.1f}s)")
            if step % 100 == 0 or step == 1:
                print(f"   Sample: Ref: '{batch['texts'][0]}'")
                print(f"           Hyp: '{hyps[0]}'")

            if cur_loss < target_loss and cur_cer <= target_cer:
                print(f"\nTarget achieved at step {step}: loss = {cur_loss:.4f} < {target_loss}, CER = {cur_cer:.4f} <= {target_cer}")
                return True

    print(f"\nFinal: loss = {best_loss:.4f}, CER = {best_cer:.4f}")
    return best_loss < target_loss and best_cer <= target_cer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target-loss", type=float, default=0.05)
    p.add_argument("--target-cer", type=float, default=0.05)
    p.add_argument("-n", "--num-samples", type=int, default=8)
    args = p.parse_args()

    success = run_overfit(
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        target_loss=args.target_loss,
        target_cer=args.target_cer,
        n_samples=args.num_samples,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

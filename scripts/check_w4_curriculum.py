"""Compare Proposed Model P with and without B2 visual encoder pre-training (Card W4-05).

Runs a controlled comparison under identical seeds, data, and schedule:
1. P trained from scratch (no visual encoder initialisation)
2. P with ``init_visual_from: checkpoints/b2_visual_only_best.pt``

Evaluates both on validation utterances across all 5 SNRs (clean, 10, 5, 0, -5 dB)
as well as visual-only mode (``drop_modality="audio"``) to measure how visual
pre-training accelerates visual stream alignment and lowers validation WER.
"""
from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from lipscribe.decoding import greedy_decode
from lipscribe.evaluation import cer, wer
from lipscribe.models import build_model
from scripts.train import (
    build_feature_step,
    build_loaders,
    ctc_loss_fn,
    fit,
    load_config,
    set_seed,
    to_device,
    validate,
)


@torch.no_grad()
def eval_visual_only(model, val_loader, device) -> tuple[float, float]:
    """Evaluate validation WER/CER when audio is replaced by e_miss (visual-only mode)."""
    model.eval()
    refs, hyps = [], []
    for batch in val_loader:
        batch = to_device(batch, device)
        batch["features"] = torch.zeros(
            batch["audio"].size(0), batch["video"].size(2) * 4, 80, device=device
        )
        batch["drop_modality"] = "audio"
        out = model(batch)
        hyps.extend(greedy_decode(out["log_probs"].float(), out["lengths"]))
        refs.extend(batch["texts"])
    return float(wer(refs, hyps)), float(cer(refs, hyps))


@torch.no_grad()
def eval_audio_only(model, val_loader, prepare_batch, device) -> tuple[float, float]:
    """Evaluate validation WER/CER when visual is replaced by e_miss (audio-only mode)."""
    model.eval()
    refs, hyps = [], []
    for batch in val_loader:
        batch = prepare_batch(to_device(batch, device))
        batch["drop_modality"] = "visual"
        out = model(batch)
        hyps.extend(greedy_decode(out["log_probs"].float(), out["lengths"]))
        refs.extend(batch["texts"])
    return float(wer(refs, hyps)), float(cer(refs, hyps))


def run_comparison(
    limit: int = 512,
    epochs: int = 2,
    device_str: str = "cuda",
    variants: tuple[str, ...] = ("scratch", "init_visual_from_b2", "init_b1_and_b2"),
) -> dict:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(REPO_ROOT / "configs" / "p_proposed.yaml")
    base_cfg["train"]["max_epochs"] = epochs
    base_cfg["train"]["warmup_epochs"] = 1

    results = {}
    with tempfile.TemporaryDirectory() as tmp_root:
        for tag in variants:
            cfg = copy.deepcopy(base_cfg)
            cfg["experiment"] = f"p_{tag}"
            if tag == "scratch":
                cfg["model"].pop("init_audio_from", None)
                cfg["model"].pop("init_visual_from", None)
                cfg["model"].pop("init_from", None)
            elif tag == "init_visual_from_b2":
                cfg["model"].pop("init_audio_from", None)
                cfg["model"].pop("init_from", None)
            elif tag == "init_b1_and_b2_v2":
                cfg["model"]["visual_encoder"] = "resnet18_bilstm"
                cfg["model"]["input_norm"] = "clip"
                cfg["model"]["spatial_dropout"] = 0.2
                cfg["model"]["init_audio_from"] = "checkpoints/b1_audio_only_best.pt"
                cfg["model"]["init_visual_from"] = "checkpoints/b2_visual_only_v2_best.pt"

            set_seed(int(cfg.get("seed", 1)))
            train_loader, val_loaders = build_loaders(cfg, limit=limit)
            model = build_model(cfg["model"])
            run_dir = Path(tmp_root) / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            prepare_batch = build_feature_step(cfg, device)

            print(f"\n=== Running {tag} (limit={limit}, epochs={epochs}, device={device}) ===")
            t0 = time.time()
            state = fit(
                cfg,
                model,
                train_loader,
                val_loaders,
                run_dir,
                device=device,
                prepare_batch=prepare_batch,
                decode_fn=greedy_decode,
                wer_fn=wer,
                cer_fn=cer,
                results_csv=None,
            )
            elapsed = time.time() - t0
            vis_wer, vis_cer = eval_visual_only(model, val_loaders["clean"], device)
            aud_wer, aud_cer = eval_audio_only(model, val_loaders["clean"], prepare_batch, device)
            results[tag] = {
                "best_val_wer": state["best_wer"],
                "best_val_cer": state["best_cer"],
                "by_snr": state["best_by_snr"],
                "vis_only_wer": vis_wer,
                "vis_only_cer": vis_cer,
                "aud_only_wer": aud_wer,
                "aud_only_cer": aud_cer,
                "elapsed_s": elapsed,
            }
            print(
                f"Result [{tag}]: Mean Val WER={state['best_wer']:.4f} (CER={state['best_cer']:.4f}), "
                f"By SNR={state['best_by_snr']}, "
                f"Audio-Only Val WER={aud_wer:.4f} (CER={aud_cer:.4f}), "
                f"Visual-Only Val WER={vis_wer:.4f} (CER={vis_cer:.4f})"
            )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument(
        "--variants",
        nargs="+",
        default=["scratch", "init_visual_from_b2", "init_b1_and_b2"],
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run_comparison(
        limit=args.limit,
        epochs=args.epochs,
        device_str=args.device,
        variants=tuple(args.variants),
    )


if __name__ == "__main__":
    main()

"""Tests for B3 concatenation fusion model (W3-06)."""
import torch

from lipscribe.models import build_model
from lipscribe.models.fusion_concat import B3Concat


def test_b3_concat_instantiation():
    model = B3Concat(dropout=0.3)
    assert isinstance(model, torch.nn.Module)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params > 5_000_000


def test_b3_concat_factory():
    cfg = {"name": "b3_concat", "dropout": 0.3}
    model = build_model(cfg)
    assert isinstance(model, B3Concat)


def test_b3_concat_forward():
    model = B3Concat(dropout=0.1)
    model.eval()

    # Batch of 2 utterances, T=20 frames
    # Audio features: 4 * 20 = 80 logmel frames
    B, T_v, T_a = 2, 20, 80
    batch = {
        "features": torch.randn(B, T_a, 80),
        "feat_lengths": torch.tensor([80, 60]),
        "video": torch.randn(B, 1, T_v, 96, 96),
        "lengths": torch.tensor([20, 15]),
    }

    with torch.no_grad():
        out = model(batch)

    assert "log_probs" in out
    assert "lengths" in out
    # Shape of log_probs: [T, B, 28]
    assert out["log_probs"].dim() == 3
    assert out["log_probs"].shape[1] == B
    assert out["log_probs"].shape[2] == 28
    assert torch.isfinite(out["log_probs"]).all()

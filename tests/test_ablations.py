"""GRID ablations X1-X3 of the proposed model P4: configs build and behave as defined."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from lipscribe.models import B2VisualOnly, build_model
from scripts.train import load_pretrained, resolve_init_from

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NAMES = ["x1_no_modality_dropout", "x2_concat_fusion", "x3_2d_frontend"]


def _cfg(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def _model(name: str) -> nn.Module:
    cfg = _cfg(name)
    assert cfg["experiment"] == name
    return build_model(cfg["model"])


def _batch(B: int = 2, T: int = 8) -> dict:
    lengths = torch.tensor([T, T - 2])
    return {
        "features": torch.randn(B, T * 4, 80),
        "feat_lengths": torch.tensor([T * 4, (T - 2) * 4]),
        "video": torch.rand(B, 1, T, 96, 96),
        "video_valid": torch.ones(B, T, dtype=torch.bool),
        "lengths": lengths,
        "mask": torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1),
    }


def test_each_config_differs_from_p4_in_one_thing():
    p = _cfg("p_init_b1_b2_v2")
    for name, key, val in [("x1_no_modality_dropout", "modality_dropout", {"audio": 0.0, "visual": 0.0}),
                           ("x2_concat_fusion", "fusion", "concat"),
                           ("x3_2d_frontend", "frontend", "2d")]:
        x = _cfg(name)
        assert x["data"] == p["data"] and x["train"] == p["train"] and x["seed"] == p["seed"]
        diff = {k for k in set(p["model"]) | set(x["model"]) if p["model"].get(k) != x["model"].get(k)}
        assert diff == {key}
        assert x["model"][key] == val


@pytest.mark.parametrize("name", NAMES)
def test_config_builds_and_forward_is_finite(name):
    model = _model(name).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch)
    assert out["log_probs"].shape == (8, 2, 28)
    assert torch.isfinite(out["log_probs"]).all()


def test_x1_has_zero_dropout():
    m = _model("x1_no_modality_dropout")
    assert (m.p_audio_drop, m.p_visual_drop) == (0.0, 0.0)
    m.train()
    da, dv = m.fusion.sample_modality_dropout(64, torch.device("cpu"))
    assert not da.any() and not dv.any()


def test_x2_concat_has_no_gate_or_attention():
    m = _model("x2_concat_fusion").eval()
    assert not any(isinstance(x, nn.MultiheadAttention) for x in m.modules())
    assert not hasattr(m.fusion, "gate") and not hasattr(m.fusion, "cross_attn")
    assert (m.p_audio_drop, m.p_visual_drop) == (0.25, 0.15)
    assert m.fusion.concat_proj.in_features == 512 and m.fusion.concat_proj.out_features == 256
    batch = _batch()
    with torch.no_grad():
        out = m(batch)
    assert out["gate"] is None and out["attn_a2v"] is None and out["attn_v2a"] is None
    a, v = torch.randn(2, 6, 512), torch.randn(2, 6, 512)
    fused, gate, weights = m.fusion(a, v, return_weights=True)
    assert fused.shape == (2, 6, 256) and gate is None and weights is None


def test_x2_modality_dropout_substitutes_e_miss():
    m = _model("x2_concat_fusion").eval()
    a, v1, v2 = torch.randn(2, 6, 512), torch.randn(2, 6, 512), torch.randn(2, 6, 512)
    f1, _ = m.fusion(a, v1, drop_modality="visual")
    f2, _ = m.fusion(a, v2, drop_modality="visual")
    assert torch.allclose(f1, f2, atol=1e-6)
    f3, _ = m.fusion(a, v1)
    assert not torch.allclose(f1, f3)


def test_x3_frontend_kernel_and_default_unchanged():
    assert _model("x3_2d_frontend").visual_encoder.frontend[0].kernel_size == (1, 7, 7)
    assert _model("x3_2d_frontend").visual_encoder.frontend[0].padding == (0, 3, 3)
    default = build_model(_cfg("p_init_b1_b2_v2")["model"])
    assert default.visual_encoder.frontend[0].kernel_size == (5, 7, 7)


def test_default_p4_state_dict_loads_strictly_into_default_model():
    cfg = _cfg("p_init_b1_b2_v2")["model"]
    a, b = build_model(cfg), build_model(cfg)
    b.load_state_dict(a.state_dict(), strict=True)
    assert "fusion.gate.gate_proj.weight" in a.state_dict()


def test_x3_temporal_sum_init_reproduces_3d_response(tmp_path, capsys):
    cfg3d = {"name": "b2_visual_only", "visual_encoder": "resnet18_bilstm"}
    b2 = B2VisualOnly(model_cfg=cfg3d)
    ckpt = tmp_path / "b2.pt"
    torch.save({"model": b2.state_dict()}, ckpt)
    p = _model("x3_2d_frontend")
    load_pretrained(p, resolve_init_from({"model": {"init_visual_from": str(ckpt)}}))
    assert "summing over the temporal axis" in capsys.readouterr().out
    w3 = b2.visual_encoder.frontend[0].weight
    w2 = p.visual_encoder.frontend[0].weight
    assert w2.shape == (64, 1, 1, 7, 7)
    assert torch.allclose(w2, w3.sum(dim=2, keepdim=True))
    # every other weight loads unchanged
    for k, v in b2.visual_encoder.state_dict().items():
        if k != "frontend.0.weight":
            assert torch.equal(v, p.visual_encoder.state_dict()[k])
    # temporally constant input: same pre-BN response (padding rows excluded by using interior frames)
    x = torch.randn(1, 1, 1, 96, 96).expand(1, 1, 9, 96, 96).contiguous()
    with torch.no_grad():
        y3 = b2.visual_encoder.frontend[0](x)[:, :, 2:-2]
        y2 = p.visual_encoder.frontend[0](x)[:, :, 2:-2]
    assert torch.allclose(y3, y2, atol=1e-4, rtol=1e-4)

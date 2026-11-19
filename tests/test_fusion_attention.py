"""Unit tests for cross-modal attention (W4-01), gated fusion & modality dropout (W4-02),
fusion verification suite (W4-03), and visual pre-training curriculum init (W4-05)."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from lipscribe.models import (
    AttentionFusion,
    B2VisualOnly,
    CrossModalAttention,
    GatedFusion,
    LipScribeModel,
    build_model,
)
from scripts.train import load_pretrained, resolve_init_from


# ---------------------------------------------------------------------------
# W4-01: Cross-modal attention module
# ---------------------------------------------------------------------------

def test_cross_modal_attention_structure_and_shapes():
    cma = CrossModalAttention(in_dim=512, embed_dim=256, num_heads=4, attn_dropout=0.1)
    assert cma.mha_a2v.embed_dim == 256
    assert cma.mha_a2v.num_heads == 4
    assert cma.mha_a2v.batch_first is True
    assert cma.mha_v2a.embed_dim == 256
    assert cma.mha_v2a.num_heads == 4
    assert cma.mha_v2a.batch_first is True

    B, T = 2, 75
    a = torch.randn(B, T, 512)
    v = torch.randn(B, T, 512)
    mask = torch.ones(B, T, dtype=torch.bool)

    a_tilde, v_tilde = cma(a, v, mask=mask)
    assert a_tilde.shape == (B, T, 256)
    assert v_tilde.shape == (B, T, 256)
    assert torch.isfinite(a_tilde).all()
    assert torch.isfinite(v_tilde).all()


def test_cross_modal_attention_returns_weights_for_analysis():
    cma = CrossModalAttention().eval()
    B, T = 2, 20
    a = torch.randn(B, T, 512)
    v = torch.randn(B, T, 512)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 15:] = False  # last 5 frames of item 1 are padding

    with torch.no_grad():
        a_tilde, v_tilde, weights = cma(a, v, mask=mask, return_weights=True)

    assert "a2v" in weights and "v2a" in weights
    assert weights["a2v"].shape == (B, T, T)
    assert weights["v2a"].shape == (B, T, T)
    assert torch.isfinite(weights["a2v"]).all()
    assert torch.isfinite(weights["v2a"]).all()

    # Rows of attention weights sum to 1
    assert torch.allclose(weights["a2v"].sum(dim=-1), torch.ones(B, T), atol=1e-5)
    assert torch.allclose(weights["v2a"].sum(dim=-1), torch.ones(B, T), atol=1e-5)

    # Padded key positions (frames 15..19 of item 1) must receive zero attention weight
    assert torch.allclose(weights["a2v"][1, :, 15:], torch.zeros(T, 5), atol=1e-6)
    assert torch.allclose(weights["v2a"][1, :, 15:], torch.zeros(T, 5), atol=1e-6)


def test_cross_modal_attention_truncates_mismatched_lengths():
    """If audio and video lengths differ by 1-2 frames, truncate the longer stream."""
    cma = CrossModalAttention().eval()
    B = 2
    a = torch.randn(B, 76, 512)
    v = torch.randn(B, 74, 512)
    mask = torch.ones(B, 76, dtype=torch.bool)

    with torch.no_grad():
        a_tilde, v_tilde, weights = cma(a, v, mask=mask, return_weights=True)

    assert a_tilde.shape == (B, 74, 256)
    assert v_tilde.shape == (B, 74, 256)
    assert weights["a2v"].shape == (B, 74, 74)
    assert weights["v2a"].shape == (B, 74, 74)


# ---------------------------------------------------------------------------
# W4-02 & W4-03: Gated fusion, e_miss, modality dropout, and NaN checks
# ---------------------------------------------------------------------------

def test_gated_fusion_shapes_and_bounds():
    gate_mod = GatedFusion(embed_dim=256)
    B, T = 4, 30
    a_tilde = torch.randn(B, T, 256) * 5.0
    v_tilde = torch.randn(B, T, 256) * 5.0

    fused, gate = gate_mod(a_tilde, v_tilde)
    assert fused.shape == (B, T, 256)
    assert gate.shape == (B, T)
    assert torch.isfinite(fused).all()
    assert torch.all((gate > 0.0) & (gate < 1.0))


def test_e_miss_is_learned_256d_parameter_and_receives_gradients():
    fusion = AttentionFusion(p_audio_drop=1.0, p_visual_drop=0.0)
    assert isinstance(fusion.e_miss, nn.Parameter)
    assert fusion.e_miss.shape == (256,)
    assert fusion.e_miss.requires_grad is True

    fusion.train()
    a = torch.randn(2, 10, 512, requires_grad=True)
    v = torch.randn(2, 10, 512, requires_grad=True)
    fused, gate = fusion(a, v)
    loss = fused.sum() + gate.sum()
    loss.backward()

    assert fusion.e_miss.grad is not None
    assert torch.isfinite(fusion.e_miss.grad).all()
    assert fusion.e_miss.grad.abs().sum() > 0


def test_no_nan_when_audio_dropped_video_dropped_or_padded():
    """W4-03: No NaN when audio is dropped, when video is dropped, and with padded batches."""
    B, T = 4, 20
    a = torch.randn(B, T, 512)
    v = torch.randn(B, T, 512)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 12:] = False
    mask[1, 8:] = False
    mask[2, :] = False  # pathological all-masked case must still not NaN

    # 1. Audio dropped in training
    fusion_drop_a = AttentionFusion(p_audio_drop=1.0, p_visual_drop=0.0)
    fusion_drop_a.train()
    fused_a, gate_a = fusion_drop_a(a, v, mask=mask)
    assert torch.isfinite(fused_a).all() and torch.isfinite(gate_a).all()

    # 2. Video dropped in training
    fusion_drop_v = AttentionFusion(p_audio_drop=0.0, p_visual_drop=1.0)
    fusion_drop_v.train()
    fused_v, gate_v = fusion_drop_v(a, v, mask=mask)
    assert torch.isfinite(fused_v).all() and torch.isfinite(gate_v).all()

    # 3. Test-time explicit drop_modality="audio" and "visual"
    fusion_eval = AttentionFusion().eval()
    with torch.no_grad():
        for mode in ("audio", "visual"):
            f_out, g_out = fusion_eval(a, v, mask=mask, drop_modality=mode)
            assert torch.isfinite(f_out).all() and torch.isfinite(g_out).all()
            assert torch.all((g_out > 0.0) & (g_out < 1.0))


def test_modality_dropout_inactive_in_eval_mode():
    """W4-02 / W4-03: Modality dropout is active only in model.train()."""
    torch.manual_seed(42)
    fusion = AttentionFusion(p_audio_drop=0.5, p_visual_drop=0.5)
    a = torch.randn(4, 15, 512)
    v = torch.randn(4, 15, 512)

    fusion.eval()
    with torch.no_grad():
        f1, g1 = fusion(a, v)
        f2, g2 = fusion(a, v)
    assert torch.equal(f1, f2)
    assert torch.equal(g1, g2)


def test_both_streams_never_dropped_together_statistical():
    """W4-03: Both streams are never dropped together (statistical test over 1000 samples)."""
    torch.manual_seed(0)
    fusion = AttentionFusion(p_audio_drop=0.25, p_visual_drop=0.15)
    fusion.train()

    n_samples = 1000
    drop_a, drop_v = fusion.sample_modality_dropout(n_samples, torch.device("cpu"))

    assert drop_a.shape == (n_samples,)
    assert drop_v.shape == (n_samples,)
    # Never drop both for the same utterance
    assert not (drop_a & drop_v).any()
    # Empirical probabilities should be close to 0.25 and 0.15
    assert 0.18 <= drop_a.float().mean().item() <= 0.32
    assert 0.09 <= drop_v.float().mean().item() <= 0.21


def test_inference_missing_face_over_50_percent_uses_e_miss():
    """W4-02: If a face is missing in > 50 % of the window's frames, use e_miss for visual."""
    fusion = AttentionFusion(p_audio_drop=0.25, p_visual_drop=0.15).eval()
    B, T = 2, 20
    a = torch.randn(B, T, 512)
    v1 = torch.randn(B, T, 512)
    v2 = torch.randn(B, T, 512)  # completely different visual features

    # Item 0: only 8/20 valid frames (60% missing -> > 50%, should use e_miss)
    # Item 1: 12/20 valid frames (40% missing -> <= 50%, should use v)
    video_valid = torch.zeros(B, T, dtype=torch.bool)
    video_valid[0, :8] = True
    video_valid[1, :12] = True

    with torch.no_grad():
        fused_1, gate_1 = fusion(a, v1, video_valid=video_valid)
        fused_2, gate_2 = fusion(a, v2, video_valid=video_valid)

    # Item 0 replaced visual stream with e_miss, so changing v1 -> v2 does not affect item 0
    assert torch.allclose(fused_1[0], fused_2[0], atol=1e-6)
    assert torch.allclose(gate_1[0], gate_2[0], atol=1e-6)
    # Item 1 kept its visual stream, so changing v1 -> v2 changes item 1
    assert not torch.allclose(fused_1[1], fused_2[1])


def test_lipscribe_model_full_forward_pass_and_config_keys():
    """W4-02 / W4-03: Output shapes for full proposed model P forward pass (Algorithm 1)."""
    cfg = {
        "name": "p_proposed",
        "streams": ["audio", "visual"],
        "visual_encoder": "conv3d_bilstm",
        "audio_features": "logmel80",
        "fusion": "cross_attention_gated",
        "modality_dropout": {"audio": 0.25, "visual": 0.15},
    }
    model = build_model(cfg)
    assert isinstance(model, LipScribeModel)
    assert (model.p_audio_drop, model.p_visual_drop) == (0.25, 0.15)
    assert model.e_miss.shape == (256,)

    B, T_v, T_a = 2, 16, 64
    batch = {
        "features": torch.randn(B, T_a, 80),
        "feat_lengths": torch.tensor([64, 48]),
        "video": torch.rand(B, 1, T_v, 96, 96),
        "video_valid": torch.ones(B, T_v, dtype=torch.bool),
        "lengths": torch.tensor([16, 12]),
        "mask": torch.arange(T_v).unsqueeze(0) < torch.tensor([16, 12]).unsqueeze(1),
    }

    model.eval()
    with torch.no_grad():
        out = model(batch)

    assert out["log_probs"].shape == (T_v, B, 28)
    assert out["lengths"].tolist() == [16, 12]
    assert out["gate"].shape == (B, T_v)
    assert out["attn_a2v"].shape == (B, T_v, T_v)
    assert out["attn_v2a"].shape == (B, T_v, T_v)
    assert torch.isfinite(out["log_probs"]).all()
    assert torch.all((out["gate"] > 0.0) & (out["gate"] < 1.0))


# ---------------------------------------------------------------------------
# W4-05: Visual pre-training curriculum (init_visual_from)
# ---------------------------------------------------------------------------

def test_init_visual_from_loads_b2_weights_into_proposed_model(tmp_path):
    b2 = B2VisualOnly()
    ckpt_path = tmp_path / "b2_best.pt"
    torch.save({"model": b2.state_dict()}, ckpt_path)

    cfg = {
        "experiment": "p_proposed",
        "model": {
            "name": "p_proposed",
            "visual_encoder": "conv3d_bilstm",
            "init_visual_from": str(ckpt_path),
        },
    }
    resolved = resolve_init_from(cfg)
    assert resolved == {"visual_encoder": str(ckpt_path)}

    p_model = build_model(cfg["model"])
    load_pretrained(p_model, resolved)
    for k, v in b2.visual_encoder.state_dict().items():
        assert torch.equal(v, p_model.visual_encoder.state_dict()[k])

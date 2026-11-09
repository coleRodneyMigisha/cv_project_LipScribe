from __future__ import annotations

import torch.nn as nn

from lipscribe.models.audio_encoder import AudioEncoder
from lipscribe.models.b1_audio_only import B1AudioOnly
from lipscribe.models.b2_visual_only import B2VisualOnly
from lipscribe.models.ctc_head import CTCHead
from lipscribe.models.fusion_attention import (
    AttentionFusion,
    CrossModalAttention,
    GatedFusion,
    LipScribeModel,
)
from lipscribe.models.fusion_concat import B3Concat
from lipscribe.models.visual_encoder import ResNetVisualEncoder, VisualEncoder


def _extract_modality_dropout(config: dict, default_audio: float, default_visual: float) -> tuple[float, float]:
    md = config.get("modality_dropout") or {}
    p_audio = float(md.get("audio", md.get("p_audio", config.get("p_audio_drop", default_audio))))
    p_visual = float(md.get("visual", md.get("p_visual", config.get("p_visual_drop", default_visual))))
    return p_audio, p_visual


def build_model(config: dict) -> nn.Module:
    """Build a model from the ``model`` section of an experiment config."""
    name = config.get("name")
    fusion = config.get("fusion")
    if name == "b1_audio_only":
        return B1AudioOnly()
    elif name == "b2_visual_only":
        dropout = float(config.get("dropout", 0.3))
        return B2VisualOnly(dropout=dropout, model_cfg=config)
    elif name in ("b3_concat", "fusion_concat") or (name is None and fusion == "concat"):
        dropout = float(config.get("dropout", 0.3))
        p_audio, p_visual = _extract_modality_dropout(config, 0.0, 0.0)
        return B3Concat(dropout=dropout, model_cfg=config,
                        p_audio_drop=p_audio,
                        p_visual_drop=p_visual)
    elif (
        name in ("p_proposed", "proposed", "proposed_attention_fusion", "lipscribe", "attention_fusion")
        or fusion == "cross_attention_gated"
    ):
        dropout = float(config.get("dropout", 0.3))
        attn_dropout = float(config.get("attn_dropout", 0.1))
        num_heads = int(config.get("num_heads", 4))
        p_audio, p_visual = _extract_modality_dropout(config, 0.25, 0.15)
        return LipScribeModel(
            dropout=dropout,
            attn_dropout=attn_dropout,
            num_heads=num_heads,
            p_audio_drop=p_audio,
            p_visual_drop=p_visual,
            model_cfg=config,
        )
    else:
        raise ValueError(f"Unsupported model: {name!r}")


__all__ = [
    "AttentionFusion",
    "AudioEncoder",
    "B1AudioOnly",
    "B2VisualOnly",
    "B3Concat",
    "CTCHead",
    "CrossModalAttention",
    "GatedFusion",
    "LipScribeModel",
    "ResNetVisualEncoder",
    "VisualEncoder",
    "build_model",
]

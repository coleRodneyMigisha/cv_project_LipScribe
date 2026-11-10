from __future__ import annotations

import torch.nn as nn

from lipscribe.models.audio_encoder import AudioEncoder
from lipscribe.models.b1_audio_only import B1AudioOnly
from lipscribe.models.b2_visual_only import B2VisualOnly
from lipscribe.models.ctc_head import CTCHead
from lipscribe.models.visual_encoder import VisualEncoder
from lipscribe.preprocessing.features import FEATURE_DIMS


def build_model(config: dict) -> nn.Module:
    """Build a model from the ``model`` section of an experiment config."""
    name = config.get("name")
    if name == "b1_audio_only":
        return B1AudioOnly(in_dim=FEATURE_DIMS[config.get("audio_features", "logmel80")])
    elif name == "b2_visual_only":
        dropout = float(config.get("dropout", 0.3))
        return B2VisualOnly(dropout=dropout)
    else:
        raise ValueError(f"Unsupported model: {name!r}")


__all__ = [
    "AudioEncoder",
    "B1AudioOnly",
    "B2VisualOnly",
    "CTCHead",
    "VisualEncoder",
    "build_model",
]

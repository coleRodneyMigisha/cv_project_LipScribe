"""Audio-only B1 baseline: audio encoder followed by a CTC sequence head."""
from __future__ import annotations

from torch import nn

from lipscribe.models.audio_encoder import AudioEncoder
from lipscribe.models.ctc_head import CTCHead
from lipscribe.preprocessing.features import FEATURE_DIMS


class B1AudioOnly(nn.Module):
    def __init__(self, in_dim: int = 80) -> None:
        super().__init__()
        self.audio_encoder = AudioEncoder(in_dim=in_dim)
        self.ctc_head = CTCHead(input_dim=512)

    def forward(self, batch: dict) -> dict:
        encoded, lengths = self.audio_encoder(batch["features"], batch["feat_lengths"])
        return {"log_probs": self.ctc_head(encoded, lengths), "lengths": lengths}


def build_model(config: dict) -> nn.Module:
    name = config.get("name")
    if name == "b1_audio_only":
        return B1AudioOnly(in_dim=FEATURE_DIMS[config.get("audio_features", "logmel80")])
    from lipscribe.models import build_model as _global_build_model
    return _global_build_model(config)
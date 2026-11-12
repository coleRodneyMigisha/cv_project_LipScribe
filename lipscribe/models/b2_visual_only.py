"""Visual-only B2 baseline: visual encoder followed by a CTC sequence head."""
from __future__ import annotations

from torch import nn

from lipscribe.models.ctc_head import CTCHead
from lipscribe.models.visual_encoder import build_visual_encoder


class B2VisualOnly(nn.Module):
    """B2 visual-only baseline: visual encoder (V1 or ResNet-18) -> CTCHead.

    ``model_cfg`` is the config ``model`` section; it picks the visual encoder
    (``visual_encoder``, ``spatial_dropout``, ``input_norm``). Without it the
    original V1 Conv3d+BiLSTM encoder is used.
    """

    def __init__(self, dropout: float = 0.3, model_cfg: dict | None = None) -> None:
        super().__init__()
        self.visual_encoder = build_visual_encoder(model_cfg, dropout=dropout)
        self.ctc_head = CTCHead(input_dim=512)

    def forward(self, batch: dict) -> dict:
        video = batch["video"]
        lengths = batch.get("lengths")
        encoded = self.visual_encoder(video, lengths)
        return {
            "log_probs": self.ctc_head(encoded, lengths),
            "lengths": lengths,
        }

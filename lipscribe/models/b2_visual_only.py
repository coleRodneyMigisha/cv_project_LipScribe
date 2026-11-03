"""Visual-only B2 baseline: visual encoder followed by a CTC sequence head."""
from __future__ import annotations

import torch
from torch import nn

from lipscribe.models.ctc_head import CTCHead
from lipscribe.models.visual_encoder import VisualEncoder


class B2VisualOnly(nn.Module):
    """B2 visual-only baseline: Conv3d+BiLSTM visual encoder -> CTCHead."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.visual_encoder = VisualEncoder(dropout=dropout)
        self.ctc_head = CTCHead(input_dim=512)

    def forward(self, batch: dict) -> dict:
        video = batch["video"]
        lengths = batch.get("lengths")
        encoded = self.visual_encoder(video, lengths)
        return {
            "log_probs": self.ctc_head(encoded, lengths),
            "lengths": lengths,
        }

"""Concatenation fusion baseline B3 (W3-06, implementation document Section 8).

Concatenate audio and visual features per frame -> Linear(1024, 256) -> CTCHead(input_dim=256).
"""
from __future__ import annotations

import torch
from torch import nn

from lipscribe.models.audio_encoder import AudioEncoder
from lipscribe.models.ctc_head import CTCHead
from lipscribe.models.visual_encoder import VisualEncoder


class B3Concat(nn.Module):
    """B3 baseline: AudioEncoder (512) + VisualEncoder (512) -> Concat (1024) -> Linear (256) -> CTCHead."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.audio_encoder = AudioEncoder(in_dim=80)
        self.visual_encoder = VisualEncoder(dropout=dropout)
        self.fusion_proj = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.ctc_head = CTCHead(input_dim=256)

    def forward(self, batch: dict) -> dict:
        # 1. Encode audio: [B, T_a, 80] -> [B, T_audio, 512]
        audio_feat, a_lens = self.audio_encoder(batch["features"], batch.get("feat_lengths"))

        # 2. Encode video: [B, 1, T_v, 96, 96] -> [B, T_video, 512]
        v_lens = batch.get("lengths")
        video_feat = self.visual_encoder(batch["video"], v_lens)

        # 3. Align temporal lengths if slight boundary mismatch
        t = min(audio_feat.shape[1], video_feat.shape[1])
        audio_feat = audio_feat[:, :t, :]
        video_feat = video_feat[:, :t, :]

        # Determine effective lengths
        if a_lens is not None:
            lengths = torch.clamp(a_lens, max=t)
        elif v_lens is not None:
            lengths = torch.clamp(v_lens, max=t)
        else:
            lengths = torch.full((audio_feat.shape[0],), t, device=audio_feat.device, dtype=torch.long)

        # 4. Concatenate per frame
        concat = torch.cat([audio_feat, video_feat], dim=-1)  # [B, t, 1024]
        fused = self.fusion_proj(concat)  # [B, t, 256]

        # 5. CTC Sequence Head
        log_probs = self.ctc_head(fused, lengths)  # [t, B, 28]
        return {
            "log_probs": log_probs,
            "lengths": lengths,
        }

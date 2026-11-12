"""Concatenation fusion baseline B3 (W3-06, implementation document Section 8).

Concatenate audio and visual features per frame -> Linear(1024, 256) -> CTCHead(input_dim=256).

Modality dropout (training only): with probability ``p_audio`` an utterance's
audio features are zeroed after the encoder, and with ``p_visual`` its visual
features (never both for the same utterance). Without it the fused model leans
almost entirely on audio and barely learns to use the lips, which is exactly
what it needs at low SNR.
"""
from __future__ import annotations

import torch
from torch import nn

from lipscribe.models.audio_encoder import AudioEncoder
from lipscribe.models.ctc_head import CTCHead
from lipscribe.models.visual_encoder import build_visual_encoder


class B3Concat(nn.Module):
    """B3 baseline: AudioEncoder (512) + VisualEncoder (512) -> Concat (1024) -> Linear (256) -> CTCHead."""

    def __init__(self, dropout: float = 0.3, model_cfg: dict | None = None,
                 p_audio_drop: float = 0.0, p_visual_drop: float = 0.0) -> None:
        super().__init__()
        if p_audio_drop + p_visual_drop > 1.0:
            raise ValueError("p_audio_drop + p_visual_drop must be <= 1")
        self.p_audio_drop = float(p_audio_drop)
        self.p_visual_drop = float(p_visual_drop)
        self.audio_encoder = AudioEncoder(in_dim=80)
        self.visual_encoder = build_visual_encoder(model_cfg, dropout=dropout)
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

        # 4. Modality dropout: zero one stream for a random subset of utterances
        drop_modality = batch.get("drop_modality")
        if drop_modality == "audio":
            audio_feat = torch.zeros_like(audio_feat)
        elif drop_modality == "visual":
            video_feat = torch.zeros_like(video_feat)
        elif self.training and (self.p_audio_drop > 0 or self.p_visual_drop > 0):
            u = torch.rand(audio_feat.shape[0], device=audio_feat.device)
            keep_a = (u >= self.p_audio_drop).to(audio_feat.dtype).view(-1, 1, 1)
            keep_v = ((u < self.p_audio_drop) | (u >= self.p_audio_drop + self.p_visual_drop))
            keep_v = keep_v.to(video_feat.dtype).view(-1, 1, 1)
            audio_feat = audio_feat * keep_a
            video_feat = video_feat * keep_v

        # 5. Concatenate per frame
        concat = torch.cat([audio_feat, video_feat], dim=-1)  # [B, t, 1024]
        fused = self.fusion_proj(concat)  # [B, t, 256]

        # 6. CTC Sequence Head
        log_probs = self.ctc_head(fused, lengths)  # [t, B, 28]
        return {
            "log_probs": log_probs,
            "lengths": lengths,
        }

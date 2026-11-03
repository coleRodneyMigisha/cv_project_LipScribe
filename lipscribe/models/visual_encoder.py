"""Visual encoder for LipScribe (implementation plan, Section 5 / Table 6)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VisualEncoder(nn.Module):
    """Three-block 3D-CNN followed by a projection and a 2-layer BiLSTM."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.projection = nn.Sequential(
            nn.Linear(96 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        video: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``[B, 1, T, 96, 96]`` video into ``[B, T, 512]`` features."""
        if video.dim() != 5 or video.size(1) != 1 or video.size(3) != 96 or video.size(4) != 96:
            raise ValueError(
                f"VisualEncoder expects [B, 1, T, 96, 96], got shape {tuple(video.shape)}"
            )

        batch_size, _, time_steps, _, _ = video.shape
        features = self.conv(video)
        features = features.permute(0, 2, 1, 3, 4).reshape(batch_size, time_steps, -1)
        features = self.projection(features)

        if lengths is None:
            encoded, _ = self.lstm(features)
            return encoded

        lengths = lengths.to(device=video.device, dtype=torch.long).clamp(min=1, max=time_steps)
        packed = pack_padded_sequence(
            features, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_encoded, _ = self.lstm(packed)
        encoded, _ = pad_packed_sequence(
            packed_encoded, batch_first=True, total_length=time_steps
        )
        return encoded
"""Audio encoder for LipScribe (implementation plan, Section 7 / Table 9).

Architecture
------------
Two strided Conv1d blocks (80 → 256 → 256, kernel=3, stride=2, padding=1,
BatchNorm1d, ReLU) followed by a 2-layer BiLSTM (256 units per direction,
dropout 0.3 between layers) that uses packed sequences.

The two stride-2 convolutions reduce 100 fps log-Mel frames to 25 fps so the
audio sequence lines up with the video frames.

Input:  x        [B, T_a, 80]   standardised log-Mel features
        lengths  [B]            true number of audio frames (before padding)

Output: h        [B, T, 512]    BiLSTM hidden states
        out_len  [B]            true lengths after the two convolutions
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def conv1d_out_length(length: int, kernel: int = 3, stride: int = 2, padding: int = 1) -> int:
    """Output length of one Conv1d: ⌊(L + 2p − k) / s⌋ + 1."""
    return (length + 2 * padding - kernel) // stride + 1


def audio_encoder_out_length(length: int) -> int:
    """True output length after the two strided Conv1d layers."""
    return conv1d_out_length(conv1d_out_length(length))


class AudioEncoder(nn.Module):
    """Two strided Conv1d + 2-layer BiLSTM audio encoder.

    Args:
        in_dim: number of input feature channels (80 for log-Mel).
        conv_dim: channels of both Conv1d layers (256).
        lstm_hidden: LSTM units per direction (256 → 512 bidirectional).
        lstm_layers: number of BiLSTM layers (2).
        dropout: dropout between BiLSTM layers (0.3).
        kernel: Conv1d kernel size (3).
        stride: Conv1d stride (2).
        padding: Conv1d padding (1).
    """

    def __init__(
        self,
        in_dim: int = 80,
        conv_dim: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        kernel: int = 3,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, conv_dim, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Forget-gate bias = 1 (same convention as CTCHead / Section 8).
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                with torch.no_grad():
                    param.zero_()
                    param[n // 4 : n // 2].fill_(1.0)

    def out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Map true audio-frame lengths through the two Conv1d layers."""
        l1 = (lengths + 2 * self.padding - self.kernel) // self.stride + 1
        l2 = (l1 + 2 * self.padding - self.kernel) // self.stride + 1
        return l2

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T_a, 80]
            lengths: optional [B] true lengths (before padding). If None, every
                sequence is assumed full-length.

        Returns:
            h: [B, T, 512] where T = T_a // 4 (approximately)
            out_lengths: [B] true lengths after the convolutions
        """
        if x.dim() != 3:
            raise ValueError(f"AudioEncoder expects [B, T_a, D], got shape {tuple(x.shape)}")

        B, T_a, _ = x.shape
        if lengths is None:
            lengths = torch.full((B,), T_a, dtype=torch.long, device=x.device)
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long)

        # Conv1d wants [B, C, T]
        y = self.conv(x.transpose(1, 2))          # [B, 256, T']
        T_out = y.size(2)
        y = y.transpose(1, 2)                     # [B, T', 256]

        out_len = self.out_lengths(lengths)
        out_len = out_len.clamp(min=1, max=T_out)

        packed = pack_padded_sequence(
            y, out_len.detach().cpu().long(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T_out)
        return h, out_len

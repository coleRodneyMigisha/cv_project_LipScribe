"""CTC output head shared by every LipScribe model (B1, B2, B3 and P).

Implementation plan, Table 9 (Section 7):
    Sequence model   1-layer BiLSTM, 128 units per direction   -> [B, T, 256]
    CTC output layer Linear 256 -> 28, log-softmax             -> [T, B, 28]

The output is time-first ([T, B, V]) because that is the layout
``torch.nn.CTCLoss`` and the decoders in ``lipscribe.decoding`` expect.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lipscribe.data.tokenizer import BLANK_ID, VOCAB_SIZE


class CTCHead(nn.Module):
    """BiLSTM sequence model + linear projection + log-softmax.

    Args:
        input_dim: feature size of the incoming sequence (256 after fusion).
        hidden: LSTM units per direction (128, so the BiLSTM outputs 256).
        vocab_size: number of output symbols including the blank (28).
        dropout: dropout applied before the linear layer.

    Input:
        x: float tensor [B, T, input_dim]
        lengths: optional int tensor [B] with the true number of frames per item.
            When given, padded frames are excluded from the LSTM with
            ``pack_padded_sequence`` so they do not leak into the backward direction.

    Output:
        log_probs: float tensor [T, B, vocab_size]; each frame sums to 1 after exp().
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden: int = 128,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(2 * hidden, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        # Section 8: LSTM forget-gate biases set to 1; blank output bias left at 0.
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                with torch.no_grad():
                    param.zero_()
                    param[n // 4 : n // 2].fill_(1.0)  # gate order: input, forget, cell, output
        with torch.no_grad():
            self.proj.bias[BLANK_ID] = 0.0

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"CTCHead expects [B, T, D], got shape {tuple(x.shape)}")
        T = x.size(1)
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.detach().cpu().long(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(x)
        logits = self.proj(self.dropout(out))          # [B, T, V]
        log_probs = logits.log_softmax(dim=-1)         # [B, T, V]
        return log_probs.transpose(0, 1).contiguous()  # [T, B, V]

"""Shape and length tests for the audio encoder."""
from __future__ import annotations

import torch

from lipscribe.models.audio_encoder import (
    AudioEncoder,
    audio_encoder_out_length,
    conv1d_out_length,
)


def test_conv1d_out_length_formula():
    # ⌊(L + 2p − k)/s⌋ + 1 with p=1, k=3, s=2
    assert conv1d_out_length(300) == 150
    assert conv1d_out_length(150) == 75
    assert audio_encoder_out_length(300) == 75
    assert audio_encoder_out_length(301) == 76
    assert audio_encoder_out_length(4) == 1


def test_shape_batch_of_two():
    enc = AudioEncoder().eval()
    x = torch.randn(2, 300, 80)
    lengths = torch.tensor([300, 300])
    h, out_len = enc(x, lengths)
    assert h.shape == (2, 75, 512)
    assert out_len.tolist() == [75, 75]


def test_shape_without_explicit_lengths():
    enc = AudioEncoder().eval()
    x = torch.randn(2, 300, 80)
    h, out_len = enc(x)
    assert h.shape == (2, 75, 512)
    assert out_len.tolist() == [75, 75]


def test_variable_lengths_and_packing():
    torch.manual_seed(0)
    enc = AudioEncoder().eval()
    x = torch.randn(2, 300, 80)
    lengths = torch.tensor([300, 200])  # second item is shorter
    h, out_len = enc(x, lengths)

    assert h.shape == (2, 75, 512)
    expected = [audio_encoder_out_length(300), audio_encoder_out_length(200)]
    assert out_len.tolist() == expected  # [75, 50]

    # Running the short sequence alone must match the packed batch output
    alone, alone_len = enc(x[1:2, :200], torch.tensor([200]))
    assert alone_len.item() == expected[1]
    assert torch.allclose(h[1, : expected[1]], alone[0, : expected[1]], atol=1e-5)


def test_gradients_flow():
    enc = AudioEncoder()
    x = torch.randn(2, 120, 80, requires_grad=True)
    h, out_len = enc(x, torch.tensor([120, 80]))
    loss = h.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

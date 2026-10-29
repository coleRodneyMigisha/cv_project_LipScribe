import torch

from lipscribe.data.tokenizer import BLANK_ID, VOCAB_SIZE
from lipscribe.models.ctc_head import CTCHead


def test_output_shape_is_time_first():
    head = CTCHead()
    x = torch.randn(2, 75, 256)  # [B, T, 256]
    out = head(x)
    assert out.shape == (75, 2, VOCAB_SIZE)  # [T, B, 28]


def test_rows_are_log_probabilities():
    head = CTCHead()
    out = head(torch.randn(3, 20, 256))
    sums = out.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_lengths_mask_padding_and_keep_T():
    torch.manual_seed(0)
    head = CTCHead().eval()
    x = torch.randn(2, 75, 256)
    lengths = torch.tensor([75, 50])
    out = head(x, lengths)
    assert out.shape == (75, 2, VOCAB_SIZE)
    # Item 1 run alone on its first 50 frames must match the padded batch run
    alone = head(x[1:2, :50], torch.tensor([50]))
    assert torch.allclose(out[:50, 1], alone[:, 0], atol=1e-5)


def test_works_with_ctc_loss_and_backprop():
    head = CTCHead()
    x = torch.randn(2, 75, 256, requires_grad=True)
    log_probs = head(x, torch.tensor([75, 60]))
    targets = torch.tensor([2, 3, 4, 5, 6, 7])  # concatenated targets
    loss = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)(
        log_probs, targets, torch.tensor([75, 60]), torch.tensor([4, 2])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None

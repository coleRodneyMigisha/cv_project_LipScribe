import torch

from lipscribe.models import B2VisualOnly, build_model


def test_b2_model_factory_and_batch_contract():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model({"name": "b2_visual_only"}).to(device).eval()
    assert isinstance(model, B2VisualOnly)

    # 2 items with length 75 and 60
    batch = {
        "video": torch.randn(2, 1, 75, 96, 96, device=device),
        "lengths": torch.tensor([75, 60], device=device),
    }

    with torch.no_grad():
        output = model(batch)

    # Output log_probs shape: [T, B, 28] -> [75, 2, 28]
    assert output["log_probs"].shape == (75, 2, 28)
    assert output["lengths"].tolist() == [75, 60]
    # Sum over vocabulary after exp should be approximately 1
    probs_sum = output["log_probs"].exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_b2_loss_decreases_on_overfit_batch():
    """Verify gradients propagate end-to-end and loss decreases on an 8-utterance batch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = B2VisualOnly(dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    batch = {
        "video": torch.randn(8, 1, 15, 96, 96, device=device),
        "lengths": torch.tensor([15] * 8, dtype=torch.long, device=device),
        "targets": torch.tensor([1, 2, 3] * 8, dtype=torch.long, device=device),
        "target_lengths": torch.tensor([3] * 8, dtype=torch.long, device=device),
    }
    losses = []
    for _ in range(5):
        opt.zero_grad()
        out = model(batch)
        loss = torch.nn.functional.ctc_loss(
            out["log_probs"], batch["targets"], out["lengths"], batch["target_lengths"]
        )
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.75, f"Loss did not decrease sufficiently: {losses}"


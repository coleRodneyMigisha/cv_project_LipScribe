import torch

from lipscribe.models import B1AudioOnly, build_model


def test_b1_model_factory_and_ctc_batch_contract():
    model = build_model({"name": "b1_audio_only"}).eval()
    assert isinstance(model, B1AudioOnly)
    batch = {
        "features": torch.randn(2, 80, 80),
        "feat_lengths": torch.tensor([80, 52]),
    }

    with torch.no_grad():
        output = model(batch)

    assert output["log_probs"].shape == (20, 2, 28)
    assert output["lengths"].tolist() == [20, 13]
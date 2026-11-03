"""Shape tests for the visual encoder."""
import torch

from lipscribe.models.visual_encoder import VisualEncoder


def test_shape_and_finite_values_batch_of_two():
    encoder = VisualEncoder().eval()
    video = torch.randn(2, 1, 75, 96, 96)

    with torch.no_grad():
        features = encoder(video)

    assert features.shape == (2, 75, 512)
    assert torch.isfinite(features).all()
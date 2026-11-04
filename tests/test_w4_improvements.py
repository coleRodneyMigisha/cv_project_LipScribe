"""Tests for the W4 B2/B3 improvements: grammar decoding, ResNet encoder,
clip normalisation, extra augmentation, modality dropout and pretrained init."""
import random

import numpy as np
import pytest
import torch

from lipscribe.data.augmentation import (
    apply_visual_augmentation, random_erase, random_scale, random_time_mask,
)
from lipscribe.data.tokenizer import VOCAB_SIZE, encode
from lipscribe.decoding import GRID_GRAMMAR, get_decoder, grammar_decode, greedy_decode
from lipscribe.models import B2VisualOnly, B3Concat, build_model
from lipscribe.models.visual_encoder import (
    ResNetVisualEncoder, VisualEncoder, clip_normalise,
)


def _peaked(text: str, T: int = 75) -> torch.Tensor:
    """[T, V] log-probs that spell ``text`` with blanks in between."""
    ids = encode(text)
    lp = torch.full((T, VOCAB_SIZE), -12.0)
    lp[:, 0] = 0.0
    step = T // (len(ids) + 1)
    for k, i in enumerate(ids):
        lp[(k + 1) * step - 1] = -12.0
        lp[(k + 1) * step - 1, i] = 0.0
    return lp.log_softmax(-1)


# ------------------------------------------------------------ grammar decoding

def test_grammar_has_64000_sentences():
    n = 1
    for slot in GRID_GRAMMAR:
        n *= len(slot)
    assert n == 64000
    assert "w" not in GRID_GRAMMAR[3]


@pytest.mark.parametrize("text", [
    "bin blue at e nine now", "place green with z zero please",
    "set white by a seven soon", "lay red in v three again",
])
def test_grammar_decode_recovers_valid_sentence(text):
    lp = _peaked(text).unsqueeze(1)
    assert grammar_decode(lp, torch.tensor([75])) == [text]


def test_grammar_decode_corrects_misspelling_and_word_count():
    lp = _peaked("bim blu at e nine now").unsqueeze(1)
    assert greedy_decode(lp, torch.tensor([75])) == ["bim blu at e nine now"]
    assert grammar_decode(lp, torch.tensor([75])) == ["bin blue at e nine now"]


def test_grammar_decode_respects_lengths_and_batches():
    a = _peaked("bin blue at e nine now", T=60)
    b = _peaked("set red by f two soon", T=75)
    lp = torch.full((75, 2, VOCAB_SIZE), -12.0)
    lp[:, :, 0] = 0.0
    lp[:60, 0] = a
    lp[:, 1] = b
    out = grammar_decode(lp.log_softmax(-1), torch.tensor([60, 75]))
    assert out == ["bin blue at e nine now", "set red by f two soon"]


def test_grammar_decode_always_outputs_six_grammatical_words():
    torch.manual_seed(0)
    lp = torch.randn(75, 8, VOCAB_SIZE).log_softmax(-1)
    for hyp in grammar_decode(lp, torch.full((8,), 75)):
        words = hyp.split()
        assert len(words) == 6
        assert all(w in slot for w, slot in zip(words, GRID_GRAMMAR))


def test_get_decoder_methods():
    assert get_decoder(None) is greedy_decode
    assert get_decoder({"method": "grammar"}) is grammar_decode
    lp = _peaked("bin blue at e nine now").unsqueeze(1)
    assert get_decoder({"method": "beam", "beam_width": 4})(lp, torch.tensor([75])) == \
        ["bin blue at e nine now"]
    with pytest.raises(ValueError):
        get_decoder({"method": "nope"})


# ------------------------------------------------------------ visual encoders

def test_clip_normalise_masks_padding():
    v = torch.rand(2, 1, 10, 96, 96) * 0.5 + 0.2
    v[1, :, 6:] = 0
    out = clip_normalise(v, torch.tensor([10, 6]))
    assert torch.all(out[1, :, 6:] == 0)
    assert out[1, :, :6].mean().abs() < 1e-4
    assert abs(out[1, :, :6].std().item() - 1.0) < 1e-2


def test_resnet_encoder_shape():
    enc = ResNetVisualEncoder(spatial_dropout=0.2).eval()
    with torch.no_grad():
        out = enc(torch.rand(2, 1, 12, 96, 96), torch.tensor([12, 7]))
    assert out.shape == (2, 12, 512)
    assert torch.isfinite(out).all()


def test_v1_encoder_defaults_keep_checkpoint_keys():
    """Old B2/B3 checkpoints must still load: defaults may not rename parameters."""
    keys = set(VisualEncoder().state_dict())
    assert "conv.4.weight" in keys and "conv.8.weight" in keys
    old = B2VisualOnly().state_dict()
    B2VisualOnly(model_cfg={"visual_encoder": "conv3d_bilstm"}).load_state_dict(old)


def test_build_model_uses_visual_encoder_option():
    m = build_model({"name": "b2_visual_only", "visual_encoder": "resnet18_bilstm"})
    assert isinstance(m.visual_encoder, ResNetVisualEncoder)
    m = build_model({"name": "b2_visual_only"})
    assert isinstance(m.visual_encoder, VisualEncoder)


# ------------------------------------------------------------ augmentation

def test_new_augmentations_keep_shape_and_dtype():
    rng = random.Random(0)
    clip = np.random.default_rng(0).integers(0, 255, (20, 96, 96), dtype=np.uint8)
    for fn in (lambda c: random_scale(c, 0.1, rng), lambda c: random_time_mask(c, 2, 6, rng),
               lambda c: random_erase(c, 1.0, 0.2, rng),
               lambda c: apply_visual_augmentation(c, rng=rng, max_scale=0.1, time_masks=2,
                                                   p_erase=1.0)):
        out = fn(clip)
        assert out.shape == clip.shape and out.dtype == clip.dtype


def test_time_mask_replaces_frames_with_mean():
    clip = np.arange(10, dtype=np.float32)[:, None, None] * np.ones((10, 4, 4), np.float32)
    out = random_time_mask(clip, n_masks=1, max_width=9, rng=random.Random(3))
    changed = [t for t in range(10) if not np.allclose(out[t], clip[t])]
    assert changed and np.allclose(out[changed], clip.mean(axis=0))


def test_erase_applies_same_rectangle_to_every_frame():
    clip = np.random.default_rng(1).integers(0, 255, (5, 96, 96), dtype=np.uint8)
    out = random_erase(clip, p=1.0, max_area=0.2, rng=random.Random(2))
    diff = out != clip
    assert diff.any()
    box = out[0] == out[0, diff[0]][0]
    for t in range(5):
        assert np.all(out[t][diff[0]] == out[0][diff[0]])  # same fill, same place
    assert box.any()


# ------------------------------------------------------------ B3 modality dropout / init

def _b3_batch(B=4, T=10):
    return {"features": torch.randn(B, 4 * T, 80), "feat_lengths": torch.full((B,), 4 * T),
            "video": torch.rand(B, 1, T, 96, 96), "lengths": torch.full((B,), T)}


def test_modality_dropout_only_in_training():
    torch.manual_seed(0)
    m = B3Concat(p_audio_drop=0.5, p_visual_drop=0.5)
    batch = _b3_batch()
    m.eval()
    with torch.no_grad():
        a = m(batch)["log_probs"]
        b = m(batch)["log_probs"]
    assert torch.allclose(a, b)
    with pytest.raises(ValueError):
        B3Concat(p_audio_drop=0.8, p_visual_drop=0.5)


def test_build_model_reads_modality_dropout():
    m = build_model({"name": "b3_concat", "modality_dropout": {"audio": 0.3, "visual": 0.1}})
    assert (m.p_audio_drop, m.p_visual_drop) == (0.3, 0.1)


def test_load_pretrained_copies_encoder_weights(tmp_path):
    from scripts.train import load_pretrained
    b2 = B2VisualOnly()
    torch.save({"model": b2.state_dict()}, tmp_path / "b2.pt")
    b3 = B3Concat()
    load_pretrained(b3, {"visual_encoder": str(tmp_path / "b2.pt")})
    for k, v in b2.visual_encoder.state_dict().items():
        assert torch.equal(v, b3.visual_encoder.state_dict()[k])


def test_load_pretrained_rejects_mismatched_encoder(tmp_path):
    from scripts.train import load_pretrained
    torch.save({"model": B2VisualOnly().state_dict()}, tmp_path / "b2.pt")
    b3 = B3Concat(model_cfg={"visual_encoder": "resnet18_bilstm"})
    with pytest.raises(ValueError, match="does not match"):
        load_pretrained(b3, {"visual_encoder": str(tmp_path / "b2.pt")})

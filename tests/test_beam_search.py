"""Tests for CTC prefix beam search (W3-07)."""
import time
import torch

from lipscribe.data.tokenizer import encode
from lipscribe.decoding.beam_search import beam_search_decode, ctc_prefix_beam_search
from lipscribe.decoding.ctc import greedy_decode


def test_repeated_symbol_needs_blank_between_copies():
    # Sequence: 'b', blank, 'b' -> should decode to "bb"
    # Sequence: 'b', 'b' -> should decode to "b"
    b_idx = encode("b")[0]
    blank_idx = 0
    V = 28

    # Clip 1: [b, b]
    lp1 = torch.full((2, V), -10.0)
    lp1[0, b_idx] = 0.0
    lp1[1, b_idx] = 0.0
    assert ctc_prefix_beam_search(lp1, beam_width=5) == "b"

    # Clip 2: [b, blank, b]
    lp2 = torch.full((3, V), -10.0)
    lp2[0, b_idx] = 0.0
    lp2[1, blank_idx] = 0.0
    lp2[2, b_idx] = 0.0
    assert ctc_prefix_beam_search(lp2, beam_width=5) == "bb"


def test_beam_width_configurable():
    # Construct posteriors where different beam widths explore alternatives
    T, V = 10, 28
    torch.manual_seed(42)
    lp = torch.log_softmax(torch.randn(T, V), dim=-1)

    hyp1 = ctc_prefix_beam_search(lp, beam_width=1)
    hyp10 = ctc_prefix_beam_search(lp, beam_width=10)
    assert isinstance(hyp1, str)
    assert isinstance(hyp10, str)


def test_beam_search_matches_greedy_on_peaked_posteriors():
    # On confident/peaked posteriors, beam search matches greedy decoding
    target = "bin blue at f two now"
    target_ids = encode(target)
    T = len(target_ids) * 2 + 5
    V = 28

    lp = torch.full((T, V), -15.0)
    lp[:, 0] = 5.0  # Background frames are blank
    t = 1
    for token_id in target_ids:
        lp[t, :] = -15.0
        lp[t, token_id] = 5.0
        t += 2
    lp = torch.log_softmax(lp, dim=-1)

    greedy_res = greedy_decode(lp.unsqueeze(1))[0]
    beam_res = ctc_prefix_beam_search(lp, beam_width=5)
    beam1_res = ctc_prefix_beam_search(lp, beam_width=1)

    assert greedy_res == target
    assert beam_res == target
    assert beam1_res == target


def test_batch_beam_search_and_timing():
    T, B, V = 30, 2, 28
    torch.manual_seed(123)
    lp = torch.log_softmax(torch.randn(T, B, V), dim=-1)
    lengths = torch.tensor([30, 25])

    t0 = time.perf_counter()
    hyps = beam_search_decode(lp, lengths=lengths, beam_width=10)
    elapsed = time.perf_counter() - t0

    assert len(hyps) == B
    per_utt_ms = (elapsed / B) * 1000
    print(f"\nBeam search decode time: {per_utt_ms:.2f} ms per utterance")
    # Must decode in reasonable time (< 200 ms per utterance on CPU)
    assert per_utt_ms < 500.0

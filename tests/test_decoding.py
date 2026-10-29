import torch

from lipscribe.data.tokenizer import BLANK_ID, VOCAB_SIZE, encode
from lipscribe.decoding.ctc import collapse, collapse_ids, greedy_decode

b, i, n, e = encode("bine")
_ = BLANK_ID


def test_collapse_bin():
    assert collapse([b, b, _, i, _, n, n]) == "bin"


def test_collapse_double_letter_needs_blank():
    assert collapse([e, _, e]) == "ee"
    assert collapse([e, e]) == "e"


def test_collapse_all_blank_is_empty():
    assert collapse([_, _, _]) == ""
    assert collapse_ids([]) == []


def _one_hot_log_probs(paths, T):
    """Build [T, B, V] log-probs whose argmax follows each path."""
    lp = torch.full((T, len(paths), VOCAB_SIZE), -10.0)
    for bi, path in enumerate(paths):
        for t in range(T):
            sym = path[t] if t < len(path) else BLANK_ID
            lp[t, bi, sym] = 0.0
    return lp.log_softmax(dim=-1)


def test_greedy_decode_batch():
    paths = [[b, b, _, i, _, n, n], [e, _, e]]
    lp = _one_hot_log_probs(paths, T=7)
    assert greedy_decode(lp) == ["bin", "ee"]


def test_greedy_decode_ignores_padding():
    path = [b, i, n] + [e] * 4  # frames 3..6 are padding
    lp = _one_hot_log_probs([path], T=7)
    assert greedy_decode(lp, torch.tensor([3])) == ["bin"]
    assert greedy_decode(lp, [7]) == ["bine"]

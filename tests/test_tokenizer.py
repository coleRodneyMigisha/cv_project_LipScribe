import pytest
from lipscribe.data.tokenizer import (
    BLANK_ID, VOCAB_SIZE, encode, decode,
)

# GRID closed vocabulary (51 words) — command x colour x preposition x letter x digit x adverb
GRID_WORDS = (
    ["bin", "lay", "place", "put"]
    + ["blue", "green", "red", "white"]
    + ["at", "in"]
    + list("bcdfghjklmnpstvwz")          # 19 consonants used in GRID letter slot
    + ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    + ["again", "now", "please"]
)

GRID_SENTENCES = [
    "bin blue at a zero now",
    "place green at b four please",
    "put red in f nine again",
    "lay white at z seven now",
]


@pytest.mark.parametrize("word", GRID_WORDS)
def test_word_roundtrip(word):
    assert decode(encode(word)) == word


@pytest.mark.parametrize("sent", GRID_SENTENCES)
def test_sentence_roundtrip(sent):
    assert decode(encode(sent)) == sent


def test_encode_never_emits_blank():
    for s in GRID_SENTENCES:
        assert BLANK_ID not in encode(s)


def test_vocab_size():
    assert VOCAB_SIZE == 28
    assert len(encode("a")) == 1
    assert encode("a")[0] == 2  # 'a' is index 2


def test_unknown_char_raises():
    with pytest.raises(ValueError):
        encode("hello!")  # '!' not in vocab


def test_decode_skips_blank():
    # blank in the middle is ignored
    assert decode([2, BLANK_ID, 3]) == "ab"
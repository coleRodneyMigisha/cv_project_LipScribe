import pytest
from lipscribe.data.tokenizer import (
    BLANK_ID, VOCAB_SIZE, encode, decode,
)
from lipscribe.data.transcripts import parse_align, parse_align_text

# GRID closed vocabulary (51 words): command x colour x preposition x letter x digit x adverb
GRID_WORDS = (
    ["bin", "lay", "place", "set"]
    + ["blue", "green", "red", "white"]
    + ["at", "by", "in", "with"]
    + list("abcdefghijklmnopqrstuvxyz")    # 25 letters; GRID has no "w"
    + ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    + ["again", "now", "please", "soon"]
)

GRID_SENTENCES = [
    "bin blue at a zero now",
    "place green by b four please",
    "set red in f nine again",
    "lay white with z seven soon",
    "place green with e eight please",   # longest GRID transcript: 31 characters
]

ALIGN = """0 23750 sil
23750 29500 bin
29500 34000 blue
34000 35500 at
35500 38000 sp
38000 41000 f
41000 47250 two
47250 53000 now
53000 74500 sil
"""


def test_grid_vocabulary_has_51_words():
    assert len(GRID_WORDS) == len(set(GRID_WORDS)) == 51


@pytest.mark.parametrize("word", GRID_WORDS)
def test_word_roundtrip(word):
    assert decode(encode(word)) == word


@pytest.mark.parametrize("sent", GRID_SENTENCES)
def test_sentence_roundtrip(sent):
    assert decode(encode(sent)) == sent


def test_encode_never_emits_blank():
    for s in GRID_SENTENCES:
        assert BLANK_ID not in encode(s)


def test_longest_transcript_fits_75_frames():
    # CTC needs T >= L + repeated characters; T = 75 for a 3 s clip.
    longest = max(GRID_SENTENCES, key=len)
    ids = encode(longest)
    repeats = sum(a == b for a, b in zip(ids, ids[1:]))
    assert len(ids) == 31 and len(ids) + repeats <= 75


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


def test_parse_align_drops_silence_and_pauses():
    assert parse_align_text(ALIGN) == "bin blue at f two now"


def test_parse_align_reads_file(tmp_path):
    path = tmp_path / "bbaf2n.align"
    path.write_text(ALIGN)
    transcript = parse_align(path)
    assert transcript == "bin blue at f two now"
    assert decode(encode(transcript)) == transcript

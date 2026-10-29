import pytest

from lipscribe.evaluation.metrics import cer, error_counts, wer


def test_wer_one_substitution_in_six_words():
    assert wer(["bin blue at f two now"], ["bin blue at s two now"]) == pytest.approx(1 / 6)


def test_wer_accepts_single_strings():
    assert wer("bin blue at f two now", "bin blue at s two now") == pytest.approx(1 / 6)


def test_perfect_match_is_zero():
    s = "place green at b four please"
    assert wer([s], [s]) == 0.0
    assert cer([s], [s]) == 0.0


def test_corpus_level_not_mean_of_utterances():
    refs = ["bin blue at f two now", "red"]   # 6 + 1 = 7 words
    hyps = ["bin blue at f two now", "blue"]  # 0 + 1 errors
    # corpus level: 1 / 7; mean of per-utterance rates would be (0 + 1) / 2
    assert wer(refs, hyps) == pytest.approx(1 / 7)


def test_cer_counts_characters():
    # "bin" vs "ban": 1 substitution over 3 characters
    assert cer(["bin"], ["ban"]) == pytest.approx(1 / 3)


def test_empty_hypothesis_is_all_deletions():
    k = error_counts(["bin blue"], [""])
    assert k["word_del"] == 2
    assert wer(["bin blue"], [""]) == pytest.approx(1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        wer(["a", "b"], ["a"])

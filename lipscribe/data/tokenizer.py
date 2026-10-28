"""Character tokenizer for LipScribe CTC output.

Vocabulary: blank (0), space (1), a-z (2-27) -> 28 symbols total.
"""
from __future__ import annotations

BLANK_ID = 0
SPACE_ID = 1
CHARS = "abcdefghijklmnopqrstuvwxyz"

VOCAB = {c: i + 2 for i, c in enumerate(CHARS)}
VOCAB[" "] = SPACE_ID
ID_TO_CHAR = {i: c for c, i in VOCAB.items()}
VOCAB_SIZE = 28


def encode(text: str) -> list[int]:
    """Lower-case text -> list of symbol ids. Never emits blank."""
    text = text.lower()
    try:
        return [VOCAB[c] for c in text]
    except KeyError as e:
        raise ValueError(f"Character {e.args[0]!r} not in vocabulary: {text!r}") from e


def decode(ids) -> str:
    """Symbol ids -> string. Blank is skipped."""
    return "".join(ID_TO_CHAR[i] for i in ids if i != BLANK_ID)
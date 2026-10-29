"""Build transcripts from GRID .align files."""
from pathlib import Path

SILENCE_TOKENS = ("sil", "sp")


def parse_align_text(text: str) -> str:
    """Contents of one .align file -> lower-case transcript string.

    GRID .align format, one token per line as ``start end word`` (times in
    1/25000 s units):
        0 23750 sil            <- leading silence (drop)
        23750 29500 bin
        29500 34000 blue
        ...
        53000 74500 sil        <- trailing silence (drop)
    Short pauses (``sp``) are dropped as well. Letters stay single characters
    ("b") and digits are already words ("four") in GRID.
    """
    words = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        word = parts[-1].lower()
        if word in SILENCE_TOKENS:
            continue
        words.append(word)
    return " ".join(words)


def parse_align(align_path: str | Path) -> str:
    """Parse one .align file -> lower-case transcript string."""
    return parse_align_text(Path(align_path).read_text(encoding="utf-8", errors="replace"))

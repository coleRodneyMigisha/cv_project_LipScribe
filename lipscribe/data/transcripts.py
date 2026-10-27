"""Build transcripts from GRID .align files."""
from pathlib import Path


def parse_align(align_path: str | Path) -> str:
    """Parse one .align file -> lower-case transcript string.

    GRID .align format, one token per line:
        sil 0.00 0.10          <- leading silence (drop)
        put 0.10 0.30
        r 0.30 0.50
        ...
        sp 2.90 2.95           <- short pause (drop)
        ...
        sil 3.20 3.30          <- trailing silence (drop)
    """
    words = []
    with open(align_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            word = parts[0].lower()
            if word in ("sil", "sp"):
                continue
            words.append(word)
    return " ".join(words)
"""Build transcripts from GRID .align files."""
from pathlib import Path


def parse_align(align_path: str | Path) -> str:
    """Parse one .align file -> lower-case transcript string.

    GRID .align format: one token per line, columns are:
        start_time(ms)  end_time(ms)  word

    sil = silence, sp = short pause; both are dropped.
    Example: "0 14000 sil" -> dropped, "14000 20500 bin" -> "bin"
    """
    words = []
    with open(align_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            word = parts[2].lower()   # word is the THIRD column
            if word in ("sil", "sp"):
                continue
            words.append(word)
    return " ".join(words)
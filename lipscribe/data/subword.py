"""Auto-AVSR's unigram-5000 word pieces for CTC (pretrained/unigram5000.model + _units.txt).

Ids follow Auto-AVSR's token list: blank 0, the 5,047 units 1..5047 (``<unk>`` = 1), ``<eos>``
5048 -> 5,049 outputs, so its pretrained CTC layer (``ctc.ctc_lo``) can be fine-tuned as is.
Our transcripts are lower-case letters and spaces; the pieces are upper-case.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPM_MODEL = ROOT / "pretrained" / "unigram5000.model"
UNITS = ROOT / "pretrained" / "unigram5000_units.txt"
BLANK_ID = 0
VOCAB_SIZE = 5049


@lru_cache(maxsize=1)
def _tables():
    import sentencepiece as spm
    units = [ln.split()[0] for ln in UNITS.read_text(encoding="utf8").splitlines()]
    token_list = ["<blank>"] + units + ["<eos>"]
    if len(token_list) != VOCAB_SIZE:
        raise ValueError(f"{UNITS}: expected {VOCAB_SIZE - 2} units, got {len(units)}")
    return spm.SentencePieceProcessor(model_file=str(SPM_MODEL)), token_list, {u: i for i, u in enumerate(token_list)}


def encode(text: str) -> list[int]:
    sp, _, p2i = _tables()
    unk = p2i["<unk>"]
    return [p2i.get(p, unk) for p in sp.encode(text.upper(), out_type=str)]


def decode(ids) -> str:
    """Collapsed ids (no repeats) -> lower-case text; blank and <eos> are skipped."""
    _, tl, _ = _tables()
    s = "".join(tl[i] for i in ids if 0 < i < VOCAB_SIZE - 1)
    return " ".join(s.replace("▁", " ").split()).lower()

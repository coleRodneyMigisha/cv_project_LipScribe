from lipscribe.decoding.beam_search import beam_search_decode, ctc_prefix_beam_search
from lipscribe.decoding.ctc import collapse, collapse_ids, greedy_decode
from lipscribe.decoding.grammar import GRID_GRAMMAR, GrammarCTCDecoder, grammar_decode

__all__ = [
    "GRID_GRAMMAR",
    "GrammarCTCDecoder",
    "beam_search_decode",
    "collapse",
    "collapse_ids",
    "ctc_prefix_beam_search",
    "get_decoder",
    "grammar_decode",
    "greedy_decode",
]


def get_decoder(decode_cfg: dict | None):
    """Return ``fn(log_probs [T,B,V], lengths) -> list[str]`` for a config ``decode`` section.

    ``method``: ``greedy`` (default), ``beam`` (prefix beam search, ``beam_width``)
    , ``grammar`` (GRID grammar-constrained Viterbi) or ``lexicon_lm`` (open-vocabulary
    lexicon + word-bigram LM beam search; ``lm_path``, ``alpha``, ``beta``, ``beam_width``).
    """
    decode_cfg = decode_cfg or {}
    method = decode_cfg.get("method", "greedy")
    if method == "greedy":
        return greedy_decode
    if method == "grammar":
        return grammar_decode
    if method == "beam":
        width = int(decode_cfg.get("beam_width", 10))
        return lambda lp, lengths: beam_search_decode(lp, lengths, beam_width=width)
    if method == "lexicon_lm":
        from lipscribe.decoding.lexicon_lm import LexiconLMDecoder, WordBigramLM
        return LexiconLMDecoder(WordBigramLM.load(decode_cfg["lm_path"]),
                                alpha=float(decode_cfg.get("alpha", 0.8)), beta=float(decode_cfg.get("beta", 1.5)),
                                beam_width=int(decode_cfg.get("beam_width", 32)))
    raise ValueError(f"Unknown decode method {method!r} (greedy, beam, grammar or lexicon_lm)")

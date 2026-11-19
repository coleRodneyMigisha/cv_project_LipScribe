#!/usr/bin/env bash
# Flix v4 decoding (14 Nov 2026): build the LibriSpeech + own-transcript 4-gram LM, then beam-search
# decode every word-piece model with both LMs (own transcripts only, and LibriSpeech + own); alpha and
# beta are chosen on validation per model. fusion_gated is added once scripts/flix_v4_fusion4.sh is done.
#   nohup scripts/flix_v4_beam_run.sh > logs/flix_v4_beam_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; R=reports/flix_v4; S=$R/state
note() { echo "- $(date '+%Y-%m-%d %H:%M') $*" | tee -a "$R/decision_log.md"; }
GZ=lm/librispeech/librispeech-lm-norm.txt.gz
while pgrep -f "curl -sSL -o librispeech-lm-norm" > /dev/null; do sleep 20; done
gzip -t "$GZ" || { note "LibriSpeech LM text download is corrupt ($GZ)."; exit 1; }
if [ ! -f lm/flix_libri_4gram.arpa ]; then
  $PY -m scripts.flix_v4_lm_big --libri "$GZ" --libri-words 40000000 --out lm/flix_libri_4gram.arpa \
      > logs/flix_v4_lm_big.log 2>&1 || { note "building the LibriSpeech LM failed (logs/flix_v4_lm_big.log)"; exit 1; }
fi
PERP=$($PY - <<'PYEOF'
import csv, kenlm
out = []
for lm in ("lm/flix_train_4gram.arpa", "lm/flix_libri_4gram.arpa"):
    m = kenlm.Model(lm)
    for name in ("ted_val", "film_val"):
        rows = list(csv.DictReader(open(f"data/flix_v4c/{name}.csv")))
        lp = sum(m.score(r["transcript"].lower(), bos=True, eos=True) for r in rows)
        n = sum(len(r["transcript"].split()) + 1 for r in rows)
        out.append(f"{lm.split('/')[-1]} {name} perplexity {10 ** (-lp / n):.0f}")
print("; ".join(out))
PYEOF
)
note "LMs built (training transcripts only, and 40M LibriSpeech words + transcripts x5, 3/4-gram singletons pruned): $PERP"
PYTHONPATH=. $PY scripts/flix_v4_beam.py decode --names twin av visual --lm lm/flix_libri_4gram.arpa \
    --out $R/beam_libri > logs/flix_v4_beam_libri.log 2>&1 || note "LibriSpeech-LM decoding failed (logs/flix_v4_beam_libri.log)"
while [ ! -f "$R/FUSION4_DONE" ]; do sleep 60; done
PYTHONPATH=. $PY scripts/flix_v4_beam.py dump --run gated="$(cat $S/fusion_gated.run)" >> logs/flix_v4_beam_dump.log 2>&1
PYTHONPATH=. $PY scripts/flix_v4_beam.py decode --names twin av visual gated --out $R/beam >> logs/flix_v4_beam_decode.log 2>&1
PYTHONPATH=. $PY scripts/flix_v4_beam.py decode --names twin av visual gated --lm lm/flix_libri_4gram.arpa \
    --out $R/beam_libri >> logs/flix_v4_beam_libri.log 2>&1
{ echo; echo "Beam search + LM, own-transcript LM:"; echo; cat $R/beam.md; echo
  echo "Beam search + LM, LibriSpeech + own-transcript LM:"; echo; cat $R/beam_libri.md; } >> $R/decision_log.md
note "Decoding finished."
touch $R/BEAM_DONE

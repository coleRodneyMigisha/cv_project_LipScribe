#!/usr/bin/env bash
# X6 'Flix': after the sweep, decode every model with the lexicon + bigram LM ("grammar" for films).
#   nohup setsid bash scripts/flix_lm_run.sh > logs/flix_lm.log 2>&1 &
# SWEEP=reports/flix_sweep DATA=data/flix_only OUT=reports/flix_lm (override for the full set)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SWEEP=${SWEEP:-reports/flix_sweep}; DATA=${DATA:-data/flix_only}; OUT=${OUT:-reports/flix_lm}
mkdir -p "$OUT"; rm -f "$OUT/DONE" "$OUT/FAILED"
trap 'echo "FAILED at line $LINENO" | tee "$OUT/FAILED"' ERR
until [ -f "$SWEEP/DONE" ]; do [ -f "$SWEEP/FAILED" ] && { echo "sweep failed"; false; }; sleep 60; done
echo "== $(date +%T) sweep done, LM decoding"
ARGS=(--ckpt base=checkpoints/flix_base_p4.pt --ckpt tuned=checkpoints/flix_finetuned_p4.pt)
while IFS=, read -r seed run; do ARGS+=(--ckpt "sweep_s$seed=$run/best.pt"); done < <(
  $PY -c "import csv,sys; [print(r['seed']+','+r['run_dir']) for r in csv.DictReader(open('$SWEEP/sweep.csv')) if r['trial'].startswith('final') and r['run_dir']]")
$PY -m scripts.flix_lm_eval --lm data/flix_lm.json --data "$DATA" --out "$OUT" "${ARGS[@]}"
touch "$OUT/DONE"; echo "== $(date +%T) done"

#!/usr/bin/env bash
# X6 'Flix' Alienware stage: prepare -> zero-shot test -> fine-tune a copy of P4 -> re-test -> report.
# Runs in ~/Desktop/lipscribe_flix (a separate checkout); the main model in
# ../cv_project_LipScribe is only read, never written.
#   BUDGET=25 nohup setsid bash scripts/flix_run.sh > logs/flix_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BUDGET=${BUDGET:-25}
OUT=reports/flix
BASE=checkpoints/flix_base_p4.pt
mkdir -p "$OUT" logs
rm -f "$OUT/DONE" "$OUT/FAILED"
trap 'echo "FAILED at line $LINENO" | tee "$OUT/FAILED"' ERR
echo "== $(date +%T) start (fine-tune budget ${BUDGET} min)"

$PY -m scripts.flix_prepare --flix flix --grid-root ../cv_project_LipScribe --out data/flix

evals() {  # <checkpoint> <prefix>
  $PY -m scripts.flix_eval --checkpoint "$1" --manifest data/flix/flix_test.csv --tag "$2_flix"
  $PY -m scripts.flix_eval --checkpoint "$1" --manifest data/flix/flix_test.csv --tag "$2_flix_lips_only" --drop-modality audio
  $PY -m scripts.flix_eval --checkpoint "$1" --manifest data/flix/flix_test.csv --tag "$2_flix_sound_only" --drop-modality visual
  $PY -m scripts.flix_eval --checkpoint "$1" --manifest data/flix/grid_test.csv --tag "$2_grid"
}

echo "== $(date +%T) zero-shot: original P4 copy"
evals "$BASE" base

echo "== $(date +%T) fine-tune"
$PY -m scripts.train --config configs/flix_finetune.yaml --time-budget-min "$BUDGET" --no-results
RUN=$(ls -td runs/flix_finetune_p4_* | head -1)
cp "$RUN/best.pt" checkpoints/flix_finetuned_p4.pt
echo "$RUN" > "$OUT/run_dir.txt"

echo "== $(date +%T) re-test fine-tuned"
evals checkpoints/flix_finetuned_p4.pt tuned

$PY -m scripts.flix_report --dir "$OUT" --run "$RUN"
md5sum ../cv_project_LipScribe/checkpoints/p_proposed_best.pt "$BASE" | tee "$OUT/main_model_md5.txt"
touch "$OUT/DONE"
echo "== $(date +%T) done"

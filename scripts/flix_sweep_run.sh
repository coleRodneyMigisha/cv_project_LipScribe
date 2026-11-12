#!/usr/bin/env bash
# X6 'Flix' stage 2 on the Alienware: retrain Model P on Flix only (P4 recipe) + hyperparameter sweep.
#   TRIALS=24 nohup setsid bash scripts/flix_sweep_run.sh > logs/flix_sweep.log 2>&1 &
# Resumable: rerunning skips finished trials (reports/flix_sweep/sweep.csv).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=reports/flix_sweep
mkdir -p "$OUT" logs checkpoints
rm -f "$OUT/DONE" "$OUT/FAILED"
trap 'echo "FAILED at line $LINENO" | tee "$OUT/FAILED"' ERR
echo "== $(date +%T) start"
# P4 recipe needs the B1 / B2-v2 encoders: read-only copies of the main checkpoints
for c in b1_audio_only_best.pt b2_visual_only_v2_best.pt; do
  [ -f checkpoints/$c ] || cp ../cv_project_LipScribe/checkpoints/$c checkpoints/
done
$PY -m scripts.flix_prepare --flix flix --grid-root ../cv_project_LipScribe --out data/flix_only \
    --grid-train 0 --grid-val 0 --flix-repeat 1 --extra-val i_robot skyfall
$PY -m scripts.flix_sweep --trials "${TRIALS:-24}" --max-epochs "${MAX_EPOCHS:-80}"
$PY -m scripts.flix_sweep_report
md5sum ../cv_project_LipScribe/checkpoints/p_proposed_best.pt | tee "$OUT/main_model_md5.txt"
touch "$OUT/DONE"
echo "== $(date +%T) done"

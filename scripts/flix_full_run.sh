#!/usr/bin/env bash
# X6 'Flix' full set on the Alienware: mine EVERY subtitle cue of the copied films/series,
# wait for the first sweep (scripts/flix_sweep_run.sh) to finish, then retrain Model P on
# the full set (top-5 configs of the first sweep + 150-epoch finals, 3 seeds) and report.
# All earlier models are re-scored on the new, larger test set so every row is comparable.
#   VIDEOS=~/Desktop/lipscribe_flix/videos nohup setsid bash scripts/flix_full_run.sh > logs/flix_full.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
PY=.venv/bin/python
OUT=reports/flix_full
VID=${VIDEOS:-$PWD/videos}
mkdir -p "$OUT" logs flix_full/clips
rm -f "$OUT/DONE" "$OUT/FAILED"
trap 'echo "FAILED at line $LINENO" | tee "$OUT/FAILED"' ERR
echo "== $(date +%T) start (videos: $VID)"
until command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null; do
  echo "$(date +%T) waiting for ffmpeg/ffprobe (sudo apt install ffmpeg, or ~/bin)"; sleep 600
done
[ -d "$VID/movies" ] || { echo "no videos at $VID"; false; }

# 1. mine every cue (resume: start from the 305 clips already mined)
if [ ! -f flix_full/manifest_local.csv ]; then
  cp flix/manifest_local.csv flix_full/ && cp -n flix/clips/* flix_full/clips/
fi
$PY -m scripts.mine_flix --out flix_full --videos "$VID" --per-movie 100000 --per-episode 100000 \
    --workers "${WORKERS:-12}" --deadline "$(date -d '-1 min' +%H:%M)"
echo "== $(date +%T) mined $(($(wc -l < flix_full/manifest_local.csv) - 1)) clips"
# the films are only needed for mining: do not leave them on the shared machine
[ "${KEEP_VIDEOS:-0}" = 1 ] || { rm -rf "$VID"; echo "removed copied videos ($VID)"; }

# 2. wait for the first sweep to release the GPU
until [ -f reports/flix_sweep/DONE ] || [ -f reports/flix_sweep/FAILED ]; do sleep 300; done
echo "== $(date +%T) first sweep finished"

# 3. data: same title split as before (test titles unchanged, i_robot + skyfall in val)
$PY -m scripts.flix_prepare --flix flix_full --grid-root ../cv_project_LipScribe --out data/flix_full \
    --grid-train 0 --grid-val 0 --flix-repeat 1 --extra-val i_robot skyfall

evals() {  # <checkpoint> <out dir> <tag prefix>
  local F="--manifest data/flix_full/flix_test.csv --splits data/flix_full/splits.csv --out $2"
  $PY -m scripts.flix_eval --checkpoint "$1" $F --tag "$3_flix"
  $PY -m scripts.flix_eval --checkpoint "$1" $F --tag "$3_flix_lips_only" --drop-modality audio
  $PY -m scripts.flix_eval --checkpoint "$1" $F --tag "$3_flix_sound_only" --drop-modality visual
  $PY -m scripts.flix_eval --checkpoint "$1" --manifest data/flix/grid_test.csv --splits data/flix/splits.csv \
      --out "$2" --tag "$3_grid"
}
# 4. earlier models on the new test set
evals checkpoints/flix_base_p4.pt "$OUT/first" base
evals checkpoints/flix_finetuned_p4.pt "$OUT/first" tuned
$PY - <<'EOF' > "$OUT/prev_checkpoints.txt"
import csv
for r in csv.DictReader(open("reports/flix_sweep/sweep.csv")):
    if r["trial"].startswith("final") and r["run_dir"]:
        print(r["seed"], r["run_dir"] + "/best.pt")
EOF
while read -r seed ck; do evals "$ck" "$OUT/prev" "sweep_s$seed"; done < "$OUT/prev_checkpoints.txt"

# 5. retrain on the full set
$PY -m scripts.flix_sweep --data data/flix_full --out "$OUT" --from-sweep reports/flix_sweep/sweep.csv \
    --top "${TOP:-5}" --max-epochs "${MAX_EPOCHS:-80}" --final-epochs "${FINAL_EPOCHS:-150}"
$PY -m scripts.flix_sweep_report --out "$OUT" --first "$OUT/first" --stats data/flix_full/stats.json \
    --prev "$OUT/prev" --prev-label "Retrained on the first 305 clips"
touch "$OUT/DONE"   # sweep part finished (flix_lm_run waits on it)
# 6. film "grammar": lexicon + bigram LM decoding of the full-set models
SWEEP="$OUT" DATA=data/flix_full OUT="$OUT/lm" bash scripts/flix_lm_run.sh
md5sum ../cv_project_LipScribe/checkpoints/p_proposed_best.pt | tee "$OUT/main_model_md5.txt"
touch "$OUT/DONE"
echo "== $(date +%T) done"

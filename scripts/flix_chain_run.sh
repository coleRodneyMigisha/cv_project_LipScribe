#!/usr/bin/env bash
# X6 'Flix' chain on the Alienware (val/test clips never change, so every stage compares):
#  v2      v1 clips + laptop-mined batch-1 clips -> retrain (t03, 3 seeds x 150) -> word grammar
#  ResNet  11 ways of training the eyes, tuned on the v2 data (no finals)
#  v3      v2 + Whisper-labelled films + the full TED harvest: all 11 ResNet variations (60 epochs,
#          patience 20), winner x 3 seeds x 60 epochs -> film test, TED test -> word grammar
#   nohup setsid bash scripts/flix_chain_run.sh > logs/flix_chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
V2=reports/flix_v2; RN=reports/flix_resnet; V3=reports/flix_v3
mkdir -p "$V2" "$RN" "$V3" logs flix_v2/clips flix_v3/clips
rm -f "$V3/ALL_DONE" "$V3/FAILED"
trap 'echo "FAILED at line $LINENO" | tee "$V3/FAILED"' ERR

merge() {  # <base manifest> <laptop manifest> <out manifest>
  $PY - "$1" "$2" "$3" <<'PYEOF'
import csv, sys
base, lap, out = sys.argv[1:]
rows = list(csv.DictReader(open(base)))
seen = {(r["title"], f"{float(r['start_s']):.3f}") for r in rows}
new = [r for r in csv.DictReader(open(lap)) if (r["title"], f"{float(r['start_s']):.3f}") not in seen]
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows + new)
print(f"merged {out}: {len(rows)} + {len(new)} new clips")
PYEOF
}
prep() {  # <clip dir> <data dir>
  $PY -m scripts.flix_prepare --flix "$1" --grid-root ../cv_project_LipScribe --out "$2" \
      --grid-train 0 --grid-val 0 --flix-repeat 1 --extra-val i_robot skyfall
}

# ---- v2
if [ ! -f "$V2/lm/DONE" ]; then
  echo "== $(date +%T) v2: waiting for the laptop's batch-1 clips"
  until [ -f flix_v2/.mined_done ]; do sleep 60; done
  cp -n flix/clips/* flix_v2/clips/
  merge flix/manifest_local.csv flix_v2/manifest_laptop.csv flix_v2/manifest_local.csv
  prep flix_v2 data/flix_v2
  $PY -m scripts.flix_sweep --data data/flix_v2 --out "$V2" --from-sweep reports/flix_sweep/sweep.csv \
      --finals-only --final-epochs 150
  touch "$V2/DONE"
  SWEEP="$V2" DATA=data/flix_v2 OUT="$V2/lm" bash scripts/flix_lm_run.sh
fi

# ---- ResNet tuning on v2 (winner only, no finals)
# (if a tuning run from an earlier start of this chain is still going, wait for it, never run two)
while pgrep -f "^$PY -m scripts.flix_sweep --data data/flix_v2 --out $RN" > /dev/null; do sleep 60; done
if [ ! -f "$RN/best.json" ]; then
  echo "== $(date +%T) ResNet tuning"
  $PY -m scripts.flix_sweep --data data/flix_v2 --out "$RN" --from-sweep "$V2/sweep.csv" --resnet \
      --max-epochs 80 --patience 40 --final-epochs 0
fi

# ---- v3
echo "== $(date +%T) v3: waiting for the laptop's Whisper-labelled clips"
until [ -f flix_v3/.mined_done ]; do sleep 60; done
echo "== $(date +%T) v3: waiting for the TED harvester to finish (full TED set in v3)"
until [ -f flix_ted/.harvest_done ]; do sleep 300; done     # written only when the whole list is done
cp -n flix_v2/clips/* flix_v3/clips/
merge flix_v2/manifest_local.csv flix_v3/manifest_laptop.csv flix_v3/manifest_v2w.csv
# TED: snapshot of what the harvester has mined so far (it keeps running for a later v4)
$PY - <<'PYEOF'
import csv, os
rows = [r for r in csv.DictReader(open("flix_ted/manifest_local.csv")) if os.path.exists(f"flix_ted/clips/{r['utt_id']}.npz")]
with open("flix_v3/manifest_ted.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"TED snapshot: {len(rows)} clips, {sum(float(r['dur_s']) for r in rows) / 3600:.1f} h")
PYEOF
cut -d, -f1 flix_v3/manifest_ted.csv | tail -n +2 | while read -r u; do ln -f "flix_ted/clips/$u.npz" "flix_ted/clips/$u.wav" flix_v3/clips/ 2>/dev/null; done
merge flix_v3/manifest_v2w.csv flix_v3/manifest_ted.csv flix_v3/manifest_local.csv
# 3 % of TED training talks become a TED validation set; best.pt / early stopping use the equal-
# weight mean of film-val and TED-val CER (real-world use is closer to TED than to films)
$PY -m scripts.flix_prepare --flix flix_v3 --grid-root ../cv_project_LipScribe --out data/flix_v3 \
    --grid-train 0 --grid-val 0 --flix-repeat 1 --extra-val i_robot skyfall --ted-val-frac 0.03 --lm
# ---- v3: all 11 ResNet (eyes) variations on the full data (v2 + Whisper films + TED), 60 epochs
# each like the GRID models, patience 20; then the winner 3 times (seeds 1-3) x 60 epochs with no
# early stopping; test titles scored once. Compare with $RN (the same 11 on the v2 data).
# Runs for days (no rush): ~5-7 h per variation on ~42k clips.
$PY -m scripts.flix_sweep --data data/flix_v3 --out "$V3" --from-sweep "$V2/sweep.csv" --resnet \
    --max-epochs "${V3_TUNE_EPOCHS:-60}" --patience "${V3_PATIENCE:-20}" \
    --final-epochs "${V3_EPOCHS:-60}" --seeds ${V3_SEEDS:-1 2 3}
touch "$V3/DONE"
# held-out TED talks, scored on their own (in-domain: clean, frontal speakers)
if [ -f data/flix_v3/ted_test.csv ]; then
  while IFS=, read -r seed run; do
    $PY -m scripts.flix_eval --checkpoint "$run/best.pt" --manifest data/flix_v3/ted_test.csv \
        --splits data/flix_v3/splits.csv --out "$V3" --tag "v3_s${seed}_ted_test"
  done < <($PY -c "import csv; [print(r['seed']+','+r['run_dir']) for r in csv.DictReader(open('$V3/sweep.csv')) if r['trial'].startswith('final') and r['run_dir']]")
fi
LM=data/flix_v3/lm_train.json SWEEP="$V3" DATA=data/flix_v3 OUT="$V3/lm" bash scripts/flix_lm_run.sh
md5sum ../cv_project_LipScribe/checkpoints/p_proposed_best.pt | tee "$V3/main_model_md5.txt"
touch "$V3/ALL_DONE"; echo "== $(date +%T) all done"

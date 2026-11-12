#!/usr/bin/env bash
# X6 'Flix' chain on the Alienware (val/test clips never change, so every stage compares):
#  v2      v1 clips + laptop-mined batch-1 clips -> retrain (t03, 3 seeds x 150) -> word grammar
#  ResNet  11 ways of training the eyes, tuned on the v2 data (no finals)
#  v3      v2 + Whisper-labelled films + the full TED harvest: all 11 ResNet variations (10 epochs
#          each), winner to 30 epochs -> film test, TED test -> word grammar
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
while pgrep -f "^$PY -u -m scripts.ted_harvest" > /dev/null; do sleep 300; done
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
prep flix_v3 data/flix_v3
# ---- v3: all 11 ResNet (eyes) variations on the full data (v2 + Whisper films + TED), each
# V3_TUNE_EPOCHS (10) epochs - already more clip-updates per run than the v2 tuning had - then the
# winner trained on to V3_EPOCHS (30); test titles scored once. Compare with $RN (same 11 on v2 data).
$PY -m scripts.flix_sweep --data data/flix_v3 --out "$V3" --from-sweep "$V2/sweep.csv" --resnet \
    --max-epochs "${V3_TUNE_EPOCHS:-10}" --patience "${V3_TUNE_EPOCHS:-10}" \
    --final-epochs "${V3_EPOCHS:-30}" --seeds ${V3_SEEDS:-1}
touch "$V3/DONE"
# held-out TED talks, scored on their own (in-domain: clean, frontal speakers)
if [ -f data/flix_v3/ted_test.csv ]; then
  while IFS=, read -r seed run; do
    $PY -m scripts.flix_eval --checkpoint "$run/best.pt" --manifest data/flix_v3/ted_test.csv \
        --splits data/flix_v3/splits.csv --out "$V3" --tag "v3_s${seed}_ted_test"
  done < <($PY -c "import csv; [print(r['seed']+','+r['run_dir']) for r in csv.DictReader(open('$V3/sweep.csv')) if r['trial'].startswith('final') and r['run_dir']]")
fi
SWEEP="$V3" DATA=data/flix_v3 OUT="$V3/lm" bash scripts/flix_lm_run.sh
md5sum ../cv_project_LipScribe/checkpoints/p_proposed_best.pt | tee "$V3/main_model_md5.txt"
touch "$V3/ALL_DONE"; echo "== $(date +%T) all done"

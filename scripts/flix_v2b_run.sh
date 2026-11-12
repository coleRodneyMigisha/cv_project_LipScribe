#!/usr/bin/env bash
# X6 'Flix' v2 on the Alienware, on the training clips the laptop mines and ships:
#  1 merge the v1 clips with the clips mined on the laptop (scripts/flix_mine_laptop.sh)
#  2 rebuild the data (val/test clips unchanged, so results compare with v1)
#  3 retrain: best v1 setting (t03), seeds 1-3, 150 epochs        -> reports/flix_v2
#  4 word-grammar (lexicon + bigram LM) decoding                  -> reports/flix_v2/lm
#  5 ResNet (eyes) tuning: 11 variations, then 150-epoch finals   -> reports/flix_resnet
#  6 word-grammar decoding of the ResNet finals                    -> reports/flix_resnet/lm
#   nohup setsid bash scripts/flix_v2_run.sh > logs/flix_v2.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
V2=reports/flix_v2; RN=reports/flix_resnet
mkdir -p "$V2" "$RN" logs flix_v2/clips
rm -f "$V2/FAILED" "$V2/ALL_DONE"
trap 'echo "FAILED at line $LINENO" | tee "$V2/FAILED"' ERR
echo "== $(date +%T) waiting for the laptop's mined clips (scripts/flix_mine_laptop.sh)"
until [ -f flix_v2/.mined_done ]; do sleep 60; done

# 1. merge: the 305 v1 clips + the laptop's new training clips (val/test unchanged)
cp -n flix/clips/* flix_v2/clips/
$PY - <<'PYEOF'
import csv
rows = list(csv.DictReader(open("flix/manifest_local.csv")))
seen = {(r["title"], f"{float(r['start_s']):.3f}") for r in rows}
new = [r for r in csv.DictReader(open("flix_v2/manifest_laptop.csv"))
       if (r["title"], f"{float(r['start_s']):.3f}") not in seen]
with open("flix_v2/manifest_local.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows + new)
print(f"merged: {len(rows)} v1 clips + {len(new)} new clips")
PYEOF

# 2. data
$PY -m scripts.flix_prepare --flix flix_v2 --grid-root ../cv_project_LipScribe --out data/flix_v2 \
    --grid-train 0 --grid-val 0 --flix-repeat 1 --extra-val i_robot skyfall

# 3. retrain with the best v1 setting
$PY -m scripts.flix_sweep --data data/flix_v2 --out "$V2" --from-sweep reports/flix_sweep/sweep.csv \
    --finals-only --final-epochs "${FINAL_EPOCHS:-150}"
touch "$V2/DONE"

# 4. word grammar
SWEEP="$V2" DATA=data/flix_v2 OUT="$V2/lm" bash scripts/flix_lm_run.sh

# 5. ResNet tuning on the v2 data, starting from the v2 setting
$PY -m scripts.flix_sweep --data data/flix_v2 --out "$RN" --from-sweep "$V2/sweep.csv" --resnet \
    --max-epochs "${MAX_EPOCHS:-80}" --patience 40 --final-epochs "${FINAL_EPOCHS:-150}"
touch "$RN/DONE"

# 6. word grammar for the ResNet finals
SWEEP="$RN" DATA=data/flix_v2 OUT="$RN/lm" bash scripts/flix_lm_run.sh
md5sum ../cv_project_LipScribe/checkpoints/p_proposed_best.pt | tee "$V2/main_model_md5.txt"
touch "$V2/ALL_DONE"
echo "== $(date +%T) all done"

#!/usr/bin/env bash
# Flix v4 fusion v2 (14 Nov 2026): train v4_fusion_wp (lips + audio) and its audio-only twin
# one after the other, then score both on the test sets (clean, 0, -5, -10 dB) and probe
# whether the lips are used (shuffled / zeroed video). Notes go to the decision log.
#   nohup scripts/flix_v4_fusion2.sh > logs/flix_v4_fusion2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=reports/flix_v4; S=$R/state
note() { echo "- $(date '+%Y-%m-%d %H:%M') $*" | tee -a "$R/decision_log.md"; }

run_stage() {  # <name>: train configs/flix_v4/<name>.yaml (resume if interrupted)
  local n=$1 run
  [ -f "$S/$n.done" ] && return 0
  run=$(cat "$S/$n.run" 2>/dev/null || true)
  if [ -n "$run" ] && [ -f "$run/latest.pt" ]; then
    $PY -m scripts.train --resume "$run/latest.pt" --no-results >> "logs/flix_v4_$n.log" 2>&1 || return 1
  else
    note "Stage $n started (configs/flix_v4/$n.yaml)."
    $PY -m scripts.train --config "configs/flix_v4/$n.yaml" --no-results --runs-root runs/flix_v4 \
        >> "logs/flix_v4_$n.log" 2>&1 &
    local pid=$!
    for _ in $(seq 120); do
      run=$(grep -o "Run directory: .*" "logs/flix_v4_$n.log" | tail -1 | cut -d' ' -f3-)
      [ -n "$run" ] && [ -d "$run" ] && { echo "$run" > "$S/$n.run"; break; }; sleep 5
    done
    wait $pid || return 1
  fi
  local best
  best=$($PY - "$(cat $S/$n.run)" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] + "/history.csv")))
b = min(rows, key=lambda r: float(r["val_wer"]))
cols = [c for c in rows[0] if c.startswith("wer_")]
print(f"best epoch {b['epoch']}/{len(rows)}: mean val WER {float(b['val_wer']):.1%} ("
      + ", ".join(f"{c[4:]} {float(b[c]):.1%}" for c in cols) + ")")
PYEOF
)
  note "Stage $n done: $best. Run: $(cat $S/$n.run)"
  touch "$S/$n.done"
}

run_stage fusion_wp || { note "Stage fusion_wp failed (logs/flix_v4_fusion_wp.log)."; exit 1; }
run_stage fusion_wp_audio || { note "Stage fusion_wp_audio failed (logs/flix_v4_fusion_wp_audio.log)."; exit 1; }

$PY -m scripts.flix_v4_eval --run fusion_wp="$(cat $S/fusion_wp.run)" fusion_wp_audio="$(cat $S/fusion_wp_audio.run)" \
    --tests film_test=data/flix_v4c/flix_test.csv ted_test=data/flix_v4c/ted_test.csv \
    --out "$R/test_results_wp.json" >> logs/flix_v4_eval.log 2>&1 || note "fusion v2 evaluation failed (logs/flix_v4_eval.log)"
PYTHONPATH=. $PY scripts/flix_v4_probe.py "$(cat $S/fusion_wp.run)" 1122 "$R/probe_fusion_wp.json" \
    >> logs/flix_v4_probe.log 2>&1 || note "probe failed (logs/flix_v4_probe.log)"
{ echo; echo "Fusion v2 (word pieces, lip-anchored) vs its audio-only twin, test sets:"; echo
  cat "$R/test_results_wp.md"; echo; echo "Lip-use probe (WER %, real / shuffled / zeroed video / visual removed):"
  echo '```'; cat "$R/probe_fusion_wp.json"; echo '```'; } >> "$R/decision_log.md"
note "Fusion v2 finished."
touch "$R/FUSION2_DONE"

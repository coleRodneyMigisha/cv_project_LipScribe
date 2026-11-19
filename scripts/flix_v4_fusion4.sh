#!/usr/bin/env bash
# Flix v4 fusion v3 (14 Nov 2026): train v4_fusion_gated (symmetric, reliability-gated fusion of the
# frozen audio twin and fusion_wp's lip stream), then test (clean, 0, -5, -10 dB) and probe lip use.
#   nohup scripts/flix_v4_fusion4.sh > logs/flix_v4_fusion4.log 2>&1 &
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


cat >> "$R/decision_log.md" <<'NOTE'

## Fusion v3: symmetric reliability-gated fusion (14 Nov 2026)

- fusion_wp was lip-anchored (f = v + MLP) and over-trusted the lips. A fine-tune could not undo that.
- Combining the outputs at a fixed weight helped at every SNR, but only slightly, and it lost the large −10 dB gain.
- v4_fusion_gated builds in neither stream:
  - Frozen audio stream from the twin and frozen lip stream from fusion_wp.
  - A 768-channel per-frame gate on both features plus each stream's own confidence (CTC entropy and blank probability).
  - A zero-initialised synergy MLP.
  - 3 Transformer layers and a new word-piece head.
  - Balanced modality dropout (0.15 / 0.15), noise levels spread evenly from clean to −10 dB.
- Success criterion: no worse than the twin on clean audio, and better in noise.
NOTE
run_stage fusion_gated || { note "Stage fusion_gated failed (logs/flix_v4_fusion_gated.log)."; exit 1; }
$PY -m scripts.flix_v4_eval --run fusion_gated="$(cat $S/fusion_gated.run)" \
    --tests film_test=data/flix_v4c/flix_test.csv ted_test=data/flix_v4c/ted_test.csv \
    --out "$R/test_results_wp.json" >> logs/flix_v4_eval.log 2>&1 || note "fusion_gated evaluation failed (logs/flix_v4_eval.log)"
PYTHONPATH=. $PY scripts/flix_v4_probe.py "$(cat $S/fusion_gated.run)" 1122 "$R/probe_fusion_gated.json" \
    >> logs/flix_v4_probe.log 2>&1 || note "probe failed (logs/flix_v4_probe.log)"
{ echo; echo "Fusion v3 (fusion_gated) test results, with fusion_wp and the twin for comparison:"; echo
  cat "$R/test_results_wp.md"; echo; echo "Lip-use probe for fusion_gated:"
  echo '```'; cat "$R/probe_fusion_gated.json"; echo '```'; } >> "$R/decision_log.md"
note "Fusion v3 finished."
touch "$R/FUSION4_DONE"

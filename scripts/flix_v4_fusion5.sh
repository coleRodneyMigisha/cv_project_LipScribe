#!/usr/bin/env bash
# Flix v4 fusion v3, warm start (15 Nov 2026): train fusion_gated_ws (starts as the audio-only twin),
# then test, probe and beam-search decode it with the LibriSpeech LM.
#   nohup scripts/flix_v4_fusion5.sh > logs/flix_v4_fusion5.log 2>&1 &
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

## Fusion v3, warm start (15 Nov 2026)

fusion_gated beat the twin at 0 dB and below, but trailed it on clean audio (TED 18.5 vs 17.4).
fusion_gated_ws keeps the symmetric reliability-gated fusion, but takes the twin's whole frozen
audio path as the audio stream. It starts with the gate 95 % open to audio, the one new layer as
the identity, and the twin's head. At step 0 it matches the twin (TED val sample 20.9 vs 20.8 %);
training then only has to learn where the lips help.
NOTE
run_stage fusion_gated_ws || { note "Stage fusion_gated_ws failed (logs/flix_v4_fusion_gated_ws.log)."; exit 1; }
$PY -m scripts.flix_v4_eval --run fusion_gated_ws="$(cat $S/fusion_gated_ws.run)" \
    --tests film_test=data/flix_v4c/flix_test.csv ted_test=data/flix_v4c/ted_test.csv \
    --out "$R/test_results_wp.json" >> logs/flix_v4_eval.log 2>&1 || note "fusion_gated_ws evaluation failed (logs/flix_v4_eval.log)"
PYTHONPATH=. $PY scripts/flix_v4_probe.py "$(cat $S/fusion_gated_ws.run)" 1122 "$R/probe_fusion_gated_ws.json" \
    >> logs/flix_v4_probe.log 2>&1 || note "probe failed (logs/flix_v4_probe.log)"
PYTHONPATH=. $PY scripts/flix_v4_beam.py dump --run gated_ws="$(cat $S/fusion_gated_ws.run)" >> logs/flix_v4_beam_dump.log 2>&1
PYTHONPATH=. $PY scripts/flix_v4_beam.py decode --names twin av visual gated gated_ws --lm lm/flix_libri_4gram.arpa \
    --out $R/beam_libri >> logs/flix_v4_beam_libri.log 2>&1 || note "beam decoding failed (logs/flix_v4_beam_libri.log)"
{ echo; echo "fusion_gated_ws test results (greedy), with the other models:"; echo; cat "$R/test_results_wp.md"
  echo; echo "Lip-use probe for fusion_gated_ws:"; echo '```'; cat "$R/probe_fusion_gated_ws.json"; echo '```'
  echo; echo "Beam search + LibriSpeech LM:"; echo; cat $R/beam_libri.md; } >> "$R/decision_log.md"
note "Fusion v3 warm start finished."
touch "$R/FUSION5_DONE"

#!/usr/bin/env bash
# Flix v5 (Alienware, autonomous): re-cut every clip with 0.32 s padding per side so audio, lips and
# label cover the same window, relabel from Whisper word timestamps, retrain the v4 chain on it,
# then score v4 and v5 models once on the old and the re-cut test sets.
#   setsid nohup bash scripts/flix_v5_run.sh >> logs/flix_v5.log 2>&1 < /dev/null &
# Resumable at every stage; scripts/flix_boot.sh restarts it after a reboot.
# The film clips are cut on the laptop (scripts/flix_v5_films_laptop.sh), which touches flix_v5/.films_done.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=asr_pkgs:.
export PYTHONUNBUFFERED=1                      # log lines (e.g. "Run directory") appear at once
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 # WavLM is cached; never wait on the network
R=reports/flix_v5; S=$R/state; mkdir -p "$S" logs configs/flix_v5
note() { echo "- $(date '+%Y-%m-%d %H:%M') $*" | tee -a "$R/decision_log.md"; }
fail() { note "$*"; touch "$R/FAILED"; exit 1; }  # FAILED: scripts/flix_watchdog.sh leaves it stopped
exec 9> "$R/.lock"                             # one instance only (boot hook, watchdog, by hand);
flock -n 9 || { echo "$(date '+%F %T') already running"; exit 0; }  # training children keep it
rm -f "$R/FAILED"                              # a (re)start clears an earlier stop
grepv() { grep --line-buffered -v "^W0000\|^I0000\|absl\|InitGoogle\|XNNPACK"; }

if [ ! -f "$R/decision_log.md" ]; then cat > "$R/decision_log.md" <<'NOTE'
# Flix v5 decision log

## Spec (written 15 Nov 2026, before any v5 run)

- **Why.** v4 clips are cut tightly around their words. On the TED test the audio-only twin gets
  the first / last word of a clip wrong 22 % of the time against 11 % for inner words, and
  widening the test audio by 0.32 s per side alone cut that to 19 % (edge check, 15 Nov).
- **Change (one).** Every clip's window grows by 8 frames per side. Mouth crops (same Auto-AVSR
  crop) and audio are cut from the same source for that window; the label is every Whisper
  large-v3-turbo word (context cut ±1.5 s, word timestamps) with ≥ 50 % of its duration inside
  the window. Clip ids, splits, model configs and training recipes stay as in v4.
- **Retrained chain.** visual_c → twin (fusion_wp_audio) → fusion_wp → warm start (fusion_gated_ws).
  audio_reg is kept: its audio and labels already matched.
- **Locked evaluation.** Test sets touched once, at the end: v4c test sets (old clips, old labels)
  and v5 test sets (re-cut clips, new labels; film test labels are now Whisper, not the
  human-checked subtitles). Greedy CTC, clean / 0 / −5 / −10 dB, scripts/flix_v5_eval.py.
- **Baseline.** The v4 twin and v4 warm start scored on the same v5 test sets.
- **Primary metric.** TED v5 test WER, warm start, clean and the mean over the four SNRs.
- **Expected.** Best: TED clean ≈ 10 %. Realistic: 12–13 %. Worst: no better than the v4 warm
  start on the v5 test set.
- **Success.** ws_v5 beats ws_v4 on the v5 TED test by ≥ 1 point (clean and SNR mean) with a
  95 % CI excluding 0, and the edge-word error falls below 1.5 × the inner-word error.
- **Stop rule.** One pass of the chain; no tuning runs without the user. A failed stage is logged
  and the pipeline stops.
NOTE
fi

stage_done() { [ -f "$S/$1.done" ]; }

# 1. jobs ------------------------------------------------------------------------------------
[ -f flix_v5/jobs_ted.csv ] || $PY -m scripts.flix_v5_recut jobs

# 2. Whisper relabel worker (GPU) runs alongside the cutting (CPU) -----------------------------
if ! stage_done relabel && ! pgrep -f "^$PY -m scripts.flix_v5_recut relabel" > /dev/null; then
  ($PY -m scripts.flix_v5_recut relabel >> logs/flix_v5_relabel.log 2>&1 &)
  note "Relabel worker started."
fi

# 3. TED cut -----------------------------------------------------------------------------------
if [ ! -f flix_v5/.ted_done ]; then
  note "TED re-cut started (flix_v5/jobs_ted.csv)."
  rm -rf ted/tmp_v5/*
  $PY -m scripts.flix_v5_recut ted --workers 12 --downloaders 4 2>&1 | grepv >> logs/flix_v5_ted.log
  [ -f flix_v5/.ted_done ] || fail "TED re-cut failed (logs/flix_v5_ted.log)."
  note "TED re-cut done: $(grep -c . flix_v5/talks_done.tsv) talks, $(grep -c '"ok"' flix_v5/crop_stats.jsonl) clips cropped."
fi

# 4. wait for the laptop's film clips and the relabel worker ------------------------------------
n=0
until [ -f flix_v5/.films_done ]; do
  [ $((n % 60)) -eq 0 ] && note "Waiting for the laptop's film clips (flix_v5/.films_done)."
  n=$((n + 1)); sleep 60
done
for try in 1 2 3; do                                      # a crashed worker resumes from its cache
  stage_done relabel && break
  while pgrep -f "^$PY -m scripts.flix_v5_recut relabel" > /dev/null; do sleep 60; done
  if grep -q "^DONE" logs/flix_v5_relabel.log 2>/dev/null; then
    touch "$S/relabel.done"; note "Relabel done: $(grep -c . flix_v5/whisper_words.jsonl) clips."; break
  fi
  note "Relabel worker not finished (try $try); restarting it."
  $PY -m scripts.flix_v5_recut relabel >> logs/flix_v5_relabel.log 2>&1
done
stage_done relabel || fail "Relabel failed three times (logs/flix_v5_relabel.log)."

# 5. manifests ---------------------------------------------------------------------------------
if ! stage_done manifests; then
  $PY -m scripts.flix_v5_recut manifests >> logs/flix_v5_manifests.log 2>&1 || fail "Manifests failed."
  note "Manifests written to data/flix_v5: $(tr -d '\n ' < data/flix_v5/recut_stats.json | cut -c1-600)"
  touch "$S/manifests.done"
fi

# 6. training chain ----------------------------------------------------------------------------
mkcfg() {  # <v4 name> -> configs/flix_v5/<name>.yaml with v5 data and the v5 runs it starts from
  $PY - "$1" <<'PYEOF'
import sys
from pathlib import Path
n = sys.argv[1]
t = Path(f"configs/flix_v4/{n}.yaml").read_text()
t = t.replace("data/flix_v3/", "data/flix_v5/").replace("data/flix_v4c/", "data/flix_v5/")
t = t.replace("experiment: flix_v4_", "experiment: flix_v5_")
old = {"visual_c": "runs/flix_v4/flix_v4_visual_c_20261113-233839",
       "fusion_wp_audio": "runs/flix_v4/flix_v4_fusion_wp_audio_20261114-142621",
       "fusion_wp": "runs/flix_v4/flix_v4_fusion_wp_20261114-110340"}
for k, p in old.items():
    if p in t:
        new = Path(f"reports/flix_v5/state/{k}.run").read_text().strip()
        t = t.replace(p, new)
body = "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("#"))  # comments may name old dirs
assert "data/flix_v4" not in body and "data/flix_v3" not in body and "flix_v4_visual_c_" not in body, n
Path(f"configs/flix_v5/{n}.yaml").write_text("# generated by scripts/flix_v5_run.sh from configs/flix_v4\n" + t)
PYEOF
}

run_once() {  # <name>: one training attempt of configs/flix_v5/<name>.yaml (resumes if it can)
  local n=$1 run
  run=$(cat "$S/$n.run" 2>/dev/null || true)
  if [ -n "$run" ] && [ -f "$run/latest.pt" ]; then
    note "Stage $n resumed from $run/latest.pt."
    $PY -m scripts.train --resume "$run/latest.pt" --no-results >> "logs/flix_v5_$n.log" 2>&1 || return 1
  else
    mkcfg "$n" || return 1
    rm -f "$S/$n.run"                          # never fall back on an earlier attempt's run
    note "Stage $n started (configs/flix_v5/$n.yaml)."
    local log0; log0=$(cat "logs/flix_v5_$n.log" 2>/dev/null | wc -l)  # only lines this attempt writes
    $PY -m scripts.train --config "configs/flix_v5/$n.yaml" --no-results --runs-root runs/flix_v5 \
        >> "logs/flix_v5_$n.log" 2>&1 &
    local pid=$!
    newrun() { tail -n +$((log0 + 1)) "logs/flix_v5_$n.log" | grep -o "Run directory: .*" | tail -1 | cut -d' ' -f3-; }
    for _ in $(seq 120); do                    # recorded early, so a crash mid-run can resume
      run=$(newrun); [ -n "$run" ] && [ -d "$run" ] && { echo "$run" > "$S/$n.run"; break; }; sleep 5
    done
    wait $pid || return 1
    run=$(newrun); [ -n "$run" ] && [ -d "$run" ] && echo "$run" > "$S/$n.run"
  fi
  [ -f "$S/$n.run" ] && [ -f "$(cat $S/$n.run)/best.pt" ] || { note "Stage $n: no best.pt for this attempt."; return 1; }
}

run_stage() {  # <name>: up to 3 attempts (a crash resumes from latest.pt), then the summary line
  local n=$1 try
  stage_done "$n" && return 0
  for try in 1 2 3; do
    run_once "$n" && break
    [ "$try" -lt 3 ] || return 1
    note "Stage $n attempt $try failed (logs/flix_v5_$n.log); retrying in 2 min."; sleep 120
  done
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

for st in visual_c fusion_wp_audio fusion_wp fusion_gated_ws; do
  run_stage $st || fail "Stage $st failed three times (logs/flix_v5_$st.log)."
done

# 7. one test evaluation (decoded hypotheses are cached in $R/hyps.json, so a retry continues) ----
if ! stage_done eval; then
  V4=reports/flix_v4/state
  for try in 1 2; do
    $PY scripts/flix_v5_eval.py --run twin_v4="$(cat $V4/fusion_wp_audio.run)" ws_v4="$(cat $V4/fusion_gated_ws.run)" \
        twin_v5="$(cat $S/fusion_wp_audio.run)" ws_v5="$(cat $S/fusion_gated_ws.run)" >> logs/flix_v5_eval.log 2>&1 && break
    [ "$try" -lt 2 ] || fail "Evaluation failed twice (logs/flix_v5_eval.log)."
    note "Evaluation attempt $try failed (logs/flix_v5_eval.log); retrying in 2 min."; sleep 120
  done
  { echo; echo "## Test results (greedy), $(date '+%Y-%m-%d %H:%M')"; echo; cat $R/test_results.md; } >> "$R/decision_log.md"
  touch "$S/eval.done"
fi
note "Flix v5 finished."
touch "$R/ALL_DONE"

#!/usr/bin/env bash
# Flix v4 on the Alienware, fully unattended and resumable (crontab @reboot via flix_boot.sh):
#   0  stop v3 for good (reports/flix_v3/STOPPED)
#   1  relabel TED + training clips with Whisper from each clip's own audio -> data/flix_v4
#   2  zero-shot scan of Auto-AVSR crop zoom on relabelled ted_val -> pick the best zoom
#   3  IN PARALLEL:
#      audio: WavLM Base+ fine-tune, then audio_reg (stronger regularisation); keep the lower val WER
#      visual: wait for the Auto-AVSR-style re-crop (scripts/flix_recrop.py: TED on this machine,
#              films shipped from the laptop) -> data/flix_v4c; stop the old-crop visual run (kept
#              as the baseline visual_old); fine-tune Auto-AVSR on the new crops (visual_c, no zoom)
#   4  fusion (Model P way, initialised from audio + visual_c, new crops)
#   5  test-set evaluation (new-crop test manifests; visual_old separately on the old crops)
# Every step appends to reports/flix_v4/decision_log.md. Markers live in reports/flix_v4/state/.
#   setsid nohup bash scripts/flix_v4_run.sh >> logs/flix_v4.log 2>&1 < /dev/null &
#   --audio-reg-only: run only audio_reg + the pick (used by scripts/flix_v4_handover.sh)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=reports/flix_v4; S=$R/state; C=$R/cfg; LOG=$R/decision_log.md
mkdir -p "$S" "$C" logs runs/flix_v4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/pretrained/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
rm -f "$R/FAILED"

note() {  # append a timestamped line to the decision log (and print it)
  if ! grep -q "^## Automatic pipeline log" "$LOG"; then
    printf '\n## Automatic pipeline log (scripts/flix_v4_run.sh)\n\n' >> "$LOG"
  fi
  echo "- $(date '+%F %H:%M') $*" | tee -a "$LOG"
}
fail() { note "**FAILED:** $*"; echo "$*" > "$R/FAILED"; exit 1; }

# ---- 0: stop v3 (never restarted: flix_boot.sh checks STOPPED)
if [ ! -f reports/flix_v3/STOPPED ]; then
  touch reports/flix_v3/STOPPED
  pkill -f "^bash scripts/flix_chain_run.sh" ; pkill -f "^$PY -m scripts.flix_sweep"
  pkill -f "^$PWD/.venv/bin/python -m scripts.train --config $PWD/configs/flix_v3/" ; sleep 20
  note "Stopped v3 (r00, r01 finished; r02_ft0.3 stopped mid-run). Its runs and logs stay in runs/flix_v3, reports/flix_v3."
fi

# ---- 1: relabel
if [ ! -f "$S/relabel.done" ]; then
  note "Relabelling with Whisper large-v3-turbo from each clip's own audio (TED cue text was misaligned)."
  PYTHONPATH=asr_pkgs:. $PY -m scripts.flix_v4_relabel >> logs/flix_v4_relabel.log 2>&1 || fail "relabel (logs/flix_v4_relabel.log)"
  touch "$S/relabel.done"
  note "Relabel done: $($PY -c "import json;d=json.load(open('data/flix_v4/relabel_stats.json'));print('; '.join(f\"{k}: {v['kept']}/{v['before']} kept, old-label WER vs Whisper {v['old_label_wer_vs_whisper']:.1%}\" for k,v in d.items()))")"
fi

# ---- 2: crop zoom for Auto-AVSR
if [ ! -f "$S/zoom" ]; then
  $PY -m scripts.flix_v4_zeroshot --manifests data/flix_v4/ted_val.csv data/flix_v4/film_val.csv \
      --splits data/flix_v4/splits.csv --zooms 1 0.85 0.7 0.6 --out "$R/zeroshot_autoavsr.json" \
      >> logs/flix_v4_zeroshot.log 2>&1 || fail "zero-shot scan (logs/flix_v4_zeroshot.log)"
  $PY - "$R/zeroshot_autoavsr.json" "$S/zoom" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
z = {}
for k, v in d.items():
    name, zoom = k.split("@zoom"); z.setdefault(zoom, []).append(v["cer"])
best = min(z, key=lambda k: sum(z[k]) / len(z[k]))
open(sys.argv[2], "w").write(best)
EOF
  note "Zero-shot Auto-AVSR (unmodified, greedy subword CTC) on relabelled val: $($PY -c "import json;d=json.load(open('$R/zeroshot_autoavsr.json'));print(', '.join(f'{k} WER {v[\"wer\"]:.1%}/CER {v[\"cer\"]:.1%}' for k,v in d.items()))"). Chosen zoom: $(cat $S/zoom)."
fi
ZOOM=$(cat "$S/zoom")

# configs for this run: data/flix_v4 and the chosen zoom (repo configs stay untouched)
mkcfg() {  # <name> [bs] [accum]
  $PY - "$1" "$ZOOM" "${2:-}" "${3:-}" <<'EOF'
import sys, yaml
name, zoom, bs, acc = sys.argv[1:]
c = yaml.safe_load(open(f"configs/flix_v4/{name}.yaml"))
d = c["data"]
newcrops = name in ("visual_c", "fusion")          # Auto-AVSR-style crops: data/flix_v4c, no zoom
dd = "flix_v4c" if newcrops else "flix_v4"
zoom = 1.0 if newcrops else zoom
for k in ("manifest", "splits"):
    d[k] = d[k].replace("flix_v3", dd)
d["val_manifests"] = {k: v.replace("flix_v3", dd) for k, v in d["val_manifests"].items()}
if "visual_encoder_cfg" in c["model"]:
    c["model"]["visual_encoder_cfg"]["zoom"] = float(zoom)
if bs:
    c["train"]["batch_size"], c["train"]["grad_accum"] = int(bs), int(acc)
if name == "fusion":
    st = "reports/flix_v4/state"
    c["model"]["init_from"] = {"audio_encoder": open(f"{st}/audio.run").read().strip() + "/best.pt",
                               "visual_encoder": open(f"{st}/visual_c.run").read().strip() + "/best.pt"}
yaml.safe_dump(c, open(f"reports/flix_v4/cfg/{name}.yaml", "w"), sort_keys=False)
EOF
}

train() {  # <name> [bs] [accum]: fresh, or resume the stage's run; returns train.py's exit code
  local n=$1 run
  run=$(cat "$S/$n.run" 2>/dev/null || true)
  if [ -n "$run" ] && [ -f "$run/latest.pt" ]; then
    $PY -m scripts.train --resume "$run/latest.pt" --no-results >> "logs/flix_v4_$n.log" 2>&1
  else
    mkcfg "$@" || return 1
    $PY -m scripts.train --config "$C/$n.yaml" --no-results --runs-root runs/flix_v4 >> "logs/flix_v4_$n.log" 2>&1 &
    local pid=$!
    for _ in $(seq 120); do
      run=$(grep -o "Run directory: .*" "logs/flix_v4_$n.log" | tail -1 | cut -d' ' -f3-)
      [ -n "$run" ] && [ -d "$run" ] && { echo "$run" > "$S/$n.run"; break; }; sleep 5
    done
    wait $pid
  fi
}

summary() {  # <name>: best epoch and val numbers from the run's history
  $PY - "$(cat $S/$1.run)" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] + "/history.csv")))
cols = [c for c in rows[0] if c.startswith(("wer_", "cer_"))]
key = "val_wer" if "audio" in sys.argv[1] or "fusion" in sys.argv[1] else "val_cer"
b = min(rows, key=lambda r: float(r[key]))
print(f"best epoch {b['epoch']}/{len(rows)}: mean val WER {float(b['val_wer']):.1%}, CER {float(b['val_cer']):.1%} ("
      + ", ".join(f"{c} {float(b[c]):.1%}" for c in cols) + ")")
EOF
}

stage() {  # <name>: train with one retry at half batch (OOM when both share the GPU)
  local n=$1
  [ -f "$S/$n.done" ] && return 0
  note "Stage $n started ($(grep -c . data/flix_v4/train_mix.csv) training rows)."
  if ! train "$n"; then
    if grep -q "out of memory" "logs/flix_v4_$n.log"; then
      note "Stage $n ran out of GPU memory; retrying with batch 4 x 4 accumulation."
      rm -f "$S/$n.run"
      while pgrep -f "^$PY -m scripts.train" > /dev/null; do sleep 120; done   # the other stage alone first
      train "$n" 4 4 || { note "Stage $n failed again (logs/flix_v4_$n.log)."; return 1; }
    else
      note "Stage $n crashed (logs/flix_v4_$n.log): $(grep -E 'Error|error' logs/flix_v4_$n.log | tail -1)"; return 1
    fi
  fi
  touch "$S/$n.done"
  note "Stage $n done: $(summary $n). Run: $(cat $S/$n.run)"
}

best_wer() {  # <run dir>: best mean val WER in its history
  $PY -c "import csv,sys;print(min(float(r['val_wer']) for r in csv.DictReader(open(sys.argv[1]+'/history.csv'))))" "$1"
}

pick_audio() {  # keep the better of audio / audio_reg in audio.run (fusion and eval read it)
  [ -f "$S/audio.pick.done" ] && return 0
  [ -f "$S/audio_v1.run" ] || cp "$S/audio.run" "$S/audio_v1.run"
  local v1 reg w1 w2
  v1=$(cat "$S/audio_v1.run"); reg=$(cat "$S/audio_reg.run" 2>/dev/null || true)
  if [ -f "$S/audio_reg.done" ] && [ -n "$reg" ]; then
    w1=$(best_wer "$v1"); w2=$(best_wer "$reg")
    if $PY -c "import sys;sys.exit(0 if float(sys.argv[2])<float(sys.argv[1]) else 1)" "$w1" "$w2"; then
      echo "$reg" > "$S/audio.run"; note "Audio model for fusion: audio_reg (val WER $w2 vs first run $w1)."
    else
      echo "$v1" > "$S/audio.run"; note "Audio model for fusion: first run kept (val WER $w1 vs audio_reg $w2)."
    fi
  else
    note "audio_reg did not finish; the first audio run is used for fusion."
  fi
  touch "$S/audio.pick.done"
}

audio_branch() {
  stage audio || return 1
  stage audio_reg || true          # a failed retrain never blocks the pipeline
  pick_audio
}

if [ "${1:-}" = "--audio-reg-only" ]; then
  [ -f "$S/audio.done" ] || fail "--audio-reg-only needs stage audio done"
  stage audio_reg || true; pick_audio; exit 0
fi

ted_done() {  # every talk of flix_avsr/jobs_ted.csv is in flix_avsr/talks_done.tsv
  $PY - <<'EOF' 2>/dev/null
import csv, sys
need = {r["title"].split("/", 1)[1] for r in csv.DictReader(open("flix_avsr/jobs_ted.csv"))}
done = {l.split("\t")[0] for l in open("flix_avsr/talks_done.tsv")}
sys.exit(0 if need <= done else 1)
EOF
}
recropping() { pgrep -f "^.venv/bin/python -m scripts.flix_recrop ted" > /dev/null; }

recrop_wait() {  # (re)start the TED re-crop if it is not running, wait for both halves, build data/flix_v4c
  [ -f "$S/recrop.done" ] && return 0
  until [ -f flix_avsr/.films_done ] && ! recropping && ted_done; do
    if ! recropping && ! ted_done && [ -f flix_avsr/jobs_ted.csv ]; then
      (PYTHONPATH=asr_pkgs:. setsid nohup $PY -m scripts.flix_recrop ted --out flix_avsr --workers 12 --downloaders 4 \
          >> logs/flix_recrop_ted.log 2>&1 < /dev/null &)
      note "TED re-crop (re)started."
    fi
    sleep 300
  done
  $PY -m scripts.flix_recrop manifests --out flix_avsr --data data/flix_v4 --dst data/flix_v4c > logs/flix_recrop_manifests.log 2>&1 \
    || { note "building data/flix_v4c failed (logs/flix_recrop_manifests.log)"; return 1; }
  touch "$S/recrop.done"
  note "Re-crop done (Auto-AVSR alignment, $(grep -c '"ok"' flix_avsr/recrop_stats.jsonl) TED + $(grep -c '"ok"' flix_avsr/film_recrop_stats.jsonl) film clips ok): $(tr '\n' ';' < logs/flix_recrop_manifests.log)"
}

stop_old_visual() {  # the old-crop visual run becomes the baseline visual_old
  [ -f "$S/visual_old.run" ] && return 0
  local pid ppid
  for pid in $(pgrep -f "^.venv/bin/python -m scripts.train --config reports/flix_v4/cfg/visual.yaml"); do
    ppid=$(ps -o ppid= -p "$pid" | tr -d ' ')
    [ -n "$ppid" ] && [ "$ppid" != "$$" ] && ps -o cmd= -p "$ppid" | grep -q "flix_v4_run.sh" && kill "$ppid" 2>/dev/null
    kill "$pid" 2>/dev/null
  done
  while pgrep -f "^.venv/bin/python -m scripts.train --config reports/flix_v4/cfg/visual.yaml" > /dev/null; do sleep 10; done
  if [ -f "$S/visual.run" ]; then
    cp "$S/visual.run" "$S/visual_old.run"
    note "Old-crop visual run stopped as the baseline visual_old: $(summary visual). Run: $(cat $S/visual.run)"
  else
    touch "$S/visual_old.run"
  fi
}

audio_side() {  # scripts/flix_v4_handover.sh may still be doing audio_reg for the old orchestrator
  while pgrep -f "^bash scripts/flix_v4_handover.sh" > /dev/null; do sleep 60; done
  audio_branch
}

visual_side() {
  recrop_wait || return 1
  stop_old_visual
  stage visual_c
}

# ---- 3: audio (+ audio_reg) in parallel with re-crop -> visual_c
audio_side > /dev/null 2>&1 & PA=$!
visual_side > /dev/null 2>&1 & PV=$!
wait $PA; A=$?; wait $PV; V=$?
[ -f "$S/audio.pick.done" ] && [ -f "$S/visual_c.done" ] || fail "stage 3 (audio exit $A, visual exit $V)"

# ---- 4: fusion
stage fusion || fail "fusion stage"

# ---- 5: test sets
if [ ! -f "$S/eval.done" ]; then
  extra=(); [ -f "$S/audio_v1.run" ] && [ "$(cat $S/audio_v1.run)" != "$(cat $S/audio.run)" ] && extra=(audio_first="$(cat $S/audio_v1.run)")
  $PY -m scripts.flix_v4_eval --run audio="$(cat $S/audio.run)" visual="$(cat $S/visual_c.run)" \
      fusion="$(cat $S/fusion.run)" "${extra[@]}" \
      --tests film_test=data/flix_v4c/flix_test.csv ted_test=data/flix_v4c/ted_test.csv \
      >> logs/flix_v4_eval.log 2>&1 || fail "evaluation (logs/flix_v4_eval.log)"
  if [ -s "$S/visual_old.run" ]; then         # baseline on the crops it was trained on
    $PY -m scripts.flix_v4_eval --run visual_old="$(cat $S/visual_old.run)" --out "$R/test_results_visual_old.json" \
        >> logs/flix_v4_eval.log 2>&1 || note "visual_old evaluation failed (logs/flix_v4_eval.log)"
  fi
  touch "$S/eval.done"
  note "Test results (greedy CTC) written to $R/test_results.md:"
  { echo; cat "$R/test_results.md"; echo; [ -f "$R/test_results_visual_old.md" ] && { echo "Old-crop baseline:"; echo; cat "$R/test_results_visual_old.md"; echo; }; } >> "$LOG"
fi
touch "$R/ALL_DONE"
note "Flix v4 pipeline finished."

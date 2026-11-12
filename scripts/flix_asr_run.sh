#!/usr/bin/env bash
# X6 'Flix' v3: Whisper subtitles for the films without subtitles (Alienware).
# Waits for asr_audio/.ready (sent by the laptop), transcribes, marks asr_srt/.done.
#   nohup setsid bash scripts/flix_asr_run.sh > logs/flix_asr.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
until [ -f asr_audio/.ready ]; do sleep 60; done
echo "== $(date +%T) transcribing $(ls asr_audio/*.ogg | wc -l) films"
PYTHONPATH=asr_pkgs .venv/bin/python -m scripts.flix_asr --audio-dir asr_audio --out asr_srt
touch asr_srt/.done
echo "== $(date +%T) done"

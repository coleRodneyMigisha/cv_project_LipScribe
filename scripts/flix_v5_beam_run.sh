#!/usr/bin/env bash
# F-04 (17 Nov 2026): dump twin_v5 / ws_v5 / ws_v6 (+ lips_v5) on the GPU, then tune and decode on the CPU.
#   nohup setsid scripts/flix_v5_beam_run.sh > logs/flix_v5_beam.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
exec 9> reports/flix_v5/.beam.lock; flock -n 9 || { echo "already running"; exit 1; }
rm -f reports/flix_v5/BEAM_DONE reports/flix_v5/BEAM_FAILED
fail() { echo "FAILED: $*"; echo "$*" > reports/flix_v5/BEAM_FAILED; exit 1; }
for i in 1 2; do
  .venv/bin/python scripts/flix_v5_beam.py dump --run \
    twin_v5=runs/flix_v5/flix_v5_fusion_wp_audio_20261116-000004 \
    ws_v5=runs/flix_v5/flix_v5_fusion_gated_ws_20261116-093641 \
    ws_v6=runs/flix_v6/flix_v6_fusion_gated_ws_20261116-105310 && break
  [ $i = 2 ] && fail "dump"
done
.venv/bin/python scripts/flix_v5_beam.py decode --workers 24 || fail "decode"
touch reports/flix_v5/BEAM_DONE; echo "ALL DONE $(date)"

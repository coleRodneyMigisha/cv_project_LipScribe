#!/usr/bin/env bash
# X6 'Flix' laptop side: wait for a DIRECT Tailscale link to the Alienware (e.g. on the
# campus network; the relay runs at ~0.1 MB/s, ~6 days for 58 GB), copy the films/series
# (resumable), then start scripts/flix_full_run.sh there.
#   nohup setsid bash scripts/flix_transfer.sh > ~/flix_mine/transfer.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=${SRC:-$HOME/Videos/entertainment}
IP=${IP:-100.115.172.88}            # Alienware's Tailscale address
HOST=alienware
DST=Desktop/lipscribe_flix

direct() { tailscale ping -c 1 --timeout 10s "$IP" 2>&1 | grep -q "via [0-9]"; }
until direct; do echo "$(date '+%F %T') no direct link yet"; sleep 600; done
echo "$(date '+%F %T') direct link up, copying $(du -sh "$SRC" | cut -f1)"
rsync -a scripts/mine_flix.py scripts/flix_prepare.py scripts/flix_sweep.py scripts/flix_sweep_report.py \
      scripts/flix_full_run.sh "$HOST:$DST/scripts/"
until rsync -a --partial --timeout=300 "$SRC/" "$HOST:$DST/videos/"; do
  echo "$(date '+%F %T') rsync interrupted, retrying in 60 s"; sleep 60
done
echo "$(date '+%F %T') copy complete, starting the full-set run"
ssh -n "$HOST" "cd $DST && (nohup setsid bash scripts/flix_full_run.sh > logs/flix_full.log 2>&1 < /dev/null &)"
echo "$(date '+%F %T') launched"

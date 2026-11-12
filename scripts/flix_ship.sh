#!/usr/bin/env bash
# X6 'Flix' laptop side: sync mined clips to the Alienware while mining runs, then
# start the Alienware stage (scripts/flix_run.sh) once the miner exits.
#   bash scripts/flix_ship.sh [~/flix_mine] [budget_min]
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=${1:-$HOME/flix_mine}
BUDGET=${2:-25}
HOST=alienware
DST=Desktop/lipscribe_flix
FILES=(scripts/mine_flix.py scripts/flix_prepare.py scripts/flix_eval.py scripts/flix_report.py
       scripts/flix_run.sh configs/flix_finetune.yaml)

sync_clips() {
  rsync -a --partial "$SRC/clips" "$SRC/manifest_local.csv" "$HOST:$DST/flix/" 2>&1 | grep -v Warning
}
while pgrep -f "mine_flix --out" > /dev/null; do
  sync_clips; echo "$(date +%T) synced $(ls "$SRC/clips" | grep -c npz) clips"; sleep 300
done
sleep 5
sync_clips
[ -f "$SRC/splits.csv" ] && rsync -a "$SRC/splits.csv" "$HOST:$DST/flix/"
rsync -aR "${FILES[@]}" "$HOST:$DST/" 2>&1 | grep -v Warning
echo "$(date +%T) miner finished; $(($(wc -l < "$SRC/manifest_local.csv") - 1)) clips shipped, starting Alienware stage"
ssh "$HOST" "cd $DST && mkdir -p logs && BUDGET=$BUDGET nohup setsid bash scripts/flix_run.sh > logs/flix_run.log 2>&1 < /dev/null &" 2>&1 | grep -v Warning
# follow until the stage finishes so whoever launched this gets one exit notification
while true; do
  sleep 60
  st=$(ssh "$HOST" "cd $DST; ls reports/flix/DONE reports/flix/FAILED 2>/dev/null; tail -1 logs/flix_run.log" 2>/dev/null)
  case "$st" in
    *DONE*) echo "$(date +%T) Alienware stage DONE"; exit 0 ;;
    *FAILED*) echo "$(date +%T) Alienware stage FAILED"; echo "$st"; exit 1 ;;
  esac
done

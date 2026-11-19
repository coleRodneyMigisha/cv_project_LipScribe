#!/usr/bin/env bash
# X6 'Flix' v2, laptop side (when the network is too slow to copy the films): mine the given
# training titles here and ship only the finished clips to the Alienware, deleting each local
# clip once copied (low disk). At the end the laptop's manifest goes over as
# flix_v2/manifest_laptop.csv and flix_v2/.mined_done tells flix_v2b_run.sh to start.
#   nohup setsid bash scripts/flix_mine_laptop.sh > ~/flix_mine/v2_mine.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-$HOME/flix_mine/v2}
TITLES=${TITLES:-"the_chosen his_hers a_haunting_in_venice malcolm_marie casino_royale"}
HOST=alienware; DST=Desktop/lipscribe_flix/flix_v2
mkdir -p "$OUT/clips" "$OUT/subs"; cp -n "$HOME/flix_mine/subs/"* "$OUT/subs/" 2>/dev/null
ssh -n "$HOST" "mkdir -p $DST/clips"
ship() {   # clips untouched for 2 min (never a file still being written)
  ( cd "$OUT/clips" && find . -maxdepth 1 -type f -mmin +2 -printf '%P\n' > "$OUT/.ship" )
  [ -s "$OUT/.ship" ] && rsync -a --timeout=300 --remove-source-files --files-from="$OUT/.ship" "$OUT/clips/" "$HOST:$DST/clips/"
}
.venv/bin/python -m scripts.mine_flix --out "$OUT" --only $TITLES --uid-by-time \
    --skip-manifest "$HOME/flix_mine/manifest_local.csv" "$OUT/manifest_local.csv" \
    --per-movie 100000 --per-episode 100000 --workers "${WORKERS:-3}" \
    --deadline "$(date -d '-1 min' +%H:%M)" > "$OUT/mine.log" 2>&1 &
MPID=$!
while kill -0 $MPID 2>/dev/null; do sleep 300; ship; echo "$(date +%T) $(grep -c . "$OUT/manifest_local.csv") manifest rows"; done
sleep 130; until ship || [ ! -s "$OUT/.ship" ]; do sleep 60; done
ship
until rsync -a --timeout=300 "$OUT/manifest_local.csv" "$HOST:$DST/manifest_laptop.csv"; do sleep 60; done
ssh -n "$HOST" "touch $DST/.mined_done"
echo "$(date +%T) done: $(grep -c . "$OUT/manifest_local.csv") rows; $(grep -E 'DONE|stopping' "$OUT/mine.log" | tail -1)"

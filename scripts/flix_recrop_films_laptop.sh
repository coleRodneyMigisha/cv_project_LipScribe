#!/usr/bin/env bash
# Flix v4 re-crop, laptop side: crop the film clips (films live only on the laptop), ship the
# crops to the Alienware's flix_avsr/clips, then mark flix_avsr/.films_done there. Resumable:
# re-running skips cropped clips; rsync retries until the network allows.
#   setsid nohup bash scripts/flix_recrop_films_laptop.sh > ~/flix_mine/avsr/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$HOME/flix_mine/avsr; HOST=alienware; DST=Desktop/lipscribe_flix/flix_avsr
.venv/bin/python -m scripts.flix_recrop films --out "$OUT" --jobs "$OUT/jobs_film.csv" --workers "${WORKERS:-2}" \
    2>&1 | grep --line-buffered -v "^W0000\|^I0000\|INFO: Created\|absl\|InitGoogle"
cp "$OUT/recrop_stats.jsonl" "$OUT/clips/../film_recrop_stats.jsonl"
until rsync -a --partial --timeout=120 "$OUT/clips/" "$HOST:$DST/clips/" \
      && rsync -a --timeout=120 "$OUT/film_recrop_stats.jsonl" "$HOST:$DST/" \
      && ssh -o ConnectTimeout=30 "$HOST" "touch $DST/.films_done"; do
  echo "$(date '+%F %T') ship failed, retrying in 5 min"; sleep 300
done
echo "$(date '+%F %T') films shipped"

#!/usr/bin/env bash
# Flix v5, laptop side: cut the film clips (films live only on the laptop) with scripts.flix_v5_recut,
# ship clips + context audio to the Alienware's flix_v5/, then touch flix_v5/.films_done there.
# Resumable: re-running skips finished clips; shipping retries until the network allows.
#   setsid nohup bash scripts/flix_v5_films_laptop.sh > ~/flix_mine/v5/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$HOME/flix_mine/v5; HOST=alienware; DST=Desktop/lipscribe_flix/flix_v5
mkdir -p "$OUT"
until scp -q "$HOST:$DST/jobs_film.csv" "$OUT/jobs_film.csv"; do sleep 300; done
.venv/bin/python -m scripts.flix_v5_recut films --out "$OUT" --jobs "$OUT/jobs_film.csv" --workers "${WORKERS:-2}" \
    2>&1 | grep --line-buffered -v "^W0000\|^I0000\|INFO: Created\|absl\|InitGoogle\|XNNPACK"
until rsync -a --partial --timeout=120 "$OUT/clips/" "$HOST:$DST/clips/" \
      && rsync -a --partial --timeout=120 "$OUT/ctx/" "$HOST:$DST/ctx/" \
      && rsync -a --timeout=120 "$OUT/crop_stats.jsonl" "$HOST:$DST/film_crop_stats.jsonl" \
      && ssh -o ConnectTimeout=30 "$HOST" "touch $DST/.films_done"; do
  echo "$(date '+%F %T') ship failed, retrying in 5 min"; sleep 300
done
echo "$(date '+%F %T') films shipped"

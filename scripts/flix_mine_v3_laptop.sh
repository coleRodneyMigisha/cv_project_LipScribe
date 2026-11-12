#!/usr/bin/env bash
# X6 'Flix' v3, laptop side: after the batch-1 mining (flix_mine_laptop.sh) has finished and the
# Alienware has written Whisper subtitles (asr_srt/.done), mine the films without subtitles
# using those subtitles, ship the clips to flix_v3/ and mark flix_v3/.mined_done.
#   nohup setsid bash scripts/flix_mine_v3_laptop.sh > ~/flix_mine/v3_mine.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$HOME/flix_mine/v3; HOST=alienware; DST=Desktop/lipscribe_flix
LIST=$HOME/flix_mine/asr_films.txt
mkdir -p "$OUT/clips" "$OUT/subs" "$OUT/srt"
while pgrep -f "^bash scripts/flix_mine_laptop.sh" >/dev/null; do sleep 120; done
echo "$(date +%T) batch-1 mining finished; waiting for Whisper subtitles"
until ssh -n "$HOST" "test -f $DST/asr_srt/.done"; do sleep 120; done
rsync -a "$HOST:$DST/asr_srt/" "$OUT/srt/"
: > "$OUT/extra.txt"
while IFS='|' read -r key rel; do
  [ -f "$OUT/srt/$key.srt" ] && echo "$key|$rel|$OUT/srt/$key.srt" >> "$OUT/extra.txt"
done < "$LIST"
echo "$(date +%T) mining $(wc -l < "$OUT/extra.txt") Whisper-labelled films"
ssh -n "$HOST" "mkdir -p $DST/flix_v3/clips"
ship() {
  ( cd "$OUT/clips" && find . -maxdepth 1 -type f -mmin +2 -printf '%P\n' > "$OUT/.ship" )
  [ -s "$OUT/.ship" ] && rsync -a --timeout=300 --remove-source-files --files-from="$OUT/.ship" "$OUT/clips/" "$HOST:$DST/flix_v3/clips/"
}
.venv/bin/python -m scripts.mine_flix --out "$OUT" --extra-titles "$OUT/extra.txt" --only $(cut -d'|' -f1 "$OUT/extra.txt") \
    --uid-by-time --skip-manifest "$OUT/manifest_local.csv" --per-movie 100000 --per-episode 100000 \
    --workers "${WORKERS:-3}" --deadline "$(date -d '-1 min' +%H:%M)" > "$OUT/mine.log" 2>&1 &
MPID=$!
while kill -0 $MPID 2>/dev/null; do sleep 300; ship; echo "$(date +%T) $(grep -c . "$OUT/manifest_local.csv") rows"; done
sleep 130; ship; ship
until rsync -a --timeout=300 "$OUT/manifest_local.csv" "$HOST:$DST/flix_v3/manifest_laptop.csv"; do sleep 60; done
ssh -n "$HOST" "touch $DST/flix_v3/.mined_done"
echo "$(date +%T) done: $(grep -E 'DONE|stopping' "$OUT/mine.log" | tail -1)"

#!/usr/bin/env bash
# X6 'Flix' v2 laptop side: copy only the titles the miner uses, in batches, with the audio
# pre-extracted (16 kHz mono FLAC) so the Alienware needs no ffmpeg. After each batch a marker
# videos/.<batch>_done is written on the Alienware; flix_v2_run.sh mines what is there.
#   nohup setsid bash scripts/flix_transfer2.sh > ~/flix_mine/transfer2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=${SRC:-$HOME/Videos/entertainment}
LIST=${LIST:-$HOME/flix_mine}
AUD=$HOME/flix_mine/audio16k
HOST=alienware; DST=Desktop/lipscribe_flix
mkdir -p "$AUD"
ssh -n "$HOST" "mkdir -p $DST/videos $DST/audio16k $DST/flix_v2/subs"
rsync -a "$HOME/flix_mine/subs/" "$HOST:$DST/flix_v2/subs/"          # embedded subtitle tracks
retry() { until "$@"; do echo "$(date '+%F %T') retry: $*" | cut -c1-120; sleep 60; done; }
for batch in batch1 batch2; do
  echo "$(date '+%F %T') $batch: extracting audio"
  while IFS= read -r rel; do
    case "$rel" in *.srt) continue ;; esac
    stem=$(basename "${rel%.*}" | sed 's/[^A-Za-z0-9_]\+/_/g')
    [ -f "$AUD/$stem.flac" ] || ffmpeg -nostdin -v error -y -i "$SRC/$rel" -vn -ac 1 -ar 16000 -c:a flac "$AUD/$stem.flac"
  done < "$LIST/$batch.txt"
  echo "$(date '+%F %T') $batch: copying audio"
  retry rsync -a --partial --timeout=300 --remove-source-files "$AUD/" "$HOST:$DST/audio16k/"
  echo "$(date '+%F %T') $batch: copying $(grep -vc '\.srt$' "$LIST/$batch.txt") videos"
  retry rsync -a --partial --timeout=300 --files-from="$LIST/$batch.txt" "$SRC/" "$HOST:$DST/videos/"
  ssh -n "$HOST" "touch $DST/videos/.${batch}_done"
  echo "$(date '+%F %T') $batch done"
done

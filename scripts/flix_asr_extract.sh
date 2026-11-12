#!/usr/bin/env bash
# extract 16 kHz mono Opus soundtracks for Whisper, ship them to the Alienware, delete locally
V=$HOME/Videos/entertainment; O=$HOME/flix_mine/asr_audio
while IFS='|' read -r key rel; do
  [ -f "$O/$key.ogg" ] || nice -n 10 ffmpeg -nostdin -v error -y -i "$V/$rel" -vn -ac 1 -ar 16000 -c:a libvorbis -q:a 2 "$O/$key.ogg" && echo "$(date +%T) $key $(du -h "$O/$key.ogg" | cut -f1)"
done < $HOME/flix_mine/asr_films.txt
ssh -n alienware "mkdir -p Desktop/lipscribe_flix/asr_audio"
until rsync -a --timeout=300 --remove-source-files "$O/" alienware:Desktop/lipscribe_flix/asr_audio/; do sleep 60; done
ssh -n alienware "touch Desktop/lipscribe_flix/asr_audio/.ready"; echo "$(date +%T) shipped"

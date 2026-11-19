#!/usr/bin/env bash
# One-off (13 Nov 2026): add the audio_reg stage to the flix_v4_run.sh instance that is already
# running. That instance was started before audio_reg existed and would go straight to fusion.
#   1 wait for its audio stage to finish
#   2 pause the orchestrator (SIGSTOP; the visual training keeps running in its own process)
#   3 run audio_reg + the pick with the updated script (writes state/audio.run)
#   4 resume the orchestrator; when visual is done it starts fusion with the chosen audio model
# After a reboot this is not needed: the updated flix_v4_run.sh includes audio_reg itself.
#   setsid nohup bash scripts/flix_v4_handover.sh <main pid> <audio stage pid> >> logs/flix_v4.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
MAIN=$1; AUDIO=$2
while kill -0 "$AUDIO" 2>/dev/null; do sleep 60; done
[ -f reports/flix_v4/state/audio.done ] || { echo "audio stage ended without audio.done; not handing over"; exit 1; }
kill -STOP "$MAIN" && echo "$(date '+%F %T') paused orchestrator $MAIN; running audio_reg"
bash scripts/flix_v4_run.sh --audio-reg-only
kill -CONT "$MAIN" && echo "$(date '+%F %T') resumed orchestrator $MAIN"

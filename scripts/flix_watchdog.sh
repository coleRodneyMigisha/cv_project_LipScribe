#!/usr/bin/env bash
# Flix watchdog (Alienware, cron every 20 min): restart a Flix pipeline whose process died without
# finishing (no ALL_DONE) and without stopping on purpose (no FAILED, written by its fail()).
#   crontab:  */20 * * * * bash $HOME/Desktop/lipscribe_flix/scripts/flix_watchdog.sh
# Each pipeline resumes where it stopped and takes a lock, so a second copy can never run.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
for v in 5 6; do
  s=scripts/flix_v${v}_run.sh; r=reports/flix_v$v
  [ -f "$s" ] || continue
  if [ -f "$r/ALL_DONE" ] || [ -f "$r/FAILED" ]; then continue; fi
  pgrep -f "^bash $s" > /dev/null && continue
  echo "$(date '+%F %T') restarting $s (no process, not finished, not stopped)" >> logs/flix_watchdog.log
  (setsid nohup bash "$s" >> "logs/flix_v$v.log" 2>&1 < /dev/null &)
done

#!/usr/bin/env bash
# X6 'Flix' on the Alienware: restart the long-running jobs after a reboot (crontab @reboot).
# Each job resumes where it stopped; nothing is started twice.
#   crontab:  @reboot bash $HOME/Desktop/lipscribe_flix/scripts/flix_boot.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
exec >> logs/boot.log 2>&1
sleep "${BOOT_DELAY:-180}"                     # network, GPU driver, filesystems
echo "== $(date '+%F %T') boot check"
if [ ! -f flix_ted/.harvest_done ] && ! pgrep -f "^.venv/bin/python -u -m scripts.ted_harvest" > /dev/null; then
  rm -rf ted/tmp/*                             # half-downloaded talks are fetched again
  (PYTHONPATH=asr_pkgs setsid nohup .venv/bin/python -u -m scripts.ted_harvest --out flix_ted --tmp ted/tmp \
      --hours 400 --workers 12 >> logs/ted_harvest.log 2>&1 < /dev/null &)
  echo "started TED harvester"
fi
if [ ! -f reports/flix_v3/ALL_DONE ] && ! pgrep -f "^bash scripts/flix_chain_run.sh" > /dev/null; then
  (setsid nohup bash scripts/flix_chain_run.sh >> logs/flix_chain.log 2>&1 < /dev/null &)
  echo "started flix chain"
fi

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
if [ ! -f reports/flix_v3/ALL_DONE ] && [ ! -f reports/flix_v3/STOPPED ] && ! pgrep -f "^bash scripts/flix_chain_run.sh" > /dev/null; then
  (setsid nohup bash scripts/flix_chain_run.sh >> logs/flix_chain.log 2>&1 < /dev/null &)
  echo "started flix chain"
fi
if [ -f reports/flix_v3/STOPPED ] && [ ! -f reports/flix_v4/ALL_DONE ] && ! pgrep -f "^bash scripts/flix_v4_run.sh" > /dev/null; then
  (setsid nohup bash scripts/flix_v4_run.sh >> logs/flix_v4.log 2>&1 < /dev/null &)
  echo "started flix v4 pipeline"
fi
if [ -f flix_v5/jobs_ted.csv ] && [ ! -f reports/flix_v5/ALL_DONE ] && [ ! -f reports/flix_v5/FAILED ] && ! pgrep -f "^bash scripts/flix_v5_run.sh" > /dev/null; then
  rm -rf ted/tmp_v5/*
  (setsid nohup bash scripts/flix_v5_run.sh >> logs/flix_v5.log 2>&1 < /dev/null &)
  echo "started flix v5 pipeline"
fi
if [ -f scripts/flix_v6_run.sh ] && [ ! -f reports/flix_v6/ALL_DONE ] && [ ! -f reports/flix_v6/FAILED ] && ! pgrep -f "^bash scripts/flix_v6_run.sh" > /dev/null; then
  (setsid nohup bash scripts/flix_v6_run.sh >> logs/flix_v6.log 2>&1 < /dev/null &)   # waits for v5 itself
  echo "started flix v6 pipeline"
fi
# web demo (backend/README.md): runs from the clean checkout ~/Desktop/lipscribe_demo; backend/run_local.sh starts
# the backend on 127.0.0.1:8800 and the public tunnel (address in that folder's .demo_url)
D="$HOME/Desktop/lipscribe_demo"
[ -x "$D/backend/run_local.sh" ] && bash "$D/backend/run_local.sh"

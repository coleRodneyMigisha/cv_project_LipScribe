#!/usr/bin/env bash
# Keep the LipScribe demo up on the Alienware: the backend on 127.0.0.1:8800 and a public
# Cloudflare quick tunnel to it (backend/README.md). Safe to run any number of times: it starts only
# what is not running. Run at boot (scripts/flix_boot.sh) and every 5 minutes from cron.
#
#   bash backend/run_local.sh            start what is missing, print the share link
#   bash backend/run_local.sh stop       stop the backend and the tunnel
#   touch .demo_no_tunnel              keep the backend private (no tunnel)
#
# Files (not in git): .demo_code (access code), .demo_url (current tunnel address), logs/demo_*.pid
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
PORT=8800
CF="$HOME/bin/cloudflared"
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

if [ "${1:-}" = "stop" ]; then
  for p in logs/demo_tunnel.pid logs/demo_server.pid; do alive $p && kill "$(cat $p)"; rm -f $p; done
  echo "stopped"; exit 0
fi

[ -f .demo_code ] || python3 -c "import secrets; print(secrets.token_urlsafe(9))" > .demo_code

if ! alive logs/demo_server.pid; then
  DEMO_CODE="$(cat .demo_code)" PYTHONPATH=.:asr_pkgs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 setsid nohup \
    .venv/bin/python -m uvicorn backend.server:app --host 127.0.0.1 --port $PORT --ws-max-size 16777216 \
    >> logs/demo_server.log 2>&1 < /dev/null &
  echo $! > logs/demo_server.pid
  echo "$(date '+%F %T') started backend (pid $!)"
fi

if [ ! -f .demo_no_tunnel ] && [ -x "$CF" ] && ! alive logs/demo_tunnel.pid; then
  setsid nohup "$CF" tunnel --no-autoupdate --url http://127.0.0.1:$PORT > logs/demo_tunnel.log 2>&1 < /dev/null &
  echo $! > logs/demo_tunnel.pid
  for _ in $(seq 30); do
    u=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" logs/demo_tunnel.log | head -1)
    [ -n "$u" ] && break; sleep 2
  done
  echo "$u" > .demo_url
  echo "$(date '+%F %T') started tunnel: $u"
fi

if [ -t 1 ] && [ -s .demo_url ]; then
  u=$(cat .demo_url); c=$(cat .demo_code)
  echo "page:        $u/?code=$c"
  echo "Pages site:  https://lipscribe.pages.dev/?backend=${u/https:/wss:}/ws/live&code=$c"
fi

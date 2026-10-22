#!/usr/bin/env bash
# Local stack: ccgw first, then the LiteLLM proxy in front of it.
#   scripts/dev.sh start|stop|restart|status
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f .env ] || cp env.example .env
set -a; source .env; set +a
PY="$ROOT/.venv/bin/python"
mkdir -p logs run

# Start a command in its own session (macOS has no setsid binary) so it outlives
# the terminal or tool shell that launched it.
detach() { nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' "$@" </dev/null & }

pid_alive() { [ -f "run/$1.pid" ] && kill -0 "$(cat "run/$1.pid")" 2>/dev/null; }

wait_http() {  # name url
  for _ in $(seq 1 60); do
    curl -fsS -o /dev/null "$2" 2>/dev/null && { echo "  $1 up: $2"; return 0; }
    sleep 1
  done
  echo "  $1 did not come up; see logs/$1.log" >&2; return 1
}

start() {
  [ -x "$PY" ] || { echo "missing .venv; run: uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt" >&2; exit 1; }
  if pid_alive ccgw; then echo "  ccgw already running"; else
    detach "$PY" -m uvicorn ccgw.app:app --host 127.0.0.1 --port "$CCGW_PORT" >>logs/ccgw.log 2>&1
    echo $! > run/ccgw.pid
  fi
  wait_http ccgw "http://127.0.0.1:$CCGW_PORT/health"
  if pid_alive litellm; then echo "  litellm already running"; else
    detach "$ROOT/.venv/bin/litellm" --config litellm/config.yaml --host "$LITELLM_HOST" --port "$LITELLM_PORT" >>logs/litellm.log 2>&1
    echo $! > run/litellm.pid
  fi
  wait_http litellm "http://127.0.0.1:$LITELLM_PORT/health/liveliness"
  echo
  echo "Point callers at:  base URL http://127.0.0.1:$LITELLM_PORT  key $LITELLM_MASTER_KEY  model claude-code"
}

stop() {
  for n in litellm ccgw; do
    if pid_alive "$n"; then kill "$(cat "run/$n.pid")" && echo "  stopped $n"; else echo "  $n not running"; fi
    rm -f "run/$n.pid"
  done
}

status() {
  for n in ccgw litellm; do
    if pid_alive "$n"; then echo "  $n running (pid $(cat "run/$n.pid"))"; else echo "  $n stopped"; fi
  done
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 1; start ;;
  status) status ;;
  *) echo "usage: $0 start|stop|restart|status" >&2; exit 2 ;;
esac

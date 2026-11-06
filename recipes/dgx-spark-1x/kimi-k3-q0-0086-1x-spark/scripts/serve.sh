#!/usr/bin/env bash
# serve.sh — llama-server for the Kimi-K3 Q0.0086 joke build.
# Everything is an env var on purpose: nothing here knows a hostname, an IP or a home directory.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
MODEL="${MODEL:-k3-q01-L12.gguf}"
NGL="${NGL:-99}"
CTX="${CTX:-4096}"
THREADS="${THREADS:-8}"
PORT="${PORT:-8099}"
HOST="${HOST:-127.0.0.1}"          # loopback by default; this thing must not face a network
PIDFILE="${PIDFILE:-$HOME/.k3-q01-l12.pid}"
LOG="${LOG:-$HOME/k3-q01-l12.log}"

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL (run scripts/build.sh)" >&2; exit 1; }
[[ -x "$LLAMA_DIR/build/bin/llama-server" ]] || { echo "llama-server not built in $LLAMA_DIR (run scripts/build.sh)" >&2; exit 1; }

nohup "$LLAMA_DIR/build/bin/llama-server" \
    -m "$MODEL" -ngl "$NGL" -c "$CTX" -t "$THREADS" \
    --host "$HOST" --port "$PORT" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"

for _ in $(seq 1 60); do
    if curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
        echo "serving $MODEL on http://$HOST:$PORT (pid $(cat "$PIDFILE"))"
        exit 0
    fi
    sleep 1
done
echo "server did not answer in 60 s; see $LOG" >&2
exit 1

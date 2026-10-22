#!/usr/bin/env bash
# stop.sh — stop the joke build. Nothing was displaced, so there is nothing to restore.
set -euo pipefail

PIDFILE="${PIDFILE:-$HOME/.k3-q01-l12.pid}"

if [[ -f "$PIDFILE" ]]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        echo "stopped pid $pid"
    else
        echo "pid $pid already gone"
    fi
    rm -f "$PIDFILE"
else
    echo "no pid file at $PIDFILE — nothing to stop"
fi

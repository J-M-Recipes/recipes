#!/usr/bin/env bash
# go-r1.sh — launch Card R1 detached (wrapper so no live cmdline contains the runner name except the runner itself)
cd "$HOME/dsv41/garner/r1" || exit 1
GO=1 setsid nohup bash r1_runner.sh </dev/null >/dev/null 2>&1 &
echo "launched pid $!"

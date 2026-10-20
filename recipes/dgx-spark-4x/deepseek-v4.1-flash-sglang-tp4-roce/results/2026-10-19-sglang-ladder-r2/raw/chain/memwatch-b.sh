#!/usr/bin/env bash
# Lane B MemAvailable watchdog. usage: memwatch-b.sh <sparkN> <logdir> <container>
NODE="$1"; D="$2"; NAME="$3"; mkdir -p "$D"; LOG="$D/memwatch-$NODE.log"; echo "$$" > "$D/memwatch-$NODE.pid"
low=0; i=0
while :; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo); gib=$(awk -v a="$a" 'BEGIN{printf "%.2f", a/1048576}')
  if [ $((i % 5)) -eq 0 ]; then g=$(nvidia-smi --query-gpu=clocks.gr,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); echo "$(date -u +%FT%TZ) avail_gib=$gib gpu(clk,util,W)=$g" >> "$LOG"; fi
  if [ "$a" -lt $((5*1048576)) ]; then echo "$(date -u +%FT%TZ) ALERT avail_gib=$gib < 5" >> "$LOG"; fi
  if [ "$a" -lt $((3*1048576)) ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 2 ]; then echo "$(date -u +%FT%TZ) KILL avail_gib=$gib < 3 twice: docker stop $NAME" >> "$LOG"; docker stop -t 5 "$NAME" >> "$LOG" 2>&1; exit 0; fi
  if [ $((i % 15)) -eq 0 ] && [ $i -gt 30 ]; then st=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo gone); if [ "$st" != "running" ]; then echo "$(date -u +%FT%TZ) container $st; watchdog exit" >> "$LOG"; exit 0; fi; fi
  i=$((i+1)); sleep 2
done

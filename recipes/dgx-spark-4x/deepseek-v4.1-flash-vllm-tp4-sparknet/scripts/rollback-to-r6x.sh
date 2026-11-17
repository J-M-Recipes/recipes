#!/usr/bin/env bash
# rollback-to-r6x.sh — put the kept r6x lane back on Sparks 5-8.
#
# The r7 promotion kept the r6x containers (`dsv41-karmic-kraken`, the config verified 2026-10-23) on all four
# nodes rather than deleting them. This stops the r7 lane and starts those containers again, workers before head.
#
#   SPARK_HOSTS="spark5 spark6 spark7 spark8" bash scripts/rollback-to-r6x.sh
#
# What it does, in order:
#   1. `docker stop -t 60 dsv41-r7-lane` on ranks 0..3 (the r7 container is kept, not removed)
#   2. `docker start dsv41-karmic-kraken` rank 3, 2, 1, then 0 — the head last, exactly the order the lane was
#      brought up in; starting the head first makes three ranks reconnect to a master that is still initialising
#   3. wait for /v1/models on the head
#   4. print the r6x KV pool line from the container log: it must read 8,580,566 tokens (8.18x at 1M) where the
#      r7 lane reads 12,034,343 (11.48x). That single line is the fastest way to tell which engine is serving.
#
# Env: SPARK_HOSTS (4 hosts, rank order), R7_NAME (dsv41-r7-lane), R6X_NAME (dsv41-karmic-kraken),
#      PORT 8002, BOOT_TIMEOUT 900.
set -u
read -r -a H <<<"${SPARK_HOSTS:?set SPARK_HOSTS to 4 ssh hosts, rank order (rank 0 = head)}"
[ "${#H[@]}" -eq 4 ] || { echo "SPARK_HOSTS needs 4 hosts" >&2; exit 2; }
HEAD=${H[0]}
PORT=${PORT:-8002}
R7_NAME=${R7_NAME:-dsv41-r7-lane}
R6X_NAME=${R6X_NAME:-dsv41-karmic-kraken}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-900}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
S() { $SSH "$1" "$2"; }

echo "== stop the r7 lane"
for n in "${H[@]}"; do S "$n" "docker stop -t 60 $R7_NAME 2>&1 | tail -1"; done

echo "== start the kept r6x containers (rank 3 -> 0)"
for r in 3 2 1 0; do
  n=${H[$r]}
  S "$n" "docker start $R6X_NAME 2>&1 | tail -1" || { echo "start FAILED on $n (rank $r)" >&2; exit 3; }
  [ $r -eq 0 ] || sleep 3
done

echo "== wait for :$PORT"
T0=$(date +%s)
while :; do
  if curl -s -m 5 "http://$HEAD:$PORT/v1/models" | grep -q deepseek; then
    echo "READY in $(( $(date +%s) - T0 ))s"; break
  fi
  [ $(( $(date +%s) - T0 )) -lt "$BOOT_TIMEOUT" ] || { echo "not ready after ${BOOT_TIMEOUT}s" >&2; exit 4; }
  sleep 10
done

S "$HEAD" "docker logs --tail 400 $R6X_NAME 2>&1 | grep -E 'GPU KV cache size' | tail -1"
echo "expected on r6x: GPU KV cache size: 8,580,566 tokens ... 8.18x (r7 reads 12,034,343 / 11.48x)"

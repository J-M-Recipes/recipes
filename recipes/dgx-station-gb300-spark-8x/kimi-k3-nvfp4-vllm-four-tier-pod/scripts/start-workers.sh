#!/bin/bash
# (Re)start the nine expert workers: Sparks 1-8 (plan targets spark0..spark7) and the Station's RTX PRO 6000 (rtx0).
# Run from any host that can ssh to all nine. Each worker accepts ONE engine connection per start, so run this
# before every engine boot.
#
#   SPARK_HOSTS="user@s1 user@s2 ... user@s8"  STATION=user@station  bash start-workers.sh
#
# Env: SPARK_HOSTS (8 ssh targets, rank order = spark0..7)  STATION (ssh target)
#      REMOTE_K3_DIR [~/k3pod]  — on every host: a3/{worker.py,k3proto.py,libk3hop.so,plan-*.json,run-worker-*.sh},
#                                 situ/venv (setup-worker-venv.sh); Spark banks in ~/k3/banks, RTX bank on the Station
set -uo pipefail
: "${SPARK_HOSTS:?set SPARK_HOSTS (8 hosts, rank order)}" "${STATION:?set STATION}"
read -r -a H <<<"$SPARK_HOSTS"
[ ${#H[@]} = 8 ] || { echo "need 8 SPARK_HOSTS, got ${#H[@]}"; exit 2; }
D=${REMOTE_K3_DIR:-'~/k3pod'}
for i in 0 1 2 3 4 5 6 7; do
  # fabric address of Spark i+1 is 10.0.0.(i+1) in this topology (see hardware profile)
  ssh -o BatchMode=yes -o ConnectTimeout=6 "${H[$i]}" "bash $D/a3/run-worker-spark.sh spark$i 10.0.0.$((i + 1)) 19600 >/dev/null" &
done
: "${RTX_UUID:?set RTX_UUID (GPU UUID of the RTX PRO 6000 in the Station)}"
ssh -o BatchMode=yes "$STATION" "RTX_UUID=$RTX_UUID bash $D/a3/run-worker-rtx.sh >/dev/null" &
wait
for t in $(seq 1 60); do
  ready=0
  for i in 0 1 2 3 4 5 6 7; do
    ssh -o BatchMode=yes "${H[$i]}" "docker logs k3-worker 2>&1 | grep -q K3HOP_LISTEN" && ready=$((ready + 1))
  done
  ssh -o BatchMode=yes "$STATION" "docker logs k3-worker-rtx 2>&1 | grep -q K3HOP_LISTEN" && ready=$((ready + 1))
  echo "$(date +%T) workers listening: $ready/9"
  [ $ready = 9 ] && exit 0
  sleep 10
done
echo "workers not ready; check 'docker logs k3-worker' (a failed WGRAPH boot check exits the worker)"; exit 1

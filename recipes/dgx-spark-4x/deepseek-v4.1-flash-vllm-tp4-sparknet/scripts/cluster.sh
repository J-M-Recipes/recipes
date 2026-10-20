#!/usr/bin/env bash
# cluster.sh — Lane C driver for 4x DGX Spark (vLLM fork, TP4). Runs on a workstation with ssh to all four nodes.
# Published form of the ladder runner (r2-arm.sh + launch-arm.sh): same hygiene, same rank order, same readiness probe.
#   SPARK_HOSTS="spark5 spark6 spark7 spark8" bash scripts/cluster.sh push     copy launch-vllm-tp4.sh to ~/dsv41-vllm-tp4/
#   SPARK_HOSTS=... bash scripts/cluster.sh up       drop caches + compact, MemAvailable gate, ranks 3,2,1 then 0, wait for /v1/models
#   SPARK_HOSTS=... bash scripts/cluster.sh down     docker stop the container on all four (head first; container kept)
#   SPARK_HOSTS=... bash scripts/cluster.sh status   container state + MemAvailable per node
# Env: PORT (8002), NAME (dsv41-karmic-kraken), MIN_AVAIL_GIB (100; the ladder's launcher refused below this),
#      MODEL_DIR / DS41F_DIR / MASTER_ADDR / IMAGE are passed through to the launcher.
set -u
D=$(cd "$(dirname "$0")" && pwd)
read -r -a _H <<<"${SPARK_HOSTS:?set SPARK_HOSTS to 4 ssh hosts, rank order (rank 0 = head)}"
[ "${#_H[@]}" -eq 4 ] || { echo "SPARK_HOSTS needs 4 hosts" >&2; exit 2; }
HEAD=${_H[0]}; host(){ echo "${_H[$1]}"; }
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
NAME=${NAME:-dsv41-karmic-kraken}; PORT=${PORT:-8002}; MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-100}
on(){ $SSH "$1" "$2"; }
PASS="SPARK_HOSTS='${SPARK_HOSTS}' NAME=$NAME${MODEL_DIR:+ MODEL_DIR=$MODEL_DIR}${DS41F_DIR:+ DS41F_DIR=$DS41F_DIR}${MASTER_ADDR:+ MASTER_ADDR=$MASTER_ADDR}${IMAGE:+ IMAGE=$IMAGE}"

case "${1:-}" in
push)
  for r in 0 1 2 3; do n=$(host $r); on "$n" 'mkdir -p ~/dsv41-vllm-tp4' && scp -q "$D/launch-vllm-tp4.sh" "$n":dsv41-vllm-tp4/ && echo "pushed $n"; done ;;
down)
  for r in 0 1 2 3; do n=$(host $r); echo "$n $(on "$n" "docker stop -t 30 $NAME 2>&1 | tail -1")"; done ;;
status)
  for r in 0 1 2 3; do n=$(host $r); echo "$n $(on "$n" "docker ps -a --filter name=^$NAME\$ --format '{{.Status}}'; awk '/MemAvailable/{printf \"avail %.1f GiB\", \$2/1048576}' /proc/meminfo" | tr '\n' ' ')"; done ;;
up)
  for r in 0 1 2 3; do
    n=$(host $r)
    if on "$n" "docker ps -a --format '{{.Names}}' | grep -qx $NAME"; then echo "$n: container $NAME exists; run down + docker rm (or rename) first" >&2; exit 3; fi
  done
  for r in 0 1 2 3; do
    n=$(host $r)
    a=$(on "$n" "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null; awk '/MemAvailable/{print \$2}' /proc/meminfo")
    echo "$n post-drop MemAvailable $((a/1048576)) GiB"
    [ "$a" -ge $((MIN_AVAIL_GIB*1048576)) ] || { echo "$n MemAvailable below $MIN_AVAIL_GIB GiB; not launching" >&2; exit 4; }
  done
  for r in 3 2 1 0; do
    n=$(host $r)
    on "$n" "$PASS bash ~/dsv41-vllm-tp4/launch-vllm-tp4.sh $r" >/dev/null || { echo "rank $r launch FAILED on $n" >&2; exit 5; }
    echo "rank $r launched on $n"; [ $r -eq 0 ] || sleep 2
  done
  t0=$(date +%s)
  until on "$HEAD" "curl -sf -m 5 http://127.0.0.1:$PORT/v1/models >/dev/null"; do
    [ $(( $(date +%s) - t0 )) -lt 1800 ] || { echo "not ready after 1800 s" >&2; exit 6; }
    for r in 0 1 2 3; do st=$(on "$(host $r)" "docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null"); [ "$st" = running ] || { echo "rank $r container state=$st" >&2; exit 7; }; done
    sleep 10
  done
  echo "HTTP READY after $(( $(date +%s) - t0 )) s" ;;
*) echo "usage: SPARK_HOSTS=\"h0 h1 h2 h3\" $0 push|up|down|status" >&2; exit 2 ;;
esac

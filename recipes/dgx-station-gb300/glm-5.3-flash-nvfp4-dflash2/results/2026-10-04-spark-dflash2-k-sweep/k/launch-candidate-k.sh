#!/usr/bin/env bash
# E1 launcher. One axis is which rank script is invoked (eager vs graphs).
# Does not arm a deadman. Does not touch DS4-F or k3-g2a containers.
set -Eeuo pipefail

mode="${1:?usage: launch-candidate-e4.sh k4|k3}"
case "$mode" in
  k4|k3|k7) ;;
  *) echo "mode must be k4 or k3" >&2; exit 2 ;;
esac

root=~/glm53-k-20261004
rank_script="$root/launch-dflash-rank-${mode}.sh"
[[ -x "$rank_script" ]]

echo "[$(date -Is)] K launch begin mode=$mode"
echo "UNAME $(uname -r)"
echo "IMAGE $(docker image inspect funland/glm53-dflash2:ccc5a78e --format '{{.Id}}')"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader || true

if docker ps --format '{{.Names}}' | grep -E '^(glm53-k-rank|glm53-dflash2-rank)'; then
  echo "refusing: a glm53 rank container is already running" >&2
  docker ps --format '{{.Names}} {{.Status}}'
  exit 3
fi

local_sha=$(sha256sum "$rank_script" | awk '{print $1}')
remote_sha=$(ssh -o BatchMode=yes milo@<rdma-peer-2> "sha256sum $rank_script" | awk '{print $1}')
echo "rank_script_sha local=$local_sha remote=$remote_sha"
[[ "$local_sha" == "$remote_sha" ]]

head_avail=$(python3 -c 'print(next(int(x.split()[1]) for x in open("/proc/meminfo") if x.startswith("MemAvailable:")))')
worker_avail=$(ssh -o BatchMode=yes milo@<rdma-peer-2> "python3 -c 'print(next(int(x.split()[1]) for x in open(\"/proc/meminfo\") if x.startswith(\"MemAvailable:\")))'")
echo "<rdma-peer-1> MemAvailable_kB=$head_avail"
echo "<rdma-peer-2> MemAvailable_kB=$worker_avail"
(( head_avail >= 90000000 ))
(( worker_avail >= 90000000 ))

echo "[$(date -Is)] launching rank1 via $rank_script"
ssh -o BatchMode=yes milo@<rdma-peer-2> "$rank_script 1"
sleep 15
echo "[$(date -Is)] launching rank0 via $rank_script"
"$rank_script" 0

ready=0
for _ in $(seq 1 160); do
  head_running=$(docker inspect glm53-k-rank0 --format '{{.State.Running}}' 2>/dev/null || true)
  worker_running=$(ssh -o BatchMode=yes milo@<rdma-peer-2> "docker inspect glm53-k-rank1 --format '{{.State.Running}}' 2>/dev/null || true")
  if [[ "$head_running" != true || "$worker_running" != true ]]; then
    echo "[$(date -Is)] rank exited head=$head_running worker=$worker_running"
    echo "===== rank0 logs tail 300 ====="
    docker logs --tail 300 glm53-k-rank0 > "$root/rank0-died-${mode}.log" 2>&1 || true
    cat "$root/rank0-died-${mode}.log" || true
    echo "===== rank1 logs tail 300 ====="
    ssh -o BatchMode=yes milo@<rdma-peer-2> "docker logs --tail 300 glm53-k-rank1" > "$root/rank1-died-${mode}.log" 2>&1 || true
    cat "$root/rank1-died-${mode}.log" || true
    exit 1
  fi
  if curl -fsS --max-time 3 http://127.0.0.1:8888/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 15
done

if (( ready != 1 )); then
  echo "[$(date -Is)] health timeout after 40 min"
  docker logs --tail 300 glm53-k-rank0 > "$root/rank0-timeout-${mode}.log" 2>&1 || true
  cat "$root/rank0-timeout-${mode}.log" || true
  exit 1
fi

ts=$(date -Is)
echo "K_READY $mode $ts"
head_ready=$(python3 -c 'print(next(int(x.split()[1]) for x in open("/proc/meminfo") if x.startswith("MemAvailable:")))')
worker_ready=$(ssh -o BatchMode=yes milo@<rdma-peer-2> "python3 -c 'print(next(int(x.split()[1]) for x in open(\"/proc/meminfo\") if x.startswith(\"MemAvailable:\")))'")
echo "READY_MEM head_kB=$head_ready worker_kB=$worker_ready"

{
  echo "===== boot log grep rank0 mode=$mode at $ts ====="
  docker logs glm53-k-rank0 2>&1 | grep -E -i -n 'cudagraph|enforce_eager|enforce-eager|compilation|speculative|draft' || true
} > "$root/boot-grep-${mode}.txt" 2>&1 || true
{
  echo "===== cudagraph/compilation/capturing/graph first 40 mode=$mode at $ts ====="
  docker logs glm53-k-rank0 2>&1 | grep -E -i -n 'cudagraph|compilation|capturing|graph' | head -n 40 || true
} > "$root/boot-cudagraph-head40-${mode}.txt" 2>&1 || true

echo "===== boot log grep (head 200) ====="
head -n 200 "$root/boot-grep-${mode}.txt" || true
echo "===== cudagraph mode first 40 ====="
cat "$root/boot-cudagraph-head40-${mode}.txt" || true
echo "[$(date -Is)] K launcher done mode=$mode"

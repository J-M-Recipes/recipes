#!/usr/bin/env bash
# Diag: AR baseline boot, then DFlash2 pin boot with per-position counters + GB300 probe prompts.
# Same Tony eager recipe. Does not touch DS4-F, k3-g2a, the Aug 27 canary, the pin, or .9.
set -Eeuo pipefail
root=~/glm53-diag-20261004
cd "$root"
log() { echo "[$(date -Is)] $*"; }
trap 'rc=$?; log "RUNNER_ERR line=$LINENO rc=$rc"' ERR
trap 'rc=$?; if (( rc != 0 )); then docker rm -f glm53-diag-rank0 >/dev/null 2>&1 || true; ssh -o BatchMode=yes -o ConnectTimeout=8 milo@<rdma-peer-2> "docker rm -f glm53-diag-rank1" >/dev/null 2>&1 || true; log "CLEANUP_RAN rc=$rc"; fi' EXIT

mem_head() { python3 -c 'print(next(int(x.split()[1]) for x in open("/proc/meminfo") if x.startswith("MemAvailable:")))'; }
mem_worker() { ssh -o BatchMode=yes -o ConnectTimeout=8 milo@<rdma-peer-2> "python3 -c 'print(next(int(x.split()[1]) for x in open(\"/proc/meminfo\") if x.startswith(\"MemAvailable:\")))'"; }

teardown() {
  docker rm -f glm53-diag-rank0 || true
  ssh -o BatchMode=yes milo@<rdma-peer-2> 'docker rm -f glm53-diag-rank1' || true
  sleep 5
  log "MEM_AFTER_$1_HEAD $(mem_head)"
  log "MEM_AFTER_$1_WORKER $(mem_worker)"
}

smoke() {
python3 - << 'PY'
import json, urllib.request
base = "http://127.0.0.1:8888"
print("SMOKE_MODELS " + urllib.request.urlopen(base + "/v1/models", timeout=30).read().decode())
payload = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Reply with one sentence: what is a DGX Spark?"}],
           "temperature": 0, "top_p": 1, "max_tokens": 64, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as resp:
    body = json.loads(resp.read().decode())
print("SMOKE_TEXT " + json.dumps(((body.get("choices") or [{}])[0].get("message") or {}).get("content")))
print("SMOKE_USAGE " + json.dumps(body.get("usage")))
PY
}

log "RUNNER_START host=$(hostname) uname=$(uname -r)"
[[ -f "$root/STOP" ]] && { log "STOP present"; exit 0; }
head_run=$(docker ps --format '{{.Names}}' || true)
worker_run=$(ssh -o BatchMode=yes -o ConnectTimeout=8 milo@<rdma-peer-2> 'docker ps --format "{{.Names}}"' || true)
if [[ -n "$head_run" || -n "$worker_run" ]]; then log "REFUSE running containers head=[$head_run] worker=[$worker_run]"; exit 3; fi
log "CLOCKS_HEAD $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader)"
log "MEM_HEAD $(mem_head)"; log "MEM_WORKER $(mem_worker)"
log "CANARY $(sha256sum ~/glm53-dflash2-canary-20260827/launch-dflash-rank.sh)"
log "PIN $(sha256sum ~/models/GLM-5.3-Flash-DFlash2-7d74cdd/model.safetensors ~/models/GLM-5.3-Flash-DFlash2-7d74cdd/config.json | tr '\n' '|')"

ssh -o BatchMode=yes milo@<rdma-peer-2> "mkdir -p $root"
rsync -aH --exclude 'raw/' --exclude '*.log' --exclude '*.pid' "$root/" "milo@<rdma-peer-2>:$root/"
for s in launch-dflash-rank-ar.sh launch-dflash-rank-dflash.sh; do
  l=$(sha256sum "$root/$s" | awk '{print $1}'); r=$(ssh -o BatchMode=yes milo@<rdma-peer-2> "sha256sum $root/$s" | awk '{print $1}')
  log "RANK_SCRIPT_SHA $s local=$l remote=$r"; [[ "$l" == "$r" ]]
done

# ---- Boot 1: AR baseline ----
log "LAUNCH_AR_BEGIN"
bash "$root/launch-candidate-diag.sh" ar > "$root/launch-ar.log" 2>&1
log "LAUNCH_AR_DONE"
log "SMOKE_AR_BEGIN"; smoke; log "SMOKE_AR_DONE"
{ echo "===== spec metric series present on AR boot? ====="; curl -fsS http://127.0.0.1:8888/metrics | grep -c spec_decode || echo 0; } > "$root/ar-metrics-check.txt" 2>&1 || true
log "MATRIX_AR_BEGIN"
python3 "$root/diag_matrix.py" --mode ar --out "$root/raw/ar" > "$root/matrix-ar.log" 2>&1
log "MATRIX_AR_DONE"
log "TEARDOWN_AR_BEGIN"; teardown AR; log "TEARDOWN_AR_DONE"
[[ -f "$root/STOP" ]] && { log "STOP after AR"; exit 0; }

# ---- Boot 2: DFlash2 pin, per-pos counters + probe prompts ----
log "LAUNCH_DFLASH_BEGIN"
bash "$root/launch-candidate-diag.sh" dflash > "$root/launch-dflash.log" 2>&1
log "LAUNCH_DFLASH_DONE"
log "SMOKE_DFLASH_BEGIN"; smoke; log "SMOKE_DFLASH_DONE"
{ echo "===== per_pos series at boot ====="; curl -fsS http://127.0.0.1:8888/metrics | grep per_pos || echo NONE; } > "$root/dflash-perpos-at-boot.txt" 2>&1 || true
{ docker logs glm53-diag-rank0 2>&1 | grep -n -i -E 'draft|dflash|speculat|tensor_parallel|TP0|TP1' | grep -v -E 'non-default args|Initializing a V1' | cut -c1-300; } > "$root/boot-draft-placement.txt" 2>&1 || true
log "MATRIX_DFLASH_BEGIN"
python3 "$root/diag_matrix.py" --mode dflash --out "$root/raw/dflash" > "$root/matrix-dflash.log" 2>&1
log "MATRIX_DFLASH_DONE"
{ echo "===== per_pos series at end ====="; curl -fsS http://127.0.0.1:8888/metrics | grep -E "per_pos|spec_decode_num" || echo NONE; } > "$root/dflash-perpos-at-end.txt" 2>&1 || true
log "TEARDOWN_DFLASH_BEGIN"; teardown DFLASH; log "TEARDOWN_DFLASH_DONE"

log "CANARY_END $(sha256sum ~/glm53-dflash2-canary-20260827/launch-dflash-rank.sh)"
log "PIN_END $(sha256sum ~/models/GLM-5.3-Flash-DFlash2-7d74cdd/model.safetensors ~/models/GLM-5.3-Flash-DFlash2-7d74cdd/config.json | tr '\n' '|')"
log "DOCKER_PS_A_HEAD_END $(docker ps -a --format '{{.Names}} {{.Status}}' | tr '\n' '|')"
log "DOCKER_PS_A_WORKER_END $(ssh -o BatchMode=yes milo@<rdma-peer-2> 'docker ps -a --format "{{.Names}} {{.Status}}"' | tr '\n' '|')"
log "CLOCKS_END $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader)"
log "RUNNER_DONE"

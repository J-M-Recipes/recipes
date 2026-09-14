#!/usr/bin/env bash
# campaign_round3.sh — Round 3 on the DSV4.1 :30006 lane. Three experiments, one axis each, same-window controls.
#   E1 depth map   : decode tok/s vs prompt depth on the v12 REF (0 boots)
#   E2 async-sched : v12 flags + --async-scheduling (1 boot, autotune hash should HIT)
#   E3 k-schedule  : v12 flags + num_speculative_tokens_per_batch_size (1 boot, NEW hash, ~80 min)
# Usage: bash campaign_round3.sh [E1|E2|E3|ALL]   (default ALL, in order). STOP-CAMPAIGN file halts between steps.
# Ends with the lane DARK (REF stopped+kept) unless RESTORE_REF=1. Never touches :30003.
set -euo pipefail
cd /home/milo/dsv41
REF=dsv41-vllm-v12-1M-k5-off60-util97-BOUND-REF
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
LEDGER=/home/milo/dsv41/results/ledger.md
CAMPLOG=/home/milo/dsv41/results/campaign-round3-$(date +%F).log
STOPF=/home/milo/dsv41/STOP-CAMPAIGN
RESTORE_REF="${RESTORE_REF:-0}"
WHICH="${1:-ALL}"
mkdir -p /home/milo/dsv41/results
exec >> "$CAMPLOG" 2>&1

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }
die() { log "FATAL: $*"; finish || true; exit 1; }
trap 'log "ERR line=$LINENO status=$?"' ERR
check_stop() { if [[ -f "$STOPF" ]]; then log "STOP-CAMPAIGN"; finish; exit 0; fi; return 0; }
check_safety() {
  local other; other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true)
  [[ -n "$other" ]] && die "other container(s) running: $other"
  dmesg -T 2>/dev/null | tail -n 40 | grep -qiE 'Out of memory|oom-kill' && die "oom in dmesg"
  [[ -s depth_knee.py ]] || die "depth_knee.py missing"
  docker ps -a --format '{{.Names}}' | grep -qx "$REF" || die "REF container missing"
  return 0
}
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; log "dropped page cache; $(free -g | awk 'NR==2{print "avail="$7"GiB used="$3"GiB"}')"; }
wait_bound() {  # $1=container name  $2=max seconds
  local name=$1; local maxs=$2; local t0=$SECONDS
  log "waiting bind for $name (max ${maxs}s)"
  while true; do
    check_stop
    docker ps --format '{{.Names}}' | grep -qx "$name" || die "container $name not running"
    if curl -sf -m 3 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $name after $((SECONDS-t0))s"; return 0; fi
    local el=$((SECONDS-t0)); local gpu; gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 | tr -cd '0-9' || true)
    [[ $el -gt $maxs ]] && die "$name exceeded ${maxs}s without bind"
    (( el % 120 < 16 )) && log "still booting $name t=${el}s gpu=${gpu:-0}%"
    sleep 15
  done
}
fast_fail() {  # $1=tag — first 5 min: container exit or a real error line
  local tag=$1; local logf=/home/milo/dsv41/dsv41-vllm-$tag.log
  for i in $(seq 1 30); do
    sleep 10
    docker ps --format '{{.Names}}' | grep -qx "dsv41-vllm-$tag" || { docker logs --tail 60 "dsv41-vllm-$tag" 2>&1 | grep -aiE "error|traceback" | tail -n 12; die "boot rejected (container exited)"; }
    if grep -aE "ValueError:|NameError:|ImportError:|AttributeError:|Traceback \(most recent|RuntimeError:" "$logf" 2>/dev/null | grep -qv Warning; then
      grep -aE "ValueError:|NameError:|ImportError:|AttributeError:|Traceback|RuntimeError:" -A 3 "$logf" | tail -n 16
      sleep 5; curl -sf -m 2 http://127.0.0.1:30006/v1/models >/dev/null 2>&1 || die "boot rejected (log error)"
    fi
  done
}
ensure_ref() {
  if ! docker ps --format '{{.Names}}' | grep -qx "$REF"; then
    local running; running=$(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-' || true)
    [[ -n "$running" ]] && { log "stopping $running"; echo "$running" | xargs -r docker stop; }
    drop_caches; log "starting $REF"; docker start "$REF"
    nohup docker logs -f "$REF" > /home/milo/dsv41/dsv41-vllm-REF-round3.log 2>&1 &
    wait_bound "$REF" 2400
  fi
}
smoke() {
  python3 - <<'PY'
import json, urllib.request, sys
B="http://127.0.0.1:30006/v1/chat/completions"
def chat(msgs, mx=64, think=False):
    body={"model":"dsv41-flash-uva","messages":msgs,"max_tokens":mx,"temperature":0,"chat_template_kwargs":{"thinking":think}}
    if think: body["reasoning_effort"]="low"
    r=json.load(urllib.request.urlopen(urllib.request.Request(B, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=300))
    m=r["choices"][0]["message"]; return (m.get("content") or ""), m
fails=[]
c,_=chat([{"role":"user","content":"What is 17*19? Return only the integer."}]); "323" in c.replace(",","") or fails.append("arith")
c,_=chat([{"role":"user","content":"Count from 1 to 30 separated by spaces. Only the numbers."}], mx=80)
nums=[int(x) for x in c.split() if x.isdigit()]; nums[:30]==list(range(1,31)) or fails.append("count")
c,m=chat([{"role":"user","content":"What is 15% of 240? Reply with the number."}], mx=400, think=True)
(c or m.get("reasoning_content") or m.get("reasoning")) or fails.append("think")
if fails: print("SMOKE_FAIL", fails); sys.exit(2)
print("SMOKE_PASS")
PY
}
boot_facts() {
  local tag=$1; local logf=/home/milo/dsv41/dsv41-vllm-$tag.log
  { echo "=== $tag boot facts $(date) ==="
    grep -aoE "autotune cache file: [^ ]*" "$logf" | head -1
    grep -aoE "Loaded [0-9]+ configs|Saved [0-9]+ configs|Autotuning process (starts|ends)" "$logf" | head -4
    grep -aoE "Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB|Available KV cache memory: [0-9.-]+ GiB|Total CPU offloaded parameters:[^\"]*|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$logf"
    grep -aoE "num_speculative_tokens_per_batch_size[^,]*|async_scheduling[^,]*" "$logf" | sort -u | head -4
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
  } | tee "/home/milo/dsv41/results/facts-$tag.txt"
}
c_of() { python3 -c "import json;r=json.load(open('/home/milo/dsv41/knee-$1.json'))['rows'];print(' '.join(f'C{c}={a:.1f}' for c,a,_ in r))" 2>/dev/null || echo NA; }
fx_of() {
  python3 - "$1" <<'PY' 2>/dev/null || echo NA
import json,sys
d=json.load(open(f"/home/milo/dsv41/agentfix-{sys.argv[1]}.json")); r=d["res"]
print(" ".join(f"{c}={r[c]['tok_s']}({r[c]['accept']:.2f})" for c in ("prose","shell_ops","code","tool_json","structured") if c in r)+f" wacc={d['weighted_accept']:.3f}")
PY
}
measure() { local tag=$1; log "SMOKE $tag"; smoke || die "smoke failed $tag"; log "KNEE $tag"; bash knee.sh "$tag"; log "FIXTURE $tag"; bash agent_fixture.sh "$tag" || true; log "RESULT $tag | $(c_of $tag) | $(fx_of $tag)"; }
ledger() { echo "$(date +%F) ROUND3 $*" >> "$LEDGER"; log "LEDGER $*"; }
finish() {
  log "FINISH: stop candidates, keep containers"
  docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-R3-' | xargs -r docker stop || true
  if [[ "$RESTORE_REF" == "1" ]]; then drop_caches; docker start "$REF"; wait_bound "$REF" 2400 || log "RESTORE bind timeout"
  else docker stop "$REF" 2>/dev/null || true; log "lane left dark (RESTORE_REF=0)"; fi
}

log "ROUND3 start: which=$WHICH"; check_safety

# ---------------- E1: depth map on REF (0 boots) ----------------
if [[ "$WHICH" == "ALL" || "$WHICH" == "E1" ]]; then
  ensure_ref; log "SMOKE REF"; smoke || die "REF smoke failed"
  log "E1 DEPTH on REF"; TAG=E1-v12-depth DEPTHS="${DEPTHS:-8000 65536 131072 262144 524288}" python3 depth_knee.py
  ledger "| E1-v12-depth | $(python3 -c "import json;print(' '.join(f\"{r['prompt_tokens']}:{r['decode_tok_s']}\" for r in json.load(open('/home/milo/dsv41/depth-E1-v12-depth.json'))['rows']))") |"
  check_stop
fi

# ---------------- E2: --async-scheduling (1 boot, hash HIT expected) ----------------
if [[ "$WHICH" == "ALL" || "$WHICH" == "E2" ]]; then
  ensure_ref; ctrl=R3-CTRL-for-E2; log "KNEE control $ctrl"; measure "$ctrl"
  docker stop "$REF"; drop_caches
  tag=R3-E2-async; launched=$(date '+%H:%M')
  TAG="$tag" OFFGB=60 UTIL=0.97 SEQS=16 CTX=1048576 SPEC="dspark:5" \
    EXTRA="--long-prefill-token-threshold 6144 --async-scheduling" MODEL="$MODEL" bash launch-dsv41-vllm.sh
  fast_fail "$tag"; wait_bound "dsv41-vllm-$tag" 1800; boot_facts "$tag"
  grep -aq "Loaded [0-9]* configs" "dsv41-vllm-$tag.log" || log "WARN E2: autotune did NOT hit (unexpected) — boot time will show it"
  measure "$tag"
  ledger "| E2 async | $launched → $(date '+%H:%M') | ctrl $(c_of $ctrl) | cand $(c_of $tag) | $(fx_of $tag) |"
  docker stop "dsv41-vllm-$tag"; docker rename "dsv41-vllm-$tag" "dsv41-vllm-$tag-EXP" 2>/dev/null || true
  check_stop
fi

# ---------------- E3: batch-size k-schedule (1 boot, NEW hash ~80 min) ----------------
if [[ "$WHICH" == "ALL" || "$WHICH" == "E3" ]]; then
  ensure_ref; ctrl=R3-CTRL-for-E3; log "KNEE control $ctrl"; measure "$ctrl"
  docker stop "$REF"; drop_caches
  tag=R3-E3-ksched; launched=$(date '+%H:%M')
  TAG="$tag" OFFGB=60 UTIL=0.97 SEQS=16 CTX=1048576 SPEC="" \
    EXTRA="--long-prefill-token-threshold 6144 --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":[[1,2,5],[3,16,1]]}" \
    MODEL="$MODEL" bash -f launch-dsv41-vllm.sh   # -f: the JSON's [..] must not glob (EXTRA is expanded unquoted)
  docker inspect "dsv41-vllm-$tag" --format '{{join .Args " "}}' | grep -q 'num_speculative_tokens_per_batch_size' || die "E3: k-schedule did not reach the container args"
  fast_fail "$tag"; wait_bound "dsv41-vllm-$tag" 10800; boot_facts "$tag"
  measure "$tag"
  ledger "| E3 ksched | $launched → $(date '+%H:%M') | ctrl $(c_of $ctrl) | cand $(c_of $tag) | $(fx_of $tag) |"
  docker stop "dsv41-vllm-$tag"; docker rename "dsv41-vllm-$tag" "dsv41-vllm-$tag-EXP" 2>/dev/null || true
fi

# ---------------- CONFIRM: E1b (fixed depth instrument) on REF + E3 pair 2 ----------------
if [[ "$WHICH" == "CONFIRM" ]]; then
  P="${PAIR:-p2}"
  ensure_ref; log "SMOKE REF"; smoke || die "REF smoke failed"
  log "E1b DEPTH (streamed) on REF"; TAG="E1b-v12-depth-$P" DEPTHS="${DEPTHS:-8000 65536 131072 262144 524288}" python3 depth_knee.py
  ledger "| E1b-v12-depth-$P | $(python3 -c "import json;print(' '.join(f\"{r['prompt_tokens']}:{r['decode_tok_s']}\" for r in json.load(open('/home/milo/dsv41/depth-E1b-v12-depth-$P.json'))['rows']))") |"
  check_stop
  ctrl=R3-CTRL-for-E3-$P; log "KNEE control $ctrl"; measure "$ctrl"
  docker stop "$REF"; drop_caches
  tag=R3-E3-ksched-$P; launched=$(date '+%H:%M')
  TAG="$tag" OFFGB=60 UTIL=0.97 SEQS=16 CTX=1048576 SPEC="" \
    EXTRA="--long-prefill-token-threshold 6144 --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":[[1,2,5],[3,16,1]]}" \
    MODEL="$MODEL" bash -f launch-dsv41-vllm.sh
  docker inspect "dsv41-vllm-$tag" --format '{{join .Args " "}}' | grep -q 'num_speculative_tokens_per_batch_size' || die "E3-$P: k-schedule did not reach the container args"
  fast_fail "$tag"; wait_bound "dsv41-vllm-$tag" 10800; boot_facts "$tag"
  measure "$tag"
  ledger "| E3 ksched $P | $launched → $(date '+%H:%M') | ctrl $(c_of $ctrl) | cand $(c_of $tag) | $(fx_of $tag) |"
  docker stop "dsv41-vllm-$tag"; docker rename "dsv41-vllm-$tag" "dsv41-vllm-$tag-EXP" 2>/dev/null || true
fi

finish; log "ROUND3 done"

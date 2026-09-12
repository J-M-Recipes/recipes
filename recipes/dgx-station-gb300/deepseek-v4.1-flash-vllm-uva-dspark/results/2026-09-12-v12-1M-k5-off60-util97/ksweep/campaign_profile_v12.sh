#!/usr/bin/env bash
# CONFIRM v12 — same-window pair. (1) knee the live v11 REF as control. (2) boot v12 (OFFGB=60 UTIL=0.97; hash 9ac7b387 cached),
# smoke, knee, fixture, prefill probe, starve. (3) if v12 C1 > control by >1.5%: v12 STAYS UP and becomes *-BOUND-REF; v11 REF
# renamed *-v11-RETIRED-REF (stopped, never removed). Else stop v12, restore v11 REF. Never touches :30003.
set -euo pipefail
cd <workdir>
AT=<workdir>/vllm-cache/flashinfer_autotune_cache/0.6.18/103a
LIVE=08c89d94da9966b94e6f696d6e5ea5023794b37f65508524bc5db239a1704796
REF=dsv41-vllm-v12-1M-k5-off60-util97-BOUND-REF
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
LEDGER=<workdir>/results/ledger.md
CAMPLOG=<workdir>/results/campaign-profile-v12-2026-09-12.log
STOPF=<workdir>/STOP-CAMPAIGN
SEED=<workdir>/seed-autotune.json
mkdir -p <workdir>/results
exec >> "$CAMPLOG" 2>&1

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }
die() { log "FATAL: $*"; restore_ref || true; exit 1; }
trap 'log "ERR line=$LINENO status=$?"' ERR
check_stop() { if [[ -f "$STOPF" ]]; then log "STOP-CAMPAIGN"; restore_ref; exit 0; fi; return 0; }

check_safety() {
  if docker ps --format '{{.Names}}' | grep -qE '^dsfv-'; then die "dsfv-* running"; fi
  if dmesg -T 2>/dev/null | tail -n 40 | grep -qiE 'Out of memory|oom-kill'; then die "oom in dmesg"; fi
  return 0
}
drop_caches() {
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  log "dropped page cache; $(free -g | awk 'NR==2{print "avail="$7"GiB used="$3"GiB"}')"
}
seed_watch() {
  local tag=$1 known; known=$(ls "$AT" 2>/dev/null | tr '\n' ' ')
  log "seed-watch $tag"
  for i in $(seq 1 1080); do
    if curl -sf -m 2 http://127.0.0.1:30006/v1/models >/dev/null 2>&1; then log "seed-watch: API bound"; return 0; fi
    for d in "$AT"/*/; do
      [[ -d "$d" ]] || continue
      local base; base=$(basename "$d")
      [[ " $known " == *" $base "* ]] && continue
      if [[ ! -f "$d/autotune_configs.json" ]]; then
        sudo cp "$SEED" "$d/autotune_configs.json"; sudo chmod 600 "$d/autotune_configs.json"
        log "seed-watch: seeded $base"; return 0
      else
        log "seed-watch: new dir $base already has configs"; return 0
      fi
    done
    sleep 5
  done
  log "seed-watch timeout"
}
wait_bound() {
  local tag=$1 logf=<workdir>/dsv41-vllm-$tag.log t0=$SECONDS
  log "waiting bind for $tag"
  while true; do
    check_stop
    if ! docker ps --format '{{.Names}}' | grep -qx "dsv41-vllm-$tag"; then die "container dsv41-vllm-$tag not running"; fi
    if curl -sf -m 3 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then
      log "BOUND $tag after $((SECONDS-t0))s"; return 0
    fi
    elapsed=$((SECONDS-t0))
    gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 | tr -cd '0-9' || true); gpu=${gpu:-0}
    if [[ $elapsed -gt 10800 ]]; then die "$tag boot exceeded 180 min without bind"; fi
    if [[ $elapsed -gt 1800 && "${gpu:-0}" -lt 40 ]]; then
      if ! grep -qE 'Loaded .* configs|Autotuning process starts' "$logf" 2>/dev/null; then
        die "$tag 30 min, GPU ${gpu}%, no autotune/bind"
      fi
    fi
    if (( elapsed % 60 < 12 )); then log "still booting $tag t=${elapsed}s gpu=${gpu}%"; fi
    sleep 10
  done
}
smoke() {
  python3 - <<'PY'
import json, urllib.request, sys
B="http://127.0.0.1:30006/v1/chat/completions"
def chat(msgs, mx=64, think=False):
    body={"model":"dsv41-flash-uva","messages":msgs,"max_tokens":mx,"temperature":0,
          "chat_template_kwargs":{"thinking":think}}
    if think: body["reasoning_effort"]="low"
    r=json.load(urllib.request.urlopen(urllib.request.Request(
        B, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=300))
    m=r["choices"][0]["message"]
    return (m.get("content") or ""), m, r.get("usage",{}), r["choices"][0].get("finish_reason")
fails=[]
c,m,u,f=chat([{"role":"user","content":"What is 17*19? Return only the integer."}])
print(f"ARITH content={c!r} finish={f}")
if "323" not in (c or "").replace(",",""): fails.append("arith")
c,m,u,f=chat([{"role":"user","content":"Count from 1 to 30 separated by spaces. Only the numbers."}], mx=80)
nums=[int(x) for x in (c or "").split() if x.isdigit()]
print(f"COUNT nums={nums[:30]}")
if nums[:30] != list(range(1,31)): fails.append("count")
c,m,u,f=chat([{"role":"user","content":"What is 15% of 240? Reply with the number."}], mx=400, think=True)
rc=m.get("reasoning_content") or m.get("reasoning") or ""
print(f"THINK content={c!r} reasoning_len={len(rc)}")
if not (c or rc): fails.append("think")
if fails:
    print("SMOKE_FAIL", fails); sys.exit(2)
print("SMOKE_PASS")
PY
}
record_facts() {
  local tag=$1 logf=<workdir>/dsv41-vllm-$tag.log
  {
    echo "=== $tag facts $(date) ==="
    grep -E "Total CPU offloaded parameters" "$logf" || true
    grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency" "$logf" || true
    grep -E "consumed memory|peak activation" "$logf" || true
    grep -E "Loaded .* configs" "$logf" || true
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
    free -g | head -2
  } | tee "<workdir>/results/facts-$tag.txt"
}
parse_boot_nums() {
  python3 - "$1" "<workdir>/results/nums-$1.env" <<'PY'
import re,sys
log=open(f"<workdir>/dsv41-vllm-{sys.argv[1]}.log",errors="replace").read()
off=re.search(r"Total CPU offloaded parameters:\s*([0-9.]+)", log)
kv=re.search(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", log)
tok=re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", log)
conc=re.search(r"Maximum concurrency for [0-9,]+ tokens per request:\s*([0-9.]+)", log)
loaded=re.search(r"Loaded (\d+) configs", log)
def g(m): return m.group(1).replace(",","") if m else ""
open(sys.argv[2],"w").write(f"OFF_GIB={g(off)}\nKV_GIB={g(kv)}\nKV_TOK={g(tok)}\nMAXCONC={g(conc)}\nLOADED={g(loaded)}\n")
print(open(sys.argv[2]).read())
PY
}
knee_vals() {
  python3 - "$1" <<'PY'
import json,sys
d=json.load(open(f"<workdir>/knee-{sys.argv[1]}.json"))
m={int(r[0]): r[1] for r in d["rows"]}
print(f"{m.get(1,float('nan')):.1f}/{m.get(8,float('nan')):.1f}/{m.get(16,float('nan')):.1f}")
print(f"{m.get(1,float('nan')):.4f}", file=sys.stderr)
PY
}
fixture_vals() {
  python3 - "$1" <<'PY'
import json,sys
d=json.load(open(f"<workdir>/agentfix-{sys.argv[1]}.json"))
r=d["res"]; acc=d.get("weighted_accept")
accs=f"{acc*100:.1f}%" if isinstance(acc,float) else "?"
print(f"{r.get('prose',{}).get('tok_s','?')}/{r.get('shell_ops',{}).get('tok_s','?')}\t{accs}")
PY
}
verdict_vs_821() {
  python3 - "$1" <<'PY'
import json,sys
d=json.load(open(f"<workdir>/knee-{sys.argv[1]}.json"))
c1={int(r[0]): r[1] for r in d["rows"]}[1]
ref=82.1
pct=(c1-ref)/ref*100
if c1 > ref*1.015: v="WIN"
elif abs(c1-ref)/ref <= 0.015: v="WASH"
elif c1 < ref: v="REGRESSION"
else: v="WASH"
sd=79.5
print(f"{v} C1={c1:.1f} vs 82.1 ({pct:+.1f}%) / vs same-day k5 79.5 ({(c1-sd)/sd*100:+.1f}%)")
PY
}
restore_ref() {
  log "RESTORE REF"
  docker ps -q --filter name=dsv41-vllm- | xargs -r docker stop || true
  echo never | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled >/dev/null || true
  drop_caches
  docker start "$REF"
  local t0=$SECONDS
  until curl -sf -m 3 http://127.0.0.1:30006/v1/models >/dev/null 2>&1; do
    sleep 20
    [[ $((SECONDS-t0)) -gt 1800 ]] && { log "RESTORE bind timeout"; return 1; }
  done
  log "REF rebound after $((SECONDS-t0))s"
}

log "PROFILE v12 campaign start (diagnostic; one boot; REF restored at end)"
check_safety
[[ -f "$SEED" ]] || sudo cp "$AT/$LIVE/autotune_configs.json" "$SEED"
running=$(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-' || true)
if [[ -n "$running" ]]; then log "stopping $running"; echo "$running" | while read -r n; do docker stop "$n"; done; fi
drop_caches
avail=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [[ "$avail" -lt 20 ]]; then die "available ${avail} < 20 after drop"; fi

tag=PROF-v12
PROF=<workdir>/prof-$tag; sudo rm -rf "$PROF"; mkdir -p "$PROF"
seed_watch "$tag" &
sw=$!
launched=$(date '+%H:%M')
# identical shapes to v12 -> autotune hash hit. Profiler dir is mounted at the same path inside the container.
TAG="$tag" OFFGB=60 UTIL=0.97 SEQS=16 CTX=1048576 SPEC=dspark:5 \
  EXTRA="--long-prefill-token-threshold 6144 --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$PROF --profiler-config.torch_profiler_with_stack=false --profiler-config.torch_profiler_use_gzip=true --profiler-config.torch_profiler_record_shapes=false --profiler-config.max_iterations=60" \
  DOCKER_ENV="-v $PROF:$PROF" MODEL="$MODEL" \
  bash launch-dsv41-vllm.sh
wait_bound "$tag"
wait "$sw" || true
record_facts "$tag"
parse_boot_nums "$tag" || true
if grep -E "Available KV cache memory" "<workdir>/dsv41-vllm-$tag.log" | grep -qE ' -'; then die "negative KV"; fi
log "SMOKE $tag"
if ! smoke; then die "smoke failed"; fi

# warm both categories before capturing
python3 - <<'PY'
import json, urllib.request
B="http://127.0.0.1:30006/v1/chat/completions"
def run(prompt, n=192):
    body={"model":"dsv41-flash-uva","messages":[{"role":"user","content":prompt}],"max_tokens":n,"temperature":0,"chat_template_kwargs":{"thinking":False},"ignore_eos":True}
    r=urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}); urllib.request.urlopen(r,timeout=600).read()
run("Write a detailed paragraph about the number 7, its history and uses. No lists.")
run("Write a bash script that finds all .log files under /var/log older than 7 days, compresses each with gzip, and prints a summary. Only code.")
PY
# capture: one C1 prose window, one C1 shell window, each ~150 decode tokens, plus one C8 prose window
capture() {  # name prompt concurrency
  local name=$1 prompt=$2 c=$3
  log "CAPTURE $name C$c"
  curl -sf -m 10 -X POST http://127.0.0.1:30006/start_profile >/dev/null || die "start_profile failed"
  python3 - "$prompt" "$c" <<'PY'
import json, sys, urllib.request, threading
B="http://127.0.0.1:30006/v1/chat/completions"; p=sys.argv[1]; c=int(sys.argv[2])
def run(i):
    body={"model":"dsv41-flash-uva","messages":[{"role":"user","content":p+f" (variant {i})"}],"max_tokens":150,"temperature":0,"chat_template_kwargs":{"thinking":False},"ignore_eos":True}
    r=urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}); urllib.request.urlopen(r,timeout=600).read()
ts=[threading.Thread(target=run,args=(i,)) for i in range(c)]
[t.start() for t in ts]; [t.join() for t in ts]
PY
  curl -sf -m 120 -X POST http://127.0.0.1:30006/stop_profile >/dev/null || die "stop_profile failed"
  sleep 15
  # newest trace dir -> tag it
  local newest; newest=$(ls -td "$PROF"/* 2>/dev/null | head -1)
  [[ -n "$newest" ]] && sudo mv "$newest" "$PROF/trace-$name-C$c" || log "no trace produced for $name"
  log "captured $name -> $(du -sh "$PROF/trace-$name-C$c" 2>/dev/null | cut -f1)"
}
capture prose "Write a detailed paragraph about the number 42, its history and uses. No lists." 1
capture shell "Write a bash script that rotates nginx logs weekly, keeps 8 archives, and emails a summary. Only code." 1
capture prose "Write a detailed paragraph about the number 42, its history and uses. No lists." 8
# knee on this boot so the profiler overhead is known
bash knee.sh "$tag" || true
c1=$(python3 -c 'import json;print(round(json.load(open("<workdir>/knee-PROF-v12.json"))["rows"][0][1],1))' 2>/dev/null || echo NA)
log "LEDGER | $tag | $launched | $(date '+%H:%M') | ${KV:-?} | ${OFFL:-?} | knee C1=$c1 (profiler on) | DIAGNOSTIC |"
docker stop "dsv41-vllm-$tag" || true
docker rename "dsv41-vllm-$tag" "dsv41-vllm-$tag-DIAGNOSTIC" || true
restore_ref
sudo chown -R milo:milo "$PROF" || true
log "PROFILE campaign done"

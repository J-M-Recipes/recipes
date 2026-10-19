#!/usr/bin/env bash
# cardS0_runner.sh — Card S0 ("Step 0"): exact-size pinned host allocations via
#   PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 on the v22 lane, bracketed CTL1 -> CAND -> CTL2, same window.
# 2026-10-18 ~09:00 CDT, Milo (hermes milo) subagent, claude-opus-5-5. Header/gate/measure = cardP1_runner.sh (today's template).
# Arms:  CTL1 = live v22 REF as-is (template convention, no reboot)
#        S0   = launch_v22.sh byte-identical except ONE added flag: -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024
#               (launch_v22.sh sets no PYTORCH_CUDA_ALLOC_CONF; image ENV sets none either -> value is exactly that string)
#               container dsv41-vllm-1017-S0-pinexact-EXP
#        CTL2 = v22 REF restarted and LEFT SERVING.
# Mechanism: torch 2.13 CachingHostAllocator rounds pinned allocs to pow2; allocs above pinned_max_cached_size_mb are never
#   rounded (string confirmed in the 0909 image's libc10.so). Live REF smaps: 2 x 128 GiB /dev/zero maps for 2 x 94.42 GiB
#   Engram + 40x1 GiB + 40x512 MiB hook cold maps -> expected ~67 GiB freed, same bytes read, same kernels.
# Per arm: boot facts (KV, Loaded N configs + hash, tune lines, hook rehome per-layer host_avail incl. layer 9, rehome done)
#   -> host-memory capture (pre-boot / bound / steady / steady-after-drop_caches: /proc/meminfo, cgroup memory.current+stat,
#      EngineCore /dev/zero (= cudaHostAlloc pinned) map Size/Rss sum + size histogram)
#   -> cold 64K prefill -> Ronan RT C1x3/C4/C16/C32 -> run_window_T1T2 -> gauntlet -> greedy parity (e2c_parity.py, 18 prompts, 16 non-tool).
# PRE-REGISTERED BARS (CAND vs MEAN of CTL1/CTL2):
#   WIN  = host pinned total (EngineCore /dev/zero map Size sum) at steady state down >= 40 GiB vs CTL mean
#          AND RT C1 >= -2% AND replay r2 >= -2% AND gauntlet 12/12
#          AND greedy non-tool parity CAND == CTL1 AND CAND == CTL2 (16/16 content+finish exact)
#          AND KV tokens unchanged (2,504,891).
#   FAIL = any decode -3% (RT C1/C4/C16 decode_agg or replay r2 below CTL mean by >3%) OR non-tool parity mismatch vs either CTL
#          (unless CTL1 vs CTL2 itself mismatches -> instrument noise, recorded) OR CAND boot death.
#   Otherwise NEUTRAL. Hash hit expected (env var does not enter kernel shapes); a live tune -> PROVISIONAL.
# Gate: waits for the P2 image pull (pull-p2-39983a5e.log ends EXIT=, no docker pull proc) + wait_quiet before ANY measurement/boot.
# Stop-and-keep only. Never :30003, never glmf-*/dsf*. Refuses foreign GPU containers. Does not touch the hook dir or run_window_T1T2.sh.
set -uo pipefail
D=$HOME/dsv41/m3; W=$HOME/dsv41/1017; R=$W/results-cardS0; mkdir -p "$R"
B=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v22-ksched1to8-BOUND-REF
CAND=dsv41-vllm-1017-S0-pinexact-EXP
ENVV='pinned_max_cached_size_mb:1024'
GDIR=$HOME/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
S=$HOME/pin-hot-experts/scripts; PROMPTS=$S/e5_parity_prompts.json
PULLLOG=$W/pull-p2-39983a5e.log
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 5; }
wait_bind(){ local n=$1 i
  for ((i=0;i<450;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $((i*10))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n"; return 1; }
    sleep 10
  done; log "TIMEOUT $n (75 min)"; return 1; }
gpu_foreign(){ local c out=""; for c in $(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-'); do
    docker inspect "$c" --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null | grep -qi gpu && out="$out $c"; done; echo "$out"; }
any_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '; }
archive_busy(){ ps -eo args | grep -E 'ark_bf1017\.sh|archive_1017\.sh|archive_sums|rsync .*miloark|sha256sum|hf download|docker pull' | grep -v grep | head -3 | cut -c1-120 | tr '\n' '|'; }
wait_quiet(){ local stage=$1 i a f
  for ((i=0;i<180;i++)); do
    a=$(archive_busy); f=$(any_foreign)
    [[ -z $a && -z ${f// /} ]] && { [[ $i -gt 0 ]] && log "QUIET for $stage after $((i*30))s"; return 0; }
    (( i % 10 == 0 )) && log "WAIT $stage: archive=[${a}] foreign=[${f}]"
    sleep 30
  done; log "WAIT $stage: still busy after 90 min: archive=[$(archive_busy)] foreign=[$(any_foreign)]"; return 1; }
wait_pull(){ local i
  for ((i=0;i<360;i++)); do
    if tail -n 3 "$PULLLOG" 2>/dev/null | grep -q '^EXIT='; then
      ps -eo args | grep -E '^docker pull' | grep -v grep >/dev/null || { log "PULL done: $(grep '^EXIT=' "$PULLLOG" | tail -1)"; return 0; }
    fi
    (( i % 10 == 0 )) && log "WAIT pull: $(tail -c 120 "$PULLLOG" 2>/dev/null | tr '\n' ' ')"
    sleep 30
  done; log "WAIT pull: not finished after 3 h"; return 1; }
strays(){ log "strays: $(ps -eo pid,rss,args --sort=-rss | sed -n 2,5p | awk '{printf "%s %.1fGiB %s %s|", $1, $2/1048576, $3, $4}') MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/1048576}' /proc/meminfo)"; }
hostmem(){ # tag phase [container]
  local tag=$1 ph=$2 c=${3:-} f="$R/hostmem-$1-$2.txt" id cg pid
  { echo "date $(date '+%F %T %Z') tag=$tag phase=$ph container=${c:--}"
    grep -E '^(MemTotal|MemFree|MemAvailable|Cached|Shmem|Mlocked|Unevictable|AnonPages|Mapped|Dirty|Writeback|CommitLimit|Committed_AS):' /proc/meminfo | awk '{printf "meminfo %s %.2f GiB\n",$1,$2/1048576}'
    if [[ -n $c ]] && docker ps --format '{{.Names}}' | grep -qx "$c"; then
      id=$(docker inspect -f '{{.Id}}' "$c"); cg=/sys/fs/cgroup/system.slice/docker-$id.scope
      [[ -d $cg ]] || cg=$(find /sys/fs/cgroup -maxdepth 3 -name "*$id*" 2>/dev/null | head -1)
      if [[ -n $cg && -r $cg/memory.current ]]; then
        awk '{printf "cgroup memory.current %.2f GiB\n",$1/1073741824}' "$cg/memory.current"
        grep -E '^(anon|file|shmem|file_mapped|unevictable|kernel) ' "$cg/memory.stat" | awk '{printf "cgroup %s %.2f GiB\n",$1,$2/1073741824}'
      fi
      pid=$(docker top "$c" -eo pid,rss,args 2>/dev/null | grep 'VLLM::EngineCore' | awk '{print $1}' | head -1)
      if [[ -n $pid ]]; then
        echo "enginecore_pid $pid"
        sudo -n awk '/^[0-9a-f]+-[0-9a-f]+ /{dz=($0 ~ /\/dev\/zero/); if(dz)n++} dz&&/^Size:/{s+=$2} dz&&/^Rss:/{r+=$2} END{printf "pinned_devzero maps=%d size_gib=%.2f rss_gib=%.2f\n",n,s/1048576,r/1048576}' /proc/$pid/smaps
        sudo -n awk '/^[0-9a-f]+-[0-9a-f]+ /{dz=($0 ~ /\/dev\/zero/)} dz&&/^Size:/{print $2}' /proc/$pid/smaps | sort -n | uniq -c | awk '{printf "pinned_hist count=%d size_kib=%d\n",$1,$2}'
        sudo -n grep -E '^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Dirty|Locked):' /proc/$pid/smaps_rollup | awk '{printf "rollup %s %.2f GiB\n",$1,$2/1048576}'
      fi
    fi
  } > "$f" 2>&1
  log "HOSTMEM $tag $ph: $(grep -E 'MemAvailable|pinned_devzero|cgroup shmem|cgroup memory.current' "$f" | tr '\n' ';' | tr -s ' ' | cut -c1-300)"; }
boot_facts(){ # tag name since
  local tag=$1 name=$2 since=$3
  if [[ -n $since ]]; then docker logs --since "$since" "$name" > "$R/boot-$tag.log" 2>&1; else docker logs "$name" > "$R/boot-$tag.log" 2>&1; fi
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -E "Loaded [0-9]+ configs|Saved [0-9]+ configs|Autotuning process|GPU KV cache size|Available KV cache memory|CUDA graph memory profiling|Graph capturing finished|CUDA graph pool memory|rehome layer=|rehome done|PIN_HOT installed|PIN_HOT abort|Engram table offloaded|Model loading took|model loading took|PYTORCH_CUDA_ALLOC_CONF|pinned_max|Unrecognized" | grep -vE 'it/s\]' | cut -c1-400 > "$R/facts-$tag.txt"
  log "FACTS $tag: loaded=$(grep -c 'Loaded [0-9]* configs' "$R/facts-$tag.txt") saved=$(grep -c 'Saved [0-9]* configs' "$R/facts-$tag.txt") | $(grep 'GPU KV cache size' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //') | $(grep 'rehome layer=9 ' "$R/facts-$tag.txt" | tail -1 | grep -oE 'host_avail=[0-9.]+GiB') L9 | $(grep 'rehome done' "$R/facts-$tag.txt" | tail -1 | sed 's/.*rehome done //' | cut -c1-120)"
}
kv_tokens(){ grep 'GPU KV cache size' "$R/facts-$1.txt" | tail -1 | sed -E 's/.*GPU KV cache size: ([0-9,]+) tokens.*/\1/' | tr -d ','; }
cold64(){ local tag=$1
  BASE_URL=$U/v1 MODEL=$M API_KEY=none SIZES=65536 N=1 THINKING=0 timeout 900 python3 $HOME/dsv41/cold_prefill_probe.py > "$R/cold64k-$tag.txt" 2>&1 || log "cold64k $tag nonzero rc"
  log "COLD64K $tag: $(grep PREFILL "$R/cold64k-$tag.txt" | tail -1)"; }
rt_bench(){ local tag=$1 N O r
  N="s0-$1-$(date +%s)"; O="$R/rt-$1"; mkdir -p "$O"
  for r in 1 2 3; do python3 $B $U $M $P/eval-c1-r$r-try1.jsonl $O/c1-r$r.json 1 --warm $P/eval-c1-r$r-try1-warm.jsonl --nonce $N-c1-r$r > "$O/c1-r$r.stdout" 2>&1; done
  python3 $B $U $M $P/eval-c16-try1.jsonl $O/c16.json 16 --warm $P/eval-c16-try1-warm.jsonl --nonce $N-c16 > "$O/c16.stdout" 2>&1
  python3 $B $U $M $P/eval-c16-try1.jsonl $O/c4.json 4 --head 8 --nonce $N-c4 > "$O/c4.stdout" 2>&1
  python3 $B $U $M $P/eval-c16-try1.jsonl $O/c32.json 32 --nonce $N-c32 > "$O/c32.stdout" 2>&1
  log "RT $tag: $(python3 - "$O" <<'PY'
import json,sys
O=sys.argv[1]; out=[]
for k in ('c1-r1','c1-r2','c1-r3','c4','c16','c32'):
    try:
        s=json.load(open(f'{O}/{k}.json'))['summary']; out.append(f"{k} agg={s['decode_agg']} out={s['output_tok_s']} ttft={s['median_ttft_s']} acc={s.get('acceptance_len')}")
    except Exception as e: out.append(f'{k} ERR {e}')
print(' ; '.join(out))
PY
)"; }
measure(){ # tag name
  local tag=$1 name=$2
  cold64 "$tag"
  rt_bench "$tag"
  bash $HOME/dsv41/run_window_T1T2.sh "S0-$tag" > "$R/nohup-$tag.log" 2>&1
  mv $HOME/dsv41/overnight-T1T2/*"S0-$tag"[-.]* "$R"/ 2>/dev/null
  mv $HOME/dsv41/overnight-T1T2/*"S0-$tag" "$R"/ 2>/dev/null
  log "WINDOW $tag: $(grep -h 'conc=' "$R/window-S0-$tag.log" 2>/dev/null | grep -E 'r2\]' | tr '\n' ';' | cut -c1-300)"
  log "REPLAY $tag: $(grep -h 'REPLAY' "$R/window-S0-$tag.log" 2>/dev/null | cut -c1-120 | tr '\n' ';')"
  rm -rf "$R/gauntlet-$tag"
  (cd "$GDIR" && python3 run_suite.py --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "s0-$tag" --cache-salt "s0-$tag" --sandbox-image "$GIMG" --workers 4) > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"
  python3 "$S/e2c_parity.py" "S0-$tag" "$PROMPTS" "$R/parity" > "$R/parity-log-$tag.txt" 2>&1 || log "parity $tag nonzero"
  log "PARITY $tag captured: $(ls "$R/parity/" 2>/dev/null | grep -c "S0-$tag")"
  hostmem "$tag" steady "$name"
  drop_caches; sleep 5
  hostmem "$tag" steady-dc "$name"
  docker logs "$name" > "$R/postlog-$tag.log" 2>&1; grep -cE 'ValueError:|Traceback|ERROR|RuntimeError:' "$R/postlog-$tag.log" > "$R/errcount-$tag.txt"
}

log "===== CARD S0 START ====="
df -h / | tail -1 | sed 's/^/df /'
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running for CTL1"; exit 5; }
docker inspect "$REF" --format '{{json .Config.Env}}' | grep -q PYTORCH_CUDA_ALLOC_CONF && log "WARN: REF already carries PYTORCH_CUDA_ALLOC_CONF"
wait_pull || { log "ABORT: pull never finished; lane untouched (REF serving)"; exit 6; }
wait_quiet CTL1 || { log "ABORT: host never quiet; lane untouched (REF serving)"; exit 6; }

# ---- CTL1: the live lane, as-is ----
boot_facts CTL1 "$REF" ""
KV_CTL1=$(kv_tokens CTL1); KV_CTL1=${KV_CTL1:-2504891}
log "boot->ready CTL1 live (no reboot); KV $KV_CTL1"
strays
hostmem CTL1 bound "$REF"
measure CTL1 "$REF"

# ---- CAND: v22 + PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 ----
cand_rc=0
if wait_quiet S0; then
  sed -e "s#-e VLLM_LOGGING_LEVEL=INFO #-e PYTORCH_CUDA_ALLOC_CONF=$ENVV -e VLLM_LOGGING_LEVEL=INFO #" $W/launch_v22.sh > "$R/launch_S0.sh"
  diff $W/launch_v22.sh "$R/launch_S0.sh" > "$R/launch_S0.diff"
  if [[ $(grep -c '^[<>]' "$R/launch_S0.diff") -ne 2 ]] || ! grep -q -- "-e PYTORCH_CUDA_ALLOC_CONF=$ENVV -e VLLM" "$R/launch_S0.sh"; then
    log "ABORT S0: sed did not apply as a one-line change"; cand_rc=1
  else
    stop_all; drop_caches; strays
    hostmem S0 pre ""
    docker ps -a --format '{{.Names}}' | grep -qx "$CAND" && { docker rename "$CAND" "$CAND-STALE-$(date +%H%M)"; log "renamed stale $CAND"; }
    t0=$(date +%s)
    if ! bash "$R/launch_S0.sh" "$CAND"; then log "launch failed S0"; cand_rc=1
    else
      log "launched $CAND env PYTORCH_CUDA_ALLOC_CONF=$ENVV"
      docker inspect "$CAND" --format '{{json .Config.Env}}' | tr ',' '\n' | grep PYTORCH > "$R/env-S0.txt"
      if ! wait_bind "$CAND"; then
        boot_facts S0 "$CAND" ""
        log "BOOTFAIL S0 tail: $(tr '\r' '\n' < "$R/boot-S0.log" | grep -E 'ValueError:|Traceback|ERROR|RuntimeError:|OutOfMemory|PIN_HOT abort|MemAvailable' | tail -3 | cut -c1-200 | tr '\n' '|')"
        docker stop "$CAND" >/dev/null 2>&1; docker rename "$CAND" "$CAND-BOOT-FAIL"; cand_rc=1
      else
        log "boot->ready S0 $(( $(date +%s) - t0 ))s"
        boot_facts S0 "$CAND" ""
        hostmem S0 bound "$CAND"
        log "KV S0: $(kv_tokens S0) tokens (CTL1 $KV_CTL1)"
        measure S0 "$CAND"
      fi
    fi
  fi
else
  log "SKIP S0: host never quiet"; cand_rc=3
fi
log "S0 rc=$cand_rc"

# ---- CTL2: v22 REF back, left serving ----
wait_quiet CTL2 || log "WARN: host not quiet before CTL2 — booting REF anyway (lane must return)"
stop_all; drop_caches; strays
hostmem CTL2 pre ""
t0=$(date +%s); since=$(date -u +%FT%TZ); docker start "$REF" >/dev/null
if wait_bind "$REF"; then log "boot->ready CTL2 $(( $(date +%s) - t0 ))s"; boot_facts CTL2 "$REF" "$since"; hostmem CTL2 bound "$REF"; measure CTL2 "$REF"
else log "CTL2 REF FAILED TO BIND — lane dark, needs hands"; fi

# ---- verdict + suffix the EXP container ----
python3 $W/cardS0_verdict.py "$R" > "$R/VERDICT.md" 2> "$R/verdict.err"
V=$(grep -m1 '^SUFFIX' "$R/VERDICT.md" | awk '{print $2}')
docker ps -a --format '{{.Names}}' | grep -qx "$CAND" && { docker stop "$CAND" >/dev/null 2>&1; docker rename "$CAND" "$CAND-${V:-DONE}"; log "kept $CAND-${V:-DONE}"; }
grep -hE ' (RT|WINDOW|REPLAY|GAUNTLET|FACTS|KV|COLD64K|HOSTMEM|PARITY|boot->ready|BOOTFAIL) ' "$R/runner.log" > "$R/SUMMARY.txt"
log "===== CARD S0 END (lane = $REF; bound=$(curl -s -m 3 $U/v1/models | grep -c $M)) ====="

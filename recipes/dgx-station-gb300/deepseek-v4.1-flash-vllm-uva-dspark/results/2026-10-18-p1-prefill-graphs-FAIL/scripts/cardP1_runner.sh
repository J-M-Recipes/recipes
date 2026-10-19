#!/usr/bin/env bash
# cardP1_runner.sh — Card P1: prefill CUDA graphs to 8,192 tokens on the v22 lane, bracketed CTL1 -> CAND -> CTL2, same window.
# 2026-10-18 ~04:50 CDT, Milo (hermes milo) subagent. Plan: PLAN-el8-ideas-2026-10-18.md §P1. Header = cardS32_confirm_runner.sh / cardM3G_runner.sh.
# Arms:  CTL1 = live v22 REF as-is (template convention, no reboot)
#        CG8K = launch_v22.sh identical except --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 256 512 1024 2048 4096 8192
#               (container dsv41-vllm-1017-P1-cg8192-EXP)
#        CG2K = only if CG8K KV loss > 0.5M tokens (or CG8K fails to bind): list capped at 2048 (dsv41-vllm-1017-P1-cg2048-EXP)
#        CTL2 = v22 REF restarted and LEFT SERVING.
# Per arm: bind -> boot facts (KV tokens, Available KV memory, graph capture/pool memory, autotune Loaded vs live tune, boot s)
#          -> cold 64K prefill (cold_prefill_probe.py, N=1) -> Ronan RT harness C1x3/C4/C16/C32 -> run_window_T1T2 -> gauntlet.
# PRE-REGISTERED BARS (CAND vs MEAN of CTL1/CTL2):
#   WIN  = TTFT 8K (RT C1 median TTFT, mean of 3) <= 0.55 s AND KV loss <= 0.3M tokens AND RT C1 >= -2% AND replay r2 >= -2%.
#   FAIL = KV loss > 0.5M tokens OR decode -3% (any of RT C1/C4/C16 below control mean by more than 3%).
#   Otherwise NEUTRAL. First boot on a new autotune hash = PROVISIONAL. Promotion is James's call (GSM8K-200 before any promotion).
# Stop-and-keep only. Never :30003. Refuses foreign GPU containers. Waits for archive/CIFS jobs to finish before any boot.
set -uo pipefail
D=$HOME/dsv41/m3; W=$HOME/dsv41/1017; R=$W/results-cardP1; mkdir -p "$R"
B=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v22-ksched1to8-BOUND-REF
C8K=dsv41-vllm-1017-P1-cg8192-EXP
C2K=dsv41-vllm-1017-P1-cg2048-EXP
GDIR=$HOME/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
BASESZ='1 2 4 6 8 12 16 18 24 32 40 48 64 96 128'
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
wait_quiet(){ # stage  — never boot during archive/CIFS copies; wait out transient foreign (CPU) containers
  local stage=$1 i a f
  for ((i=0;i<180;i++)); do
    a=$(archive_busy); f=$(any_foreign)
    [[ -z $a && -z ${f// /} ]] && { [[ $i -gt 0 ]] && log "QUIET for $stage after $((i*30))s"; return 0; }
    (( i % 10 == 0 )) && log "WAIT $stage: archive=[${a}] foreign=[${f}]"
    sleep 30
  done; log "WAIT $stage: still busy after 90 min: archive=[$(archive_busy)] foreign=[$(any_foreign)]"; return 1; }
strays(){ log "strays: $(ps -eo pid,rss,args --sort=-rss | sed -n 2,5p | awk '{printf "%s %.1fGiB %s %s|", $1, $2/1048576, $3, $4}') MemAvailable=$(awk '/MemAvailable/{printf "%.1fGiB",$2/1048576}' /proc/meminfo)"; }
boot_facts(){ # tag name since
  local tag=$1 name=$2 since=$3
  if [[ -n $since ]]; then docker logs --since "$since" "$name" > "$R/boot-$tag.log" 2>&1; else docker logs "$name" > "$R/boot-$tag.log" 2>&1; fi
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -E "Loaded [0-9]+ configs|Autotuning process|GPU KV cache size|Available KV cache memory|CUDA graph memory profiling|Graph capturing finished|CUDA graph pool memory|re-homed|rehome done|PIN_HOT installed|Model loading took|model loading took|cudagraph_capture_sizes|max_cudagraph_capture_size" | grep -vE 'it/s\]' | cut -c1-400 > "$R/facts-$tag.txt"
  log "FACTS $tag: loaded=$(grep -c 'Loaded [0-9]* configs' "$R/facts-$tag.txt") tune_starts=$(grep -c 'Autotuning process starts' "$R/facts-$tag.txt") | $(grep 'GPU KV cache size' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //') | $(grep 'CUDA graph pool memory' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //') | $(grep 'Available KV cache memory' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //')"
}
kv_tokens(){ grep 'GPU KV cache size' "$R/facts-$1.txt" | tail -1 | sed -E 's/.*GPU KV cache size: ([0-9,]+) tokens.*/\1/' | tr -d ','; }
cold64(){ local tag=$1
  BASE_URL=$U/v1 MODEL=$M API_KEY=none SIZES=65536 N=1 THINKING=0 timeout 900 python3 $HOME/dsv41/cold_prefill_probe.py > "$R/cold64k-$tag.txt" 2>&1 || log "cold64k $tag nonzero rc"
  log "COLD64K $tag: $(grep PREFILL "$R/cold64k-$tag.txt" | tail -1)"; }
rt_bench(){ local tag=$1 N O r
  N="p1-$1-$(date +%s)"; O="$R/rt-$1"; mkdir -p "$O"
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
  bash $HOME/dsv41/run_window_T1T2.sh "P1-$tag" > "$R/nohup-$tag.log" 2>&1
  mv $HOME/dsv41/overnight-T1T2/*"P1-$tag"[-.]* "$R"/ 2>/dev/null
  mv $HOME/dsv41/overnight-T1T2/*"P1-$tag" "$R"/ 2>/dev/null
  log "WINDOW $tag: $(grep -h 'conc=' "$R/window-P1-$tag.log" 2>/dev/null | grep -E 'r2\]' | tr '\n' ';' | cut -c1-300)"
  log "REPLAY $tag: $(grep -h 'REPLAY' "$R/window-P1-$tag.log" 2>/dev/null | cut -c1-120 | tr '\n' ';')"
  rm -rf "$R/gauntlet-$tag"
  (cd "$GDIR" && python3 run_suite.py --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "p1-$tag" --cache-salt "p1-$tag" --sandbox-image "$GIMG" --workers 4) > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"
  docker logs "$name" > "$R/postlog-$tag.log" 2>&1; grep -cE 'ValueError:|Traceback|ERROR|RuntimeError:' "$R/postlog-$tag.log" > "$R/errcount-$tag.txt"
}
boot_cand(){ # tag name sizes  -> 0 ok-measured, 2 kvfail, 1 boot fail, 3 host never quiet
  local tag=$1 name=$2 sizes=$3 t0 since kv loss
  wait_quiet "$tag" || return 3
  stop_all; drop_caches; strays
  sed -e "s/--cudagraph-capture-sizes $BASESZ /--cudagraph-capture-sizes $sizes /" $W/launch_v22.sh > "$R/launch_$tag.sh"
  grep -q -- "--cudagraph-capture-sizes $sizes " "$R/launch_$tag.sh" || { log "ABORT $tag: sed did not apply"; return 1; }
  docker ps -a --format '{{.Names}}' | grep -qx "$name" && { docker rename "$name" "$name-STALE-$(date +%H%M)"; log "renamed stale $name"; }
  t0=$(date +%s); since=$(date -u +%FT%TZ)
  bash "$R/launch_$tag.sh" "$name" || { log "launch failed $tag"; return 1; }
  log "launched $name sizes=[$sizes]"
  if ! wait_bind "$name"; then
    docker logs "$name" > "$R/boot-$tag.log" 2>&1; boot_facts "$tag" "$name" ""
    log "BOOTFAIL $tag tail: $(tr '\r' '\n' < "$R/boot-$tag.log" | grep -E 'ValueError:|Traceback|ERROR|RuntimeError:|OutOfMemory|host.?mem|MemAvailable' | tail -3 | cut -c1-200 | tr '\n' '|')"
    docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-BOOT-FAIL"; return 1; fi
  log "boot->ready $tag $(( $(date +%s) - t0 ))s"
  boot_facts "$tag" "$name" ""
  kv=$(kv_tokens "$tag"); loss=$(( KV_CTL1 - ${kv:-0} ))
  log "KV $tag: $kv tokens (CTL1 $KV_CTL1, loss $loss)"
  if (( loss > 500000 )); then log "KVFAIL $tag: loss $loss > 500000 — skipping benches"; docker stop "$name" >/dev/null; docker rename "$name" "$name-KVFAIL"; return 2; fi
  measure "$tag" "$name"
  return 0; }

log "===== CARD P1 START ====="
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running for CTL1"; exit 5; }
docker inspect "$REF" --format '{{json .Args}}' | grep -q '"128","--host"' || log "WARN: REF capture list not the v22 1..128 list"
wait_quiet CTL1 || { log "ABORT: host never quiet; lane untouched (REF serving)"; exit 6; }

# ---- CTL1: the live lane, as-is ----
boot_facts CTL1 "$REF" ""
KV_CTL1=$(kv_tokens CTL1); KV_CTL1=${KV_CTL1:-2504891}
log "boot->ready CTL1 live (no reboot); KV $KV_CTL1"
strays
measure CTL1 "$REF"

# ---- CAND: capture to 8192; fallback capped at 2048 if KV collapses or boot fails ----
boot_cand CG8K "$C8K" "$BASESZ 256 512 1024 2048 4096 8192"; rc=$?
log "CG8K rc=$rc"
if (( rc == 1 || rc == 2 )); then
  boot_cand CG2K "$C2K" "$BASESZ 256 512 1024 2048"; rc2=$?; log "CG2K rc=$rc2"
fi

# ---- CTL2: v22 REF back, left serving ----
wait_quiet CTL2 || log "WARN: host not quiet before CTL2 — booting REF anyway (lane must return)"
stop_all; drop_caches; strays
t0=$(date +%s); since=$(date -u +%FT%TZ); docker start "$REF" >/dev/null
if wait_bind "$REF"; then log "boot->ready CTL2 $(( $(date +%s) - t0 ))s"; boot_facts CTL2 "$REF" "$since"; measure CTL2 "$REF"
else log "CTL2 REF FAILED TO BIND — lane dark, needs hands"; fi

# ---- verdict + suffix the EXP containers ----
python3 $W/cardP1_verdict.py "$R" > "$R/VERDICT.md" 2> "$R/verdict.err"
V=$(grep -m1 '^SUFFIX' "$R/VERDICT.md" | awk '{print $2}')
for n in "$C8K" "$C2K"; do docker ps -a --format '{{.Names}}' | grep -qx "$n" && { docker stop "$n" >/dev/null 2>&1; docker rename "$n" "$n-${V:-DONE}"; log "kept $n-${V:-DONE}"; }; done
grep -hE ' (RT|WINDOW|REPLAY|GAUNTLET|FACTS|KV|COLD64K|boot->ready|KVFAIL|BOOTFAIL) ' "$R/runner.log" > "$R/SUMMARY.txt"
log "===== CARD P1 END (lane = $REF; bound=$(curl -s -m 3 $U/v1/models | grep -c $M)) ====="

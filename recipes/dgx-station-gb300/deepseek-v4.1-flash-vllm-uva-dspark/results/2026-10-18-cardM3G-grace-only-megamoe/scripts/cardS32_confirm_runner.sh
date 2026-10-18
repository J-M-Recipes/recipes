#!/usr/bin/env bash
# cardS32_confirm_runner.sh — confirm V22S32 (v22 launch, 32 seqs, ksched [[1,8,5],[9,32,1]]) with a bracketed CTL->CAND->CTL.
# 2026-10-18 ~01:15 CDT, Milo (hermes milo). James: "yes" to the confirm boot. Autotune hash for seqs32 cached from Card M3G.
# Bar (pre-registered, same as card M3G): C1/C4/C16 RT within -3% of the control mean AND C32 output tok/s > control; replay r2 >= controls -3%.
# Ends with v22 REF (24 seqs) serving — promotion is James's call.
set -uo pipefail
D=$HOME/dsv41/m3; R=$D/results-cardS32; mkdir -p "$R"
B=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v22-ksched1to8-BOUND-REF
S32=dsv41-vllm-1017-v22-seqs32-EXP
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 5; }
wait_bind(){ local n=$1 i
  for ((i=0;i<240;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $((i*10))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n"; return 1; }
    sleep 10
  done; log "TIMEOUT $n"; return 1; }
rt_bench(){ local tag=$1 N="s32-$1-$(date +%s)" O="$R/rt-$1"; mkdir -p "$O"
  for r in 1 2 3; do python3 $B $U $M $P/eval-c1-r$r-try1.jsonl $O/c1-r$r.json 1 --warm $P/eval-c1-r$r-try1-warm.jsonl --nonce $N-c1-r$r; done
  python3 $B $U $M $P/eval-c16-try1.jsonl $O/c16.json 16 --warm $P/eval-c16-try1-warm.jsonl --nonce $N-c16
  python3 $B $U $M $P/eval-c16-try1.jsonl $O/c4.json 4 --head 8 --nonce $N-c4
  python3 $B $U $M $P/eval-c16-try1.jsonl $O/c32.json 32 --nonce $N-c32; }
measure(){ local tag=$1 name=$2
  docker logs "$name" 2>&1 | grep -E "Loaded [0-9]+ configs|GPU KV cache size|Autotuning process" > "$R/facts-$tag.txt"
  log "FACTS $tag: autotune_loaded=$(grep -c 'Loaded' "$R/facts-$tag.txt") $(grep -m1 'GPU KV cache size' "$R/facts-$tag.txt" | sed 's/.*INFO//')"
  rt_bench "$tag"
  bash $HOME/dsv41/run_window_T1T2.sh "S32-$tag" > "$R/nohup-$tag.log" 2>&1
  mv $HOME/dsv41/overnight-T1T2/*"S32-$tag"* "$R"/ 2>/dev/null
  log "WINDOW $tag: $(grep -h 'conc=' "$R/window-S32-$tag.log" 2>/dev/null | grep -E 'r2\]' | tr '\n' ';' | cut -c1-300)"
  log "REPLAY $tag: $(grep -h 'REPLAY' "$R/window-S32-$tag.log" 2>/dev/null | cut -c1-120 | tr '\n' ';')"; }
boot(){ local name=$1 t0; stop_all; drop_caches; t0=$(date +%s); docker start "$name" >/dev/null
  if wait_bind "$name"; then log "boot->ready $name $(( $(date +%s) - t0 ))s"; return 0; fi; return 1; }

log "===== CARD S32 CONFIRM START ====="
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running"; exit 5; }
measure CTL1 "$REF"                                  # live lane as-is
boot "$S32" && measure CAND "$S32" || log "CAND FAILED"
boot "$REF" && measure CTL2 "$REF" || log "CTL2 REF FAILED TO BIND — lane dark, needs hands"
log "===== CARD S32 CONFIRM END (lane = $REF) ====="

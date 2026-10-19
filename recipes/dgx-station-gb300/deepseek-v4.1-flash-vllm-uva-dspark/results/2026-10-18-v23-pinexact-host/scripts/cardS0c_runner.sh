#!/usr/bin/env bash
# cardS0c_runner.sh — Card S0c: one same-window C6–C8 confirm of the S0 candidate, then PROMOTE (pass) or restore v22 (fail).
# 2026-10-18 ~10:10 CDT, Milo (hermes milo) subagent, claude-opus-5-5. Header/gates/hostmem/boot_facts = cardS0_runner.sh.
# James authorized promotion of Card S0 conditional on this one confirm window.
# Arms:  A = live v22 REF as-is (no reboot): knee6 r1, knee6 r2, knee r1, knee r2 (informational: the S0 caveat instrument), RT C1x3 (kbench)
#        B = stop REF, drop_caches, `docker start dsv41-vllm-1017-S0-pinexact-EXP-WIN` (exact v22 launch + one env
#            PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024), wait /v1/models, same cells in the same order.
# PRE-REGISTERED BAR (B vs A, knee6 r2 agg; each r2 agg = mean of knee6's two internal reps):
#   PASS = C8 r2 >= -2.0%  AND  C6 r2 >= -2.0%  AND  RT C1 (mean of 3 kbench C1 runs) >= -2.0%  AND B bound.
#   Anything else = FAIL. knee.sh r2 C8 is recorded, not gated.
# PASS: keep B serving, rename B -> dsv41-vllm-1017-v23-pinexact-BOUND-REF, REF -> dsv41-vllm-1017-v22-ksched1to8-RETIRED-REF
#       (stopped, kept), write launch_v23.sh (not run), gauntlet once on the new REF.
# FAIL: stop B, rename -> dsv41-vllm-1017-S0-pinexact-EXP-C8FAIL, drop_caches, docker start v22 REF, wait /v1/models.
# Ends with exactly one dsv41 container serving :30006. Stop-and-keep only (no rm). Never :30003, never glmf-*/dsf*.
# Refuses foreign GPU containers. Does not touch the hook dir or run_window_T1T2.sh.
set -uo pipefail
D=$HOME/dsv41; W=$D/1017; R=$W/results-cardS0c; mkdir -p "$R"
B=$D/m3/bench/kbench_chat_ds.py; P=$D/m3/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v22-ksched1to8-BOUND-REF
REF_RET=dsv41-vllm-1017-v22-ksched1to8-RETIRED-REF
CAND=dsv41-vllm-1017-S0-pinexact-EXP-WIN
V23=dsv41-vllm-1017-v23-pinexact-BOUND-REF
CFAIL=dsv41-vllm-1017-S0-pinexact-EXP-C8FAIL
ENVV='pinned_max_cached_size_mb:1024'
GDIR=$HOME/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
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
  for ((i=0;i<60;i++)); do
    a=$(archive_busy); f=$(any_foreign)
    [[ -z $a && -z ${f// /} ]] && { [[ $i -gt 0 ]] && log "QUIET for $stage after $((i*30))s"; return 0; }
    (( i % 10 == 0 )) && log "WAIT $stage: archive=[${a}] foreign=[${f}]"
    sleep 30
  done; log "WAIT $stage: still busy after 30 min"; return 1; }
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
      fi
    fi
  } > "$f" 2>&1
  log "HOSTMEM $tag $ph: $(grep -E 'MemAvailable|pinned_devzero|cgroup shmem|cgroup memory.current' "$f" | tr '\n' ';' | tr -s ' ' | cut -c1-300)"; }
boot_facts(){ # tag name since
  local tag=$1 name=$2 since=$3
  if [[ -n $since ]]; then docker logs --since "$since" "$name" > "$R/boot-$tag.log" 2>&1; else docker logs "$name" > "$R/boot-$tag.log" 2>&1; fi
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -E "Loaded [0-9]+ configs|Saved [0-9]+ configs|Autotuning process|GPU KV cache size|Available KV cache memory|rehome layer=|rehome done|PIN_HOT installed|PIN_HOT abort|Engram table offloaded|Model loading took|model loading took|PYTORCH_CUDA_ALLOC_CONF|pinned_max|Unrecognized" | grep -vE 'it/s\]' | cut -c1-400 > "$R/facts-$tag.txt"
  log "FACTS $tag: loaded=$(grep -c 'Loaded [0-9]* configs' "$R/facts-$tag.txt") saved=$(grep -c 'Saved [0-9]* configs' "$R/facts-$tag.txt") | $(grep 'GPU KV cache size' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //') | L9 $(grep 'rehome layer=9 ' "$R/facts-$tag.txt" | tail -1 | grep -oE 'host_avail=[0-9.]+GiB') | min $(grep -oE 'host_avail=[0-9.]+' "$R/facts-$tag.txt" | cut -d= -f2 | sort -n | head -1)GiB | $(grep 'rehome done' "$R/facts-$tag.txt" | tail -1 | sed 's/.*rehome done //' | cut -c1-120)"
}
cells(){ # tag  — identical cell order in both arms
  local tag=$1 k O N r
  for k in r1 r2; do
    bash $D/knee6.sh "S0c-$tag-k6$k" > "$R/knee6-$tag-$k.log" 2>&1 || log "knee6 $tag $k rc!=0"
    mv -f $D/knee-"S0c-$tag-k6$k".json "$R/knee6-$tag-$k.json" 2>/dev/null
    log "KNEE6 $tag $k: $(grep -E 'conc= ?(6|8):' "$R/knee6-$tag-$k.log" | sed 's/  */ /g' | tr '\n' ';')"
  done
  for k in r1 r2; do
    bash $D/knee.sh "S0c-$tag-kn$k" > "$R/knee-$tag-$k.log" 2>&1 || log "knee $tag $k rc!=0"
    mv -f $D/knee-"S0c-$tag-kn$k".json "$R/knee-$tag-$k.json" 2>/dev/null
    log "KNEE $tag $k: $(grep -E 'conc= ?8:' "$R/knee-$tag-$k.log" | sed 's/  */ /g' | tr '\n' ';')"
  done
  N="s0c-$tag-$(date +%s)"; O="$R/rt-$tag"; mkdir -p "$O"
  for r in 1 2 3; do python3 $B $U $M $P/eval-c1-r$r-try1.jsonl $O/c1-r$r.json 1 --warm $P/eval-c1-r$r-try1-warm.jsonl --nonce $N-c1-r$r > "$O/c1-r$r.stdout" 2>&1; done
  log "RT $tag: $(python3 -c "
import json
o=[]
for k in ('c1-r1','c1-r2','c1-r3'):
    try: s=json.load(open('$O/'+k+'.json'))['summary']; o.append(f\"{k} agg={s['decode_agg']} acc={s.get('acceptance_len')}\")
    except Exception as e: o.append(f'{k} ERR {e}')
print(' ; '.join(o))")"
}
gauntlet(){ local tag=$1
  rm -rf "$R/gauntlet-$tag"
  (cd "$GDIR" && python3 run_suite.py --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "s0c-$tag" --cache-salt "s0c-$tag" --sandbox-image "$GIMG" --workers 4) > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"; }
final_state(){ local up; up=$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')
  log "FINAL dsv41 running=[${up}] count=$(docker ps --format '{{.Names}}' | grep -cE '^dsv41-') bound=$(curl -s -m 3 $U/v1/models | grep -c $M)"; }
restore_v22(){ # used on FAIL or on any B failure
  local why=$1
  log "RESTORE v22 ($why)"
  stop_all; drop_caches; strays
  if docker ps -a --format '{{.Names}}' | grep -qx "$CAND"; then docker rename "$CAND" "$CFAIL" && log "renamed $CAND -> $CFAIL (stopped, kept)"; fi
  local t0=$(date +%s); docker start "$REF" >/dev/null
  if wait_bind "$REF"; then log "boot->ready REF $(( $(date +%s) - t0 ))s"; else log "REF FAILED TO BIND — lane dark, needs hands"; fi; }

log "===== CARD S0c START ====="
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: REF not bound"; exit 5; }
docker ps -a --format '{{.Names}}' | grep -qx "$CAND" || { log "ABORT: $CAND missing"; exit 5; }
docker inspect "$CAND" --format '{{json .Config.Env}}' | tr ',' '\n' | grep PYTORCH_CUDA_ALLOC_CONF > "$R/env-B.txt"
grep -q "PYTORCH_CUDA_ALLOC_CONF=$ENVV" "$R/env-B.txt" || { log "ABORT: $CAND env lacks $ENVV"; exit 5; }
docker inspect "$REF" --format '{{json .Config.Env}}' | grep -q PYTORCH_CUDA_ALLOC_CONF && { log "ABORT: REF unexpectedly carries PYTORCH_CUDA_ALLOC_CONF"; exit 5; }
for n in "$V23" "$REF_RET" "$CFAIL"; do docker ps -a --format '{{.Names}}' | grep -qx "$n" && { log "ABORT: target name $n already exists"; exit 5; }; done
wait_quiet A || { log "ABORT: host never quiet; lane untouched"; exit 6; }

# ---- ARM A: live v22 REF ----
strays; hostmem A live "$REF"
cells A

# ---- ARM B: S0 candidate restart ----
stop_all; drop_caches; strays
hostmem B pre ""
t0=$(date +%s); since=$(date -u +%FT%TZ); docker start "$CAND" >/dev/null
if ! wait_bind "$CAND"; then
  boot_facts B "$CAND" "$since"
  log "BOOTFAIL B tail: $(tr '\r' '\n' < "$R/boot-B.log" | grep -E 'ValueError:|Traceback|ERROR|RuntimeError:|OutOfMemory|PIN_HOT abort' | tail -3 | cut -c1-200 | tr '\n' '|')"
  restore_v22 "B failed to bind"
  python3 $W/cardS0c_verdict.py "$R" > "$R/VERDICT.md" 2> "$R/verdict.err"
  final_state; log "===== CARD S0c END (FAIL: boot) ====="; exit 0
fi
log "boot->ready B $(( $(date +%s) - t0 ))s"
boot_facts B "$CAND" "$since"
hostmem B bound "$CAND"
cells B
hostmem B steady "$CAND"

# ---- verdict ----
python3 $W/cardS0c_verdict.py "$R" > "$R/VERDICT.md" 2> "$R/verdict.err"
RES=$(grep -m1 '^RESULT' "$R/VERDICT.md" | awk '{print $2}')
log "VERDICT: $(grep -m1 '^RESULT' "$R/VERDICT.md")"

if [[ $RES == PASS ]]; then
  docker rename "$CAND" "$V23" && log "PROMOTE renamed $CAND -> $V23 (serving)"
  docker rename "$REF" "$REF_RET" && log "PROMOTE renamed $REF -> $REF_RET (stopped, kept)"
  { head -1 $W/launch_v22.sh
    echo "# launch_v23.sh — the exact v23 launch (promoted 2026-10-18, Card S0c): launch_v22.sh + ONE env -e PYTORCH_CUDA_ALLOC_CONF=$ENVV"
    echo "#   (exact-size pinned host allocs above 1 GiB: host pinned 328.0 -> 252.8 GiB; same image, same autotune hash, same kernels)."
    echo "# Rollback: docker stop $V23 && docker start $REF_RET"
    tail -n +4 $W/launch_v22.sh | sed -e "s#-e VLLM_LOGGING_LEVEL=INFO #-e PYTORCH_CUDA_ALLOC_CONF=$ENVV -e VLLM_LOGGING_LEVEL=INFO #" \
      -e "s#NAME=\${1:-$REF}#NAME=\${1:-$V23}#"
  } > $W/launch_v23.sh
  bash -n $W/launch_v23.sh && log "launch_v23.sh bash -n OK" || log "launch_v23.sh bash -n FAILED"
  diff "$W/results-cardS0/launch_S0.sh" $W/launch_v23.sh > "$R/launch_v23-vs-S0.diff"
  diff $W/launch_v22.sh $W/launch_v23.sh > "$R/launch_v23-vs-v22.diff"
  log "launch_v23 vs launch_S0 non-comment diff lines: $(grep -E '^[<>]' "$R/launch_v23-vs-S0.diff" | grep -vE '^[<>] #' | tr '\n' '|')"
  gauntlet V23
  docker logs "$V23" > "$R/postlog-V23.log" 2>&1; log "errcount V23 $(grep -cE 'ValueError:|Traceback|ERROR|RuntimeError:' "$R/postlog-V23.log")"
  hostmem V23 final "$V23"
  final_state; log "===== CARD S0c END (PASS, PROMOTED v23 = $V23) ====="
else
  restore_v22 "verdict ${RES:-UNKNOWN}"
  final_state; log "===== CARD S0c END (FAIL, v22 stays = $REF) ====="
fi

#!/usr/bin/env bash
# cardP2_runner.sh — Card P2: pin-hot hook (v19 hook dir, sha b5e4f4f8) rebased onto vLLM nightly 39983a5e, bracketed CTL1 -> CAND -> CTL2.
# Armed 2026-10-18 ~10:45 CDT by Milo (hermes milo) subagent, claude-opus-5-5, from cardP2_runner.sh.DRAFT (prep brief BRIEF-P2-prep.md)
#   retargeted to the v23 REF; conventions from cardS0c_runner.sh (hostmem, since-scoped boot_facts).
# Arms:  CTL1 = live v23 REF as-is (no reboot)
#        CAND = launch_p2_nightly39983a5e.sh (image sha256:a3bc7f09…, off60, fresh cache vllm-cache-p2-39983a5e, autotune store
#               /root/.cache/vllm/fi-autotune-v2). First boot on a NEW managed store: hard timeout 110 min.
#               ONE retry at OFFGB=54 (name …-hook-off54-EXP) ONLY if the boot died at the hook's PIN_HOT host guard (MemAvailable).
#        CTL2 = v23 REF restarted (docker start, hash hit) and LEFT SERVING.
# Per arm (identical order): boot facts, hostmem, TF capture + margin-weighted flips, BFCL dev, cold prefill N=3 @8K+64K,
#   RT C1x3/C4/C16/C32, run_window_T1T2.sh (knee r1/r2, knee6, agent fixture, replay r1/r2), gauntlet, serving-log error count.
# PRE-REGISTERED BARS (CAND vs MEAN of CTL1/CTL2; plan §P2; fail on ANY) — evaluated by cardP2_verdict.py:
#   TTFT 8K (RT C1 median TTFT, mean of 3) <= 0.45 s; cold 64K prefill tok/s >= +30%; RT C1 >= -2%; replay r2 >= -2%;
#   fidelity: flips_margin_p2.py GATE PASS (ref = CTL1 capture, calib = Card R2b file scored vs CTL1, bar 1.25x) AND gauntlet 12/12
#   AND fixture acceptance per class within 2 pts of CTL mean AND RT C1 acceptance_len within 0.10 AND BFCL dev >= CTL mean - 1 pt.
#   mechanism: "Decoder SWA bounded replay:" boot line AND "rehome done layers=40" — else INVALID.
# First boot on a new autotune store = PROVISIONAL. NO PROMOTION by this runner, ever.
# Stop-and-keep only. Never :30003, never glmf-*/dsf*. Refuses foreign GPU containers / other live runners. Does not touch
#   run_window_T1T2.sh or $HOME/pin-hot-experts/hook/.
set -uo pipefail
D=$HOME/dsv41/m3; W=$HOME/dsv41/1017; R=$W/results-cardP2; mkdir -p "$R"
B=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v23-pinexact-BOUND-REF
CAND0=dsv41-vllm-1017-P2-n39983a5e-hook-EXP
CAND54=dsv41-vllm-1017-P2-n39983a5e-hook-off54-EXP
CAND=$CAND0
LAUNCH=$W/launch_p2_nightly39983a5e.sh
IMAGE=vllm/vllm-openai@sha256:a3bc7f09c1e2e72e068021feb75118c92e98c538706caa0c5d1acc0bb9252623
PCACHE=$HOME/dsv41/vllm-cache-p2-39983a5e
S=$HOME/pin-hot-experts/scripts; CORPUS=$HOME/pin-hot-experts/tf_corpus.jsonl; BFCL=$HOME/pin-hot-experts/bfcl_data
TFRREF=$W/results-cardR/tf-R-ref.jsonl; TFCAL=$W/results-cardR/tf-R2b.jsonl; FLIPS=$W/flips_margin_p2.py
GDIR=$HOME/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
LEDGER=$HOME/dsv41/results/ledger.md
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
memavail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 5; }
gpu_foreign(){ local c out=""; for c in $(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-'); do
    docker inspect "$c" --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null | grep -qi gpu && out="$out $c"; done; echo "$out"; }
any_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '; }
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*runner\.sh' | grep -v "cardP2_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
archive_busy(){ ps -eo args | grep -E 'ark_bf1017\.sh|archive_1017\.sh|archive_sums|rsync .*miloark|sha256sum|hf download|docker pull' | grep -v grep | head -3 | cut -c1-120 | tr '\n' '|'; }
wait_quiet(){ local stage=$1 i a f
  for ((i=0;i<60;i++)); do
    a=$(archive_busy); f=$(any_foreign)
    [[ -z $a && -z ${f// /} ]] && { [[ $i -gt 0 ]] && log "QUIET for $stage after $((i*30))s"; return 0; }
    (( i % 10 == 0 )) && log "WAIT $stage: archive=[${a}] foreign=[${f}]"
    sleep 30
  done; log "WAIT $stage: still busy after 30 min"; return 1; }
strays(){ log "strays: $(ps -eo pid,rss,args --sort=-rss | sed -n 2,5p | awk '{printf "%s %.1fGiB %s %s|", $1, $2/1048576, $3, $4}') MemAvailable=$(memavail)GiB"; }
store_entries(){ find "$PCACHE/fi-autotune-v2" -path '*/entries/*.json' 2>/dev/null | wc -l; }
hostmem(){ # tag phase [container]
  local tag=$1 ph=$2 c=${3:-} f="$R/hostmem-$1-$2.txt" id cg pid
  { echo "date $(date '+%F %T %Z') tag=$tag phase=$ph container=${c:--}"
    grep -E '^(MemTotal|MemFree|MemAvailable|Cached|Shmem|Mlocked|Unevictable|AnonPages|Mapped):' /proc/meminfo | awk '{printf "meminfo %s %.2f GiB\n",$1,$2/1048576}'
    if [[ -n $c ]] && docker ps --format '{{.Names}}' | grep -qx "$c"; then
      id=$(docker inspect -f '{{.Id}}' "$c"); cg=/sys/fs/cgroup/system.slice/docker-$id.scope
      [[ -d $cg ]] || cg=$(find /sys/fs/cgroup -maxdepth 3 -name "*$id*" 2>/dev/null | head -1)
      if [[ -n $cg && -r $cg/memory.current ]]; then
        awk '{printf "cgroup memory.current %.2f GiB\n",$1/1073741824}' "$cg/memory.current"
        grep -E '^(anon|file|shmem) ' "$cg/memory.stat" | awk '{printf "cgroup %s %.2f GiB\n",$1,$2/1073741824}'
      fi
      pid=$(docker top "$c" -eo pid,rss,args 2>/dev/null | grep 'VLLM::EngineCore' | awk '{print $1}' | head -1)
      if [[ -n $pid ]]; then
        echo "enginecore_pid $pid"
        sudo -n awk '/^[0-9a-f]+-[0-9a-f]+ /{dz=($0 ~ /\/dev\/zero/); if(dz)n++} dz&&/^Size:/{s+=$2} dz&&/^Rss:/{r+=$2} END{printf "pinned_devzero maps=%d size_gib=%.2f rss_gib=%.2f\n",n,s/1048576,r/1048576}' /proc/$pid/smaps
      fi
    fi
  } > "$f" 2>&1
  log "HOSTMEM $tag $ph: $(grep -E 'MemAvailable|pinned_devzero|cgroup shmem|cgroup memory.current' "$f" | tr '\n' ';' | tr -s ' ' | cut -c1-300)"; }
# wait_bind: dump logs to a FILE then grep (never `docker logs | grep -q` under pipefail). Fast-fail only on hard errors.
wait_bind(){ # name maxmin [since] [nofastfail]
  local n=$1 maxmin=$2 since=${3:-} nff=${4:-} i t0 lf=$R/.wb-$1.log; t0=$(date +%s)
  for ((i=0;i<maxmin*6;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $(( $(date +%s)-t0 ))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n after $(( $(date +%s)-t0 ))s"; return 1; }
    if [[ -n $since ]]; then docker logs --since "$since" "$n" > "$lf" 2>&1; else docker logs "$n" > "$lf" 2>&1; fi
    if [[ -z $nff ]] && grep -qE 'ValueError:|Traceback|ERROR|RuntimeError:|PIN_HOT hook FAILED' "$lf"; then
      log "FASTFAIL $n after $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < "$lf" | grep -E 'ValueError:|Traceback|ERROR|RuntimeError:|PIN_HOT' | tail -3 | cut -c1-240 | tr '\n' '|')"; return 1; fi
    (( i % 30 == 0 )) && log "  $n waiting $((i/6)) min: store_entries=$(store_entries) MemAvail=$(memavail)GiB last=[$(tr '\r' '\n' < "$lf" | grep -E 'rehome|Autotun|Preloaded|Capturing|KV cache size|Decoder SWA|Loading safetensors|Loading weights' | tail -1 | cut -c1-160)]"
    sleep 10
  done; log "TIMEOUT $n ($maxmin min)"; return 1; }
boot_facts(){ # tag name [since]
  local tag=$1 name=$2 since=${3:-}
  if [[ -n $since ]]; then docker logs --since "$since" "$name" > "$R/boot-$tag.log" 2>&1; else docker logs "$name" > "$R/boot-$tag.log" 2>&1; fi
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -E "Decoder SWA bounded replay|SWA bounded replay|Using V[12] Model Runner|Auto-enabling VLLM_USE_BREAKABLE|Defaulting to tokenizer_mode|KV cache format|data type to store kv cache|MXFP4 MoE backend|Using .*TRTLLM|Using FlashInfer (managed )?autotune cache|Managed autotune cache|Preloaded [0-9]+/[0-9]+|Autotuning process (starts|ends)|Loaded [0-9]+ configs|Saved [0-9]+ configs|GPU KV cache size|Available KV cache memory|Graph capturing finished|PIN_HOT installed|PIN_HOT wrapped|PIN_HOT abort|PIN_HOT hook FAILED|rehome layer=|rehome done|Model loading took|model loading took|Engram table offloaded|attention backend|Initializing a V1 LLM engine" | grep -vE 'it/s\]' | cut -c1-400 > "$R/facts-$tag.txt"
  local ts te; ts=$(grep -m1 'Autotuning process starts' "$R/facts-$tag.txt" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1); te=$(grep 'Autotuning process ends' "$R/facts-$tag.txt" | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "TUNEWINDOW ${ts:-?} ${te:-?}" >> "$R/facts-$tag.txt"; echo "STORE_ENTRIES $(store_entries)" >> "$R/facts-$tag.txt"
  log "FACTS $tag: replay_line=$(grep -c 'Decoder SWA bounded replay:' "$R/facts-$tag.txt") replay_off=$(grep -c 'bounded replay is off\|needs model runner V2' "$R/facts-$tag.txt") rehome=$(grep -c 'rehome done layers=40' "$R/facts-$tag.txt") | autotune: Loaded=$(grep -oE 'Loaded [0-9]+ configs' "$R/facts-$tag.txt" | tail -1) saved=$(grep -c 'Saved [0-9]* configs' "$R/facts-$tag.txt") starts=$(grep -c 'Autotuning process starts' "$R/facts-$tag.txt") window=${ts:-?}->${te:-?} managed=$(grep -oE 'Preloaded [0-9]+/[0-9]+' "$R/facts-$tag.txt" | tail -1) store_entries=$(store_entries) | $(grep 'GPU KV cache size' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //' | cut -c1-60) | L9 $(grep 'rehome layer=9 ' "$R/facts-$tag.txt" | tail -1 | grep -oE 'host_avail=[0-9.]+GiB') | $(grep 'rehome done' "$R/facts-$tag.txt" | tail -1 | sed 's/.*rehome done //' | cut -c1-140)"; }
cold(){ local tag=$1
  BASE_URL=$U/v1 MODEL=$M API_KEY=none SIZES="8192 65536" N=3 THINKING=0 timeout 1200 python3 $HOME/dsv41/cold_prefill_probe.py > "$R/cold-$tag.txt" 2>&1 || log "cold $tag nonzero rc"
  log "COLD $tag: $(grep PREFILL "$R/cold-$tag.txt" | cut -c1-200 | tr '\n' ';')"; }
rt_bench(){ local tag=$1 N O r
  N="p2-$1-$(date +%s)"; O="$R/rt-$1"; mkdir -p "$O"
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
flips(){ # label ref cand outfile
  if [[ -s $2 && -s $3 ]]; then python3 "$FLIPS" "$2" "$3" --label "$1" --calib "$TFCAL" > "$4" 2>&1; log "FLIPGATE $1: $(tail -1 "$4")"
  else log "FLIPGATE $1: UNKNOWN (missing $2 or $3 — never treat as a number)"; fi; }
tf(){ local tag=$1
  ( cd "$R" && MAXPOS=4096 CONC=4 python3 $S/tf_logprob.py capture "$tag" "$CORPUS" ) > "$R/tf-$tag.log" 2>&1; log "TF $tag: $(tail -1 "$R/tf-$tag.log" | cut -c1-160)"
  case $tag in
    CTL1) flips "CTL1-vs-Rref" "$TFRREF" "$R/tf-CTL1.jsonl" "$R/flips-CTL1-vsRref.txt" ;;
    CAND) flips "CAND-vs-CTL1" "$R/tf-CTL1.jsonl" "$R/tf-CAND.jsonl" "$R/flips-CAND-vsCTL1.txt"
          flips "CAND-vs-Rref" "$TFRREF" "$R/tf-CAND.jsonl" "$R/flips-CAND-vsRref.txt" ;;
    CTL2) flips "CTL2-vs-CTL1" "$R/tf-CTL1.jsonl" "$R/tf-CTL2.jsonl" "$R/flips-CTL2-vsCTL1.txt"
          [[ -s $R/tf-CAND.jsonl ]] && flips "CAND-vs-CTL2" "$R/tf-CTL2.jsonl" "$R/tf-CAND.jsonl" "$R/flips-CAND-vsCTL2.txt" ;;
  esac; }
bfcl(){ local tag=$1; ( cd "$R" && CONC=8 MAXTOK=2048 THINKING=0 python3 $S/bfcl_gate.py "$tag" "$BFCL" "$R" ) > "$R/bfcl-$tag.log" 2>&1; log "BFCL $tag: $(grep -h 'BFCL' "$R/bfcl-$tag.log" | tail -1 | cut -c1-200)"; }
measure(){ # tag name
  local tag=$1 name=$2
  tf "$tag"; bfcl "$tag"
  cold "$tag"
  rt_bench "$tag"
  bash $HOME/dsv41/run_window_T1T2.sh "P2-$tag" > "$R/nohup-$tag.log" 2>&1 || log "window $tag nonzero rc"
  mv $HOME/dsv41/overnight-T1T2/*"P2-$tag"[-.]* "$R"/ 2>/dev/null
  log "WINDOW $tag: $(grep -hE 'conc= ?(1|8|16):' "$R/window-P2-$tag.log" 2>/dev/null | sed 's/  */ /g' | tr '\n' ';' | cut -c1-500)"
  log "REPLAY $tag: $(python3 -c "
import json
for r in ('r1','r2'):
    try: d=json.load(open('$R/replayc-P2-$tag-'+r+'-n4.json')); print(r, d.get('agg_tok_s'), 'acc', d.get('accept'), end=' ; ')
    except Exception as e: print(r,'ERR',e, end=' ; ')
")"
  log "FIXTURE $tag: $(python3 -c "
import json
try: d=json.load(open('$R/agentfix-P2-$tag.json')); print({k:v.get('accept') for k,v in d['res'].items() if isinstance(v,dict)})
except Exception as e: print('ERR',e)
")"
  rm -rf "$R/gauntlet-$tag"
  (cd "$GDIR" && python3 run_suite.py --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "p2-$tag" --cache-salt "p2-$tag" --sandbox-image "$GIMG" --workers 4) > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"
  hostmem "$tag" steady "$name"
  docker logs "$name" > "$R/postlog-$tag.log" 2>&1; grep -cE 'ValueError:|Traceback|ERROR|RuntimeError:' "$R/postlog-$tag.log" > "$R/errcount-$tag.txt"
  log "ERRCOUNT $tag $(cat "$R/errcount-$tag.txt") (whole container log)"; }
launch_cand(){ # offgb name -> 0 bound, 1 fail, 7 host-guard death
  local off=$1 name=$2 t0 ma
  wait_quiet "CAND-off$off" || return 1
  stop_all; drop_caches; strays; hostmem "CAND-off$off" pre ""
  ma=$(memavail); if awk -v m="$ma" 'BEGIN{exit !(m < 380)}'; then log "ABORT CAND off$off: MemAvailable ${ma} GiB < 380 after stop+drop_caches (stray?)"; return 1; fi
  docker ps -a --format '{{.Names}}' | grep -qx "$name" && { docker rename "$name" "$name-STALE-$(date +%H%M)"; log "renamed stale $name"; }
  log "store entries before CAND boot: $(store_entries)"
  t0=$(date +%s); OFFGB=$off bash "$LAUNCH" "$name" || { log "launch failed (off$off)"; return 1; }
  log "launched $name off$off image ${IMAGE##*@}"
  if wait_bind "$name" 110; then log "boot->ready CAND $(( $(date +%s)-t0 ))s (off$off, $name)"; return 0; fi
  docker logs "$name" > "$R/boot-CAND-off$off-FAIL.log" 2>&1
  local hg; hg=$(tr '\r' '\n' < "$R/boot-CAND-off$off-FAIL.log" | grep -c 'PIN_HOT abort.*MemAvailable')
  log "BOOTFAIL CAND off$off host_guard_hits=$hg tail: $(tr '\r' '\n' < "$R/boot-CAND-off$off-FAIL.log" | grep -E 'PIN_HOT abort|Error:|ERROR|EngineCore failed|OutOfMemory' | tail -3 | cut -c1-220 | tr '\n' '|')"
  docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-off$off-BOOT-FAIL" && log "kept $name-off$off-BOOT-FAIL (stopped)"
  (( hg > 0 )) && return 7; return 1; }

log "===== CARD P2 START (image ${IMAGE##*@}, REF=$REF, off60, CONFIRM=0) ====="
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running for CTL1"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: REF not bound on :30006"; exit 5; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: image not pulled"; exit 5; }
[[ -f $LAUNCH ]] && bash -n "$LAUNCH" || { log "ABORT: $LAUNCH missing or syntax"; exit 5; }
[[ -f $FLIPS && -s $TFRREF && -s $TFCAL && -f $W/cardP2_verdict.py ]] || { log "ABORT: fidelity/verdict inputs missing"; exit 5; }
for n in "$CAND0" "$CAND54"; do docker ps -a --format '{{.Names}}' | grep -qx "$n" && { log "ABORT: $n already exists"; exit 5; }; done
log "preflight OK: MemAvailable=$(memavail)GiB store_entries=$(store_entries)"
wait_quiet CTL1 || { log "ABORT: host never quiet; lane untouched (REF serving)"; exit 6; }

# ---- CTL1: live v23 lane ----
boot_facts CTL1 "$REF" "$(docker inspect -f '{{.State.StartedAt}}' "$REF")"; strays; hostmem CTL1 live "$REF"
measure CTL1 "$REF"

# ---- CAND: 39983a5e + hook, off60; one host-guard retry at off54 ----
launch_cand 60 "$CAND0"; rc=$?
if (( rc == 7 )); then log "host guard death at off60 -> retry off54 (labelled fallback; offload GiB is not in kernel shapes)"; CAND=$CAND54; launch_cand 54 "$CAND54"; rc=$?; fi
if (( rc == 0 )); then
  boot_facts CAND "$CAND"; hostmem CAND bound "$CAND"
  grep -q 'Decoder SWA bounded replay:' "$R/facts-CAND.txt" || log "WARN CAND: replay boot line ABSENT -> verdict INVALID (mechanism missing); measuring anyway"
  grep -q 'rehome done layers=40' "$R/facts-CAND.txt" || log "WARN CAND: hook did NOT re-home -> verdict INVALID"
  measure CAND "$CAND"
else log "CAND not measured (rc=$rc)"; fi

# ---- CTL2: v23 REF back, left serving ----
wait_quiet CTL2 || log "WARN: host not quiet before CTL2 — booting REF anyway (lane must return)"
stop_all; drop_caches; strays
t0=$(date +%s); since=$(date -u +%FT%TZ); docker start "$REF" >/dev/null
if wait_bind "$REF" 40 "$since" nofastfail; then log "boot->ready CTL2 $(( $(date +%s)-t0 ))s"; boot_facts CTL2 "$REF" "$since"; hostmem CTL2 bound "$REF"; measure CTL2 "$REF"
else log "CTL2 REF FAILED TO BIND — lane dark, needs hands"; fi

# ---- verdict + suffix (no promotion) ----
python3 $W/cardP2_verdict.py "$R" > "$R/VERDICT.md" 2> "$R/verdict.err"
V=$(grep -m1 '^SUFFIX' "$R/VERDICT.md" | awk '{print $2}')
log "VERDICT: $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md")"
if docker ps -a --format '{{.Names}}' | grep -qx "$CAND"; then
  docker ps --format '{{.Names}}' | grep -qx "$CAND" && docker stop "$CAND" >/dev/null 2>&1
  docker rename "$CAND" "$CAND-${V:-DONE}" && log "kept $CAND-${V:-DONE} (stopped)"
fi
grep -hE ' (RT|WINDOW|REPLAY|FIXTURE|GAUNTLET|FACTS|COLD|TF|FLIPGATE|BFCL|HOSTMEM|ERRCOUNT|boot->ready|BOUND|BOOTFAIL|FASTFAIL|DIED|TIMEOUT|WARN|ABORT|VERDICT) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card P2: hook rebase onto nightly-39983a5e vs v23 (${IMAGE##*@}; CTL1 live v23 → CAND off60 → CTL2 v23 restart) — $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md" 2>/dev/null)"
  grep -vE '^SUFFIX' "$R/VERDICT.md"; echo; echo "Runner summary lines:"; sed 's/^/    /' "$R/SUMMARY.txt"; echo "Receipts: $R (Mac copy: el8-cards-2026-10-18/P2/receipts/). Not promoted."; } >> "$LEDGER"
log "FINAL dsv41 running=[$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')] bound=$(curl -s -m 3 $U/v1/models | grep -c $M)"
log "===== CARD P2 END (lane = $REF) ====="

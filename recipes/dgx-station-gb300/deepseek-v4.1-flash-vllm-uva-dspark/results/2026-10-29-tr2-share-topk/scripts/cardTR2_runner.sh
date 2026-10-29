#!/usr/bin/env bash
# cardTR2_runner.sh — Card TR2: L2 attention/indexer top-k table sharing on the GB300-only lane (:30006),
# bracket A -> B -> A2 on the v28-prob config. Pre-registration: /home/milo/dsv41/m3/tr2/PREREG-TR2.md.
#   A / A2 : L2_SHARE=0  (overlay installed in report-only mode; every kernel runs — behaviour = control)
#   B      : L2_SHARE=2  (relaxed sharing: skip the op for layers that are not their group's index source)
# One axis. Image, hook, rowmap, k-schedule, probabilistic drafting, autotune cache: unchanged.
# Assumes the lane is already dark + isolated (cardTR2_c0.sh does that). Restores the h2-REF pair at END.
# Detached: cd ~/dsv41/m3/tr2 && setsid nohup bash cardTR2_runner.sh >/dev/null 2>&1 </dev/null & echo $! > runner.pid
# Abort: touch ~/dsv41/m3/tr2/results-cardTR2/ABORT (checked between boots/cells).
set -uo pipefail
R=$HOME/dsv41/m3/tr2/results-cardTR2
T=$HOME/dsv41/m3/tr2
U=http://127.0.0.1:30006; M=dsv41-flash-uva
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55
HOOK=$HOME/pin-hot-experts/v26ar2/hook
ROWMAP=rowmap-mixed-ar-v1.json
LAUNCH=$T/launch_tr2.sh
BENCH=$HOME/dsv41/bench; PERCLASS=$BENCH/perclass_run.sh
B1=$HOME/dsv41/m3/br1; PERCLASS_T=$B1/perclass_run_t.sh; PCDEC_T=$B1/perclass_decode_t.py
MIXED_T=$B1/ar_mixed_t.py
TF=$HOME/pin-hot-experts/scripts/tf_logprob.py; CORPUS=$HOME/pin-hot-experts/tf_corpus.jsonl
FLIPS=$HOME/dsv41/1017/flips_margin_p2.py
GDIR=$HOME/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
OWNER=$HOME/dsv41/m3/LANE-OWNER.txt
ARMS="A B A2"
TEMP_T=1.0; SEED_T=42; TOP_P_T=1.0
T_PERCLASS=2400; T_TF=900; T_GAUNTLET=1500; T_MIXED=900
SETTLE=90; ADAPT_SETTLE=90; MEM_MIN=100
BOOT_MIN=20
ABORT=$R/ABORT

RESUME=${RESUME:-0}
if [[ $RESUME == 1 ]]; then
  [[ -d $R ]] || { echo "REFUSE: RESUME=1 but $R missing"; exit 5; }
else
  [[ -e $R ]] && { echo "REFUSE: $R exists"; exit 5; }
  mkdir -p "$R" || exit 5
fi
cp -f "$T/PREREG-TR2.md" "$R/PREREG-TR2.md" 2>/dev/null
exec >> "$R/runner.log" 2>&1
exec 3>> "$R/runner.log"
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >&3; }
own(){ echo "- $(date '+%F %T %Z') $*" >> "$OWNER"; }
memavail(){ awk '/MemAvailable/{printf "%.0f",$2/1048576}' /proc/meminfo; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
gb_used(){ nvidia-smi --id=$GB300 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
is_up(){ [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; done; sleep 5; }
check_abort(){ [[ -e $ABORT ]] && { log "ABORT seen"; exit 7; }; return 0; }
arm_l2(){ case $1 in A|A2) echo 0;; B) echo 2;; esac; }
arm_name(){ echo "dsv41-vllm-1017-TR2-$1"; }
kv_tokens(){ docker logs "$1" 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | grep -oE "[0-9,]+" | tr -d ','; }
hash_hit(){ docker logs "$1" 2>&1 | grep -qE "Config cache hit|Loaded [0-9]+ configs" && echo 1 || echo 0; }
l2_installed(){ docker logs "$1" 2>&1 | grep -c "^L2 installed"; }
l2_stats(){ docker logs "$1" 2>&1 | tr '\r' '\n' | grep "L2 stats" | tail -1; }
l2_all(){ docker logs "$1" 2>&1 | tr '\r' '\n' | grep "L2 " ; }
errors(){ docker logs "$1" 2>&1 | tail -4000 | grep -cE ' ERROR [0-9]{2}-[0-9]{2}|Traceback \(most recent call last\)|EngineCore failed|OutOfMemoryError'; }
metrics(){ curl -s -m 10 $U/metrics 2>/dev/null | grep -E '^vllm:spec_decode_' > "$1"; }
wait_bind(){ local n=$1 i; for i in $(seq 1 $((BOOT_MIN*12))); do
    is_up "$n" || { log "container $n not running"; return 1; }
    curl -sf -m 3 "$U/v1/models" >/dev/null 2>&1 && { log "BOUND $n after $((i*5)) s"; return 0; }
    sleep 5; done; log "BIND TIMEOUT $n"; return 1; }
cell(){ local tag=$1 cmd=$2; check_abort; is_up "$NAME" || { log "ARM DOWN before $tag"; return 1; }
  local t0=$(date +%s); bash -c "$cmd" > "$A/$tag.log" 2>&1; local rc=$?
  log "CELL $BOOT $tag rc=$rc $(( $(date +%s)-t0 ))s: $(tail -1 "$A/$tag.log" | cut -c1-240)"; return 0; }
go_unfreeze(){ touch "$RUN/GO"; printf '%s' '{"freeze": false, "baseline": true}' > "$RUN/ctrl.json"; log "GO + ctrl unfreeze"; }

log "===== CARD TR2 START $(date '+%F %T %Z') arms=[$ARMS] ====="
# ---------------- preflight (lane must already be dark + isolated) ----------------
up=$(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-' | tr '\n' ' ')
if [[ $RESUME == 1 ]]; then
  up=$(echo " $up " | sed "s/ $(arm_name A) / /" | tr -s ' ')
  log "RESUME: ignoring the arm-A container; others: [${up:-none}]"
fi
[[ -z ${up// /} ]] || { log "REFUSE: containers running: $up"; exit 4; }
ps -eo pid=,comm=,args= | awk '$2=="python3" && index($0,"sidecar_watchdog.py"){print "REFUSE: watchdog still up", $0}' | tee /dev/stderr | grep -q . && exit 4
sudo -n iptables -C INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null \
  || { log "REFUSE: :30006 not isolated"; exit 4; }
log "preflight OK: lane dark, isolated, no watchdog"

for ARM in $ARMS; do
  check_abort
  BOOT=$ARM
  NAME=$(arm_name "$ARM"); A=$R/$ARM; RUN=$T/$ARM/ar
  mkdir -p "$A" "$RUN"
  log "---- ARM $ARM ($NAME) L2_SHARE=$(arm_l2 "$ARM")"
  if [[ $RESUME == 1 && $ARM == A ]] && is_up "$NAME"; then
    log "RESUME: arm A already serving ($NAME); reusing this boot"
    RUN=$T/A/ar
    kv=$(kv_tokens "$NAME"); hh=$(hash_hit "$NAME"); li=$(l2_installed "$NAME"); er=$(errors "$NAME")
    log "BOOT A (reused) kv=$kv hash_hit=$hh l2_installed=$li errors=$er"
    l2_all "$NAME" | grep "L2 stats" | tail -1 > "$A/l2-boot-stats.txt"
    metrics "$A/metrics-before.txt"
    [[ -d $A/perclass-t1 && -n "$(ls -A $A/perclass-t1 2>/dev/null)" ]] && log "pct1 already present -> skip" \
      || cell pct1 "cd $BENCH && ARM=TR2-$ARM-t1 OUTDIR=$A/perclass-t1 TEMP=$TEMP_T SEED=$SEED_T TOP_P=$TOP_P_T DEC=$PCDEC_T timeout $T_PERCLASS bash $PERCLASS_T"
    cell pct0 "cd $BENCH && ARM=TR2-$ARM-t0 OUTDIR=$A/perclass-t0 timeout $T_PERCLASS bash $PERCLASS"
    cell tf "cd $A && MAXPOS=4096 CONC=4 timeout $T_TF python3 $TF capture TR2-$ARM $CORPUS"
    cell gauntlet "cd $GDIR && timeout $T_GAUNTLET python3 run_suite.py --base-url $U/v1 --model $M --output $R/gauntlet-$ARM --tag tr2-$ARM --cache-salt tr2-$ARM --sandbox-image $GIMG --workers 4"
    metrics "$A/metrics-after.txt"
    l2_all "$NAME" | grep "L2 stats" | tail -1 > "$A/l2-boot-stats.txt"
    docker stop -t 60 "$NAME" >/dev/null; log "ARM A stopped (kept)"
    continue
  fi
  stop_all
  wait=0; while [[ $(memavail) -lt $MEM_MIN && $wait -lt 20 ]]; do log "wait MemAvailable=$(memavail)GiB"; sleep 15; wait=$((wait+1)); done
  drop_caches; sleep $SETTLE
  st=$(date +%s); log "settle done: MemAvailable=$(memavail)GiB GB300=$(gb_used)MiB"

  NAME=$NAME RUN=$RUN ROWMAP=$ROWMAP SPEC_EXTRA=',"draft_sample_method":"probabilistic"' \
    L2_SHARE=$(arm_l2 "$ARM") L2_VERIFY=0 L2_STATS_EVERY=1 L2_VERIFY_FILE=/ar/L2-VERIFY L2_AUDIT_FILE=/ar/L2-AUDIT \
    bash "$LAUNCH" || { log "LAUNCH FAILED $ARM"; exit 5; }
  wait_bind "$NAME" || { docker logs --tail 40 "$NAME" >&3; exit 5; }
  since=$(date -u +%FT%TZ)
  kv=$(kv_tokens "$NAME"); hh=$(hash_hit "$NAME"); li=$(l2_installed "$NAME"); er=$(errors "$NAME")
  spec=$(docker inspect -f '{{range .Config.Cmd}}{{.}} {{end}}' "$NAME")
  printf '{"arm":"%s","name":"%s","kv_tokens":"%s","hash_hit":%s,"l2_installed":%s,"errors":%s,"since":"%s"}\n' \
    "$ARM" "$NAME" "$kv" "$hh" "$li" "$er" "$since" > "$A/facts.json"
  log "BOOT $ARM kv=$kv hash_hit=$hh l2_installed=$li errors=$er"
  l2_all "$NAME" | grep "L2 stats" | tail -1 > "$A/l2-boot-stats.txt"
  go_unfreeze; sleep $ADAPT_SETTLE
  log "post-settle: MemAvailable=$(memavail)GiB stats=$(l2_stats "$NAME")"

  # ---- A only: audit window — the logical key matched; is the recomputed table identical? ----
  if [[ $ARM == A ]]; then
    echo 1 > "$RUN/L2-AUDIT"; log "audit window ON"
    for i in 1 2 3 4 5 6; do
      curl -s -m 180 "$U/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"Describe the CAP theorem in four sentences."}],"max_tokens":192,"temperature":1.0,"chat_template_kwargs":{"thinking":false}}' \
        -o /dev/null & done; wait
    sleep 2; echo 0 > "$RUN/L2-AUDIT"
    l2_all "$NAME" | grep "audit layer" > "$A/audit-lines.txt"
    l2_all "$NAME" | grep "L2 stats" | tail -1 > "$A/audit-stats.txt"
    log "audit OFF: $(tail -1 "$A/audit-stats.txt" | cut -c1-260)"
    log "audit lines: $(wc -l < "$A/audit-lines.txt")  equal=$(grep -c 'equal=True' "$A/audit-lines.txt")  diff=$(grep -c 'equal=False' "$A/audit-lines.txt")"
  fi

  # ---- B only: in-process identity window (verify on) ----
  if [[ $ARM == B ]]; then
    echo 1 > "$RUN/L2-VERIFY"
    log "verify window ON"
    for i in 1 2 3 4 5 6; do
      curl -s -m 180 "$U/v1/completions" -H 'Content-Type: application/json' \
        -d '{"model":"dsv41-flash-uva","prompt":"Write a careful paragraph about distributed systems.","max_tokens":192,"temperature":1.0}' \
        -o /dev/null -w "" &
    done; wait
    sleep 2; echo 0 > "$RUN/L2-VERIFY"
    vlog=$(docker logs "$NAME" 2>&1 | grep "^L2 stats" | tail -1)
    log "verify window OFF: $vlog"
    echo "$vlog" > "$A/l2-verify-stats.txt"
  fi

  # ---- cells ----
  metrics "$A/metrics-before.txt"
  cell pct1 "cd $BENCH && ARM=TR2-$ARM-t1 OUTDIR=$A/perclass-t1 TEMP=$TEMP_T SEED=$SEED_T TOP_P=$TOP_P_T DEC=$PCDEC_T timeout $T_PERCLASS bash $PERCLASS_T"
  cell pct0 "cd $BENCH && ARM=TR2-$ARM-t0 OUTDIR=$A/perclass-t0 timeout $T_PERCLASS bash $PERCLASS"
  cell tf "cd $A && MAXPOS=4096 CONC=4 timeout $T_TF python3 $TF capture TR2-$ARM $CORPUS"
  cell gauntlet "cd $GDIR && timeout $T_GAUNTLET python3 run_suite.py --base-url $U/v1 --model $M --output $R/gauntlet-$ARM --tag tr2-$ARM --cache-salt tr2-$ARM --sandbox-image $GIMG --workers 4"
  metrics "$A/metrics-after.txt"
  log "ARM $ARM cells done in $(( $(date +%s)-st ))s; stats=$(l2_stats "$NAME")"
  docker stop -t 60 "$NAME" >/dev/null; log "ARM $ARM stopped (kept)"
done

# ---------------- END: teacher-forced compares (CPU) ----------------
cd "$R/A" || exit 5
for pair in "A A2" "A B" "A2 B"; do set -- $pair; ref=$1; cand=$2
  [[ -s $R/$ref/tf-TR2-$ref.jsonl && -s $R/$cand/tf-TR2-$cand.jsonl ]] || { log "TF missing for $ref/$cand"; continue; }
  cp "$R/$cand/tf-TR2-$cand.jsonl" "$R/A/tf-$cand.jsonl" 2>/dev/null
  cp "$R/$ref/tf-TR2-$ref.jsonl" "$R/A/tf-$ref.jsonl" 2>/dev/null
  timeout 600 python3 "$TF" compare "tf-$ref.jsonl" "tf-$cand.jsonl" > "$R/tf-$ref-vs-$cand.log" 2>&1
  log "TF compare $ref vs $cand: $(tail -1 "$R/tf-$ref-vs-$cand.log" | cut -c1-200)"
done

# ---------------- restore the production h2-REF pair + watchdog ----------------
log "restore: starting h2-REF pair"
docker start dsv41-vllm-1017-h2-REF-rtx >/dev/null 2>&1
for i in $(seq 1 120); do docker logs --tail 5 dsv41-vllm-1017-h2-REF-rtx 2>&1 | grep -q "serving\|Application startup complete" && break; sleep 5; done
docker start dsv41-vllm-1017-h2-REF >/dev/null 2>&1
wait_bind dsv41-vllm-1017-h2-REF && log "restore: bound"
sudo -n iptables -D INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null && log "restore: ISOLATE removed"
WDARGV=$HOME/dsv41/m3/ap2/promote/wd-argv-dsv41-vllm-1017-h2-REF.txt
if [[ -s $WDARGV ]]; then
  # argv file is newline-separated, one argument per line -- read into an array, never `bash -c "$(cat ...)"`
  mapfile -t WDARGS < "$WDARGV"
  ( cd $HOME/dsv41/m3 && setsid nohup "${WDARGS[@]}" >/dev/null 2>&1 </dev/null & )
  sleep 6
  ps -eo pid,ppid,comm,args | awk '$3=="python3" && index($0,"sidecar_watchdog.py")' | head -2 | tee /dev/stderr
  grep -q '"trip": null' $HOME/dsv41/m3/logs/sidecar_watchdog.status.json || log "WARN: watchdog status not trip null"
fi
own "cardTR2 done; lane restored to dsv41-vllm-1017-h2-REF pair + watchdog; ISOLATE removed (receipts $R)"
log "===== CARD TR2 END $(date '+%F %T %Z') ====="
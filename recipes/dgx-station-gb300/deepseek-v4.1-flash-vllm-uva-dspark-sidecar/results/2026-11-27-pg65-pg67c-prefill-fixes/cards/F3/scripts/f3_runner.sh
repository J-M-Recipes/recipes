#!/usr/bin/env bash
# f3_runner.sh — Card F3 (2026-11-27): who wins ONE-USER decode today, our Pg67c lane (GB300 + RTX sidecar) or Flare's
#   current one-GB300 core (HF local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE @ b55be385, core d683cc1a, runtime
#   NVIDIA/hyperflare-inference @ 7db06e0c built with the repo Dockerfile; their README command, MTP 7, Engram in RAM)?
#   James 11/27: "test run" ... end state "whichever one wins c1".
#   PREREG: ~/hermes/research/dsv41-flare-xfer-20261124/CARD-F3-SPEC.md (Mac).  Verdict rule lives in f3_verdict.py.
#   Order: wait for f3_prep -> verify core signature/id -> window: P1 (live Pg67c, isolated) -> Pg67c down -> F26 (Flare)
#          -> verdict -> end state:  flare wins  => Flare stays up (loopback :30016 + kwproxy :30006), Pg67c stopped
#                                    pg67c wins / split / invalid / any failure => Pg67c restored, then P3 bracket cells
#   Derived from f2c_runner.sh (cells, Flare boot) and cardPF_runner.sh (down_any/up_work/restore, the 9>&- rule).
set -uo pipefail
GR=$HOME/dsv41/garner; GH=$GR/home; PROD=$HOME/dsv41/sidecar-resid/work-Pg67c; LAUNCH=$GR/launcher/launch-pg61.sh
F=$HOME/dsv41/flarecmp; P=$F/prep-F3; R=$F/results-F3; ABORT=$R/ABORT
MODEL=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
FL_DIR=/models/flare-cache/DeepSeek-V4.1-Flash-1017-FLARE-gb300-1126; FL_IMG=flare-cli:7db06e0c; FA=F26
FL_CORE=d683cc1a8f598e006750e65dfa3260c2109452da67bc73db1282950fbbac6020
FL_FP=937c0143f6856d3ee8ad164f62e3d804bd825d7a8750d0accb9a5849f703a8c1
GB_UUID=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55
OWN1=$HOME/dsv41/m3/LANE-OWNER.txt; OWN2=$HOME/dsv41/m3/logs/LANE-OWNER.txt; MAINT=$HOME/dsv41/m3/logs/MAINTENANCE
U=http://127.0.0.1:30006; M=deepseek-v4.1-flash
BENCH=$HOME/dsv41/bench; PCDEC_T=$F/perclass_decode_f2.py
GDIR=$HOME/pin-hot-experts/round7/gauntlet; GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
VER=$GR/8043a681/verify; LIB=$HOME/dsv41/m3/p3/lib; LIBPY=$HOME/dsv41/m3/p3/libenv/bin/python3
FL_MAP='{"prose":256,"structured":256,"code":256,"shell_ops":256,"tool_json":256}'
CG_C="1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128 144 160 192 224 256 384"
KV_PG=11813533; KV_FL=5264640
RESTORED=0; CUR=pg; T_DARK0=0; WINNER=none; ENDED=0

[[ -e $R ]] && { echo "REFUSE: $R exists"; exit 5; }
mkdir -p "$R"/{P1,F26,P3} || exit 5
exec 3>> "$R/runner.log"; exec >> "$R/runner.log" 2>&1
log(){ printf '%s [F3] %s\n' "$(date '+%F %T %Z')" "$*" >&3; }
own(){ local l="- $(date '+%F %T %Z') card F3 (Flare gb300 core d683cc1a vs Pg67c, C1): $*"; echo "$l" >> "$OWN1"; echo "$l" >> "$OWN2"; }
check_abort(){ [[ -e $ABORT ]] && return 1; return 0; }
dark_min(){ [[ $T_DARK0 -eq 0 ]] && echo 0 || echo $(( ( $(date +%s) - T_DARK0 ) / 60 )); }
isolate_on(){ sudo -n iptables -C INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null \
  || sudo -n iptables -I INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset; }
isolate_off(){ local n=0; while sudo -n iptables -D INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null; do n=$((n+1)); done
  if sudo -n iptables -C INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null; then log "ISOLATE STILL ON: a REJECT rule for :30006 survived"; return 1; fi
  log "ISOLATE off (removed $n rule(s))"; return 0; }
bound(){ curl -sf -m 5 $U/v1/models 2>/dev/null | grep -q "$M"; }
cur_work(){ local w; w=$(ps -eo args= | awk '{for(i=1;i<NF;i++) if ($i ~ /sidecar_watchdog_p3\.py$/ && $(i+1)=="--server" && $(i+2)=="ds41") {print $i; exit}}'); [ -n "$w" ] && dirname "$w" || echo none; }
cg_argv(){ printf '%s\n' "$1" | awk '{for(i=1;i<NF;i++) if ($i=="--cudagraph-capture-sizes") {s=""; for(j=i+1;j<=NF;j++){ if ($j ~ /^--/) break; s = s (s?" ":"") $j }; print s; exit}}'; }
argv_of(){ docker inspect -f '{{json .Config.Cmd}}' "$1" 2>/dev/null | tr -d '[]"' | tr ',' ' ' | tr -s ' '; }
settle(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45; log "settle: $(free -g | sed -n 2p | tr -s ' ')"; }
idle_wait(){ local i=0 idle=0
  while (( i < 120 )); do local r; r=$(curl -s -m 5 "$U/metrics" | awk '/^vllm:num_requests_(running|waiting)/{s+=$2} END{print s+0}')
    if [[ "$r" == 0 ]]; then idle=$((idle+1)); else idle=0; fi
    (( idle >= 6 )) && { log "idle check ok (lane quiet)"; return 0; }
    (( i == 0 )) && log "idle wait: holding the window until the lane is quiet"; sleep 5; i=$((i+1)); done
  log "REFUSE: lane stayed busy 10 min"; return 1; }

cell(){ # cell <dir> <tag> <cmd>
  local A=$1 tag=$2 cmd=$3; [[ " ${CELLSKIP:-} " == *" $tag "* ]] && return 0; check_abort || { log "ABORT before $tag"; return 1; }
  bound || { log "SERVER DOWN before $(basename $A) $tag"; return 1; }
  local t0; t0=$(date +%s); bash -c "$cmd" > "$A/$tag.log" 2>&1; local rc=$?
  log "CELL $(basename $A) $tag rc=$rc $(( $(date +%s)-t0 ))s: $(tail -1 "$A/$tag.log" | tr -d '\r' | cut -c1-180)"; }
libcmd(){ # libcmd <dir> <port> <kv> <conc> <ctx> <outfile>
  echo "cd $LIB && timeout 1800 $LIBPY -u llm_decode_bench.py --host 127.0.0.1 --port $2 --model $M --display-mode plain --no-resume --concurrency $4 --contexts $5 --duration 30 --max-tokens 8192 --decode-warmup-seconds 3 --kv-budget $3 --output $1/$6 < /dev/null"; }
cells(){ # cells <arm> <kv_budget_tokens> <lib port: engine direct>
  local arm=$1 kv=$2 lp=$3; local A=$R/$1; echo none > "$A/key"
  curl -s -m 10 $U/metrics > "$A/metrics-before.txt"; nvidia-smi --query-gpu=index,memory.used --format=csv > "$A/smi-before.txt"; free -g > "$A/free-before.txt"
  cell "$A" probe "python3 $F/flare_api_probe.py $U $M > $A/probe.json"
  grep -q '"status": *4[0-9][0-9]' "$A/probe.json" 2>/dev/null && log "NOTE $arm probe shows 4xx fields (see probe.json)"
  # one real harness request must return 200 before any cell is trusted
  local hc; hc=$(curl -s -o "$A/gate.json" -w '%{http_code}' -m 120 $U/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: lane ok\"}],\"max_tokens\":32,\"temperature\":1.0,\"top_p\":1.0,\"chat_template_kwargs\":{\"thinking\":false}}")
  local PCOK=1
  if [[ $hc == 200 ]]; then log "GATE ok $arm (HTTP 200): $(head -c 160 "$A/gate.json" | tr -d '\n')"
  else PCOK=0; log "GATE FAIL $arm: thinking-off harness request returned HTTP $hc: $(head -c 300 "$A/gate.json" | tr -d '\n') -- per-class cells skipped, lib cells (no thinking option) still run"; fi
  [[ $PCOK -eq 1 ]] && cell "$A" warm "cd $BENCH && python3 $PCDEC_T --base $U --model $M --outdir $A/warm --temperature 1.0 --top-p 1.0 --seed 7 --tag F3-$arm-warm --mode c1 --natural --fl 384"
  cell "$A" lib "$(libcmd $A $lp $kv 1,8,32 0,16k lib.json)"
  cell "$A" libc1-b "$(libcmd $A $lp $kv 1 0 libc1-b.json)"
  cell "$A" libc1-c "$(libcmd $A $lp $kv 1 0 libc1-c.json)"
  local s; [[ $PCOK -eq 1 ]] && for s in 42 43 44; do
    cell "$A" "pc-s$s" "cd $BENCH && python3 $PCDEC_T --base $U --model $M --outdir $A/pc --temperature 1.0 --top-p 1.0 --seed $s --tag F3-$arm-s$s --mode c1 --fl-map '$FL_MAP'"
  done
  [[ $PCOK -eq 1 ]] && cell "$A" pc-nat "cd $BENCH && python3 $PCDEC_T --base $U --model $M --outdir $A/pc --temperature 1.0 --top-p 1.0 --seed 42 --tag F3-$arm-nat --mode c1 --natural --fl 384"
  if [[ ${FULL:-0} -eq 1 ]]; then
    cell "$A" gbench "timeout 3600 docker run --rm --network host -v $VER:/v:ro -v $A:/out -v $A/key:/key:ro --entrypoint python3 vllm/vllm-openai:v0.31.0 /v/bench_full.py --base-url $U --key-file /key --model $M --out /out/bench.json --skip coding,capacity --reps 3 --window 60 --warmup 20 --conc 8,32"
    cell "$A" gauntlet "cd $GDIR && timeout 1500 python3 run_suite.py --base-url $U/v1 --model $M --output $A/gauntlet --tag f3-$arm --cache-salt f3-$arm-$(date +%s) --sandbox-image $GIMG --workers 4"
  fi
  curl -s -m 10 $U/metrics > "$A/metrics-after.txt"; nvidia-smi --query-gpu=index,memory.used --format=csv > "$A/smi-after.txt"
  bound && log "SURVIVED $arm: server up after all cells" || { log "DIED $arm"; return 1; }; }

down_any(){ local w; w=$(cur_work); [ "$w" = none ] && w=$PROD; log "Pg down via WORK=$w"; WORK=$w DS41_HOME=$GH bash $LAUNCH down 9>&- >> "$R/launch.log" 2>&1
  local i; for i in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^ds41\(-sidecar\)\?$' || { log "down -> containers absent after ${i} polls"; return 0; }; sleep 2; done
  log "down: ds41/ds41-sidecar STILL present after 120s -- forcing removal"; docker rm -f ds41 ds41-sidecar >> "$R/launch.log" 2>&1; }
up_pg(){ settle
  if env MODEL_DIR=$MODEL WORK=$PROD DS41_HOME=$GH bash $LAUNCH up 9>&- >> "$R/launch.log" 2>&1 && bound; then log "UP Pg67c: $(grep -a 'verify OK' $R/launch.log | tail -1 | cut -c1-200)"; CUR=pg; return 0; fi
  log "UP Pg67c FAILED: $(tail -3 $R/launch.log | tr '\n' ' ' | cut -c1-300)"; cp "$(ls -t $GH/state/logs/*-ds41.log 2>/dev/null | head -1)" "$R/pg67c-failed-ds41.log" 2>/dev/null; return 1; }
start_proxy(){ stop_proxy; setsid nohup python3 $F/kwproxy.py 30006 30016 rewrite > $R/kwproxy-$FA.log 2>&1 < /dev/null & echo $! > $R/kwproxy.pid; sleep 2; }
stop_proxy(){ [ -f $R/kwproxy.pid ] && kill "$(cat $R/kwproxy.pid)" 2>/dev/null; rm -f $R/kwproxy.pid; pkill -f "kwproxy.py 30006 30016" 2>/dev/null; true; }
gpu_free_wait(){ local i u; for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GB_UUID" 2>/dev/null | tr -dc '0-9'); [[ -n $u && $u -lt 4000 ]] && return 0; sleep 5; done; log "WARN GB300 still holds ${u:-?} MiB"; return 1; }
fl_up(){ settle; gpu_free_wait
  docker rm -f flare-gb300 >/dev/null 2>&1
  docker run -d --name flare-gb300 --restart no --device nvidia.com/gpu=$GB_UUID --ipc host --network host \
    --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 --security-opt label=disable \
    -v $FL_DIR:/core:ro -w /core $FL_IMG \
    run . --gpus 0 --host 127.0.0.1 --port 30016 --served-model-name $M --mtp-draft-tokens 7 \
    --embedding-table-storage ram --publisher-keys publisher.json --dangerously-accept-publisher local-inference-lab \
    --dangerously-approve-key $FL_FP >> "$R/launch.log" 2>&1 || { log "flare docker run FAILED"; return 1; }
  CUR=flare; local t0; t0=$(date +%s)
  while (( $(date +%s) - t0 < 2400 )); do
    [[ "$(docker inspect -f '{{.State.Running}}' flare-gb300 2>/dev/null)" == true ]] || { docker logs flare-gb300 > $R/$FA/boot-failed.log 2>&1
      log "flare container EXITED after $(( $(date +%s)-t0 ))s: $(tail -4 $R/$FA/boot-failed.log | tr '\n' ' ' | cut -c1-500)"; docker rm -f flare-gb300 >/dev/null 2>&1; CUR=none; return 1; }
    curl -sf -m 5 http://127.0.0.1:30016/v1/models >/dev/null 2>&1 && { start_proxy; bound; } && { log "UP flare after $(( $(date +%s)-t0 ))s"; docker logs flare-gb300 > $R/$FA/boot.log 2>&1
      curl -s -m 5 http://127.0.0.1:30016/v1/models > $R/$FA/models.json; return 0; }
    check_abort || { log "ABORT during flare boot"; return 1; }
    sleep 10; done
  log "flare not bound after 40 min"; docker logs flare-gb300 > $R/$FA/boot-timeout.log 2>&1; return 1; }
fl_down(){ stop_proxy; docker ps -a --format '{{.Names}}' | grep -qx flare-gb300 || { CUR=none; return 0; }
  docker logs flare-gb300 > $R/$FA/serve.log 2>&1; docker stop -t 180 flare-gb300 >/dev/null 2>&1; docker rm -f flare-gb300 >/dev/null 2>&1; log "flare down"; CUR=none; }
restore(){
  [[ $RESTORED -ne 0 ]] && return 0
  log "RESTORE Pg67c starting"; own "RESTORE Pg67c starting"
  if [[ "$(cur_work)" == "$PROD" ]] && [[ -z "$(docker ps --format '{{.Names}}' | grep -x flare-gb300)" ]] && bound; then log "Pg67c already serving"
  else fl_down; down_any; up_pg || { own "RESTORE FAILED; lane DARK; MAINTENANCE kept"; log "RESTORE FAILED"; return 1; }; fi
  local e c; e=$(docker inspect -f '{{json .Config.Env}}' ds41); c=$(argv_of ds41)
  [[ $e == *'"VLLM_PREFILL_YIELD=short"'* && $e == *w8oproj* && $e == *lowb_aux* && $e == *'"VLLM_PREFILL_CADENCE=3"'* ]] || { log "RESTORE env check FAILED (production env incomplete)"; return 1; }
  [[ "$(cg_argv "$c")" == "$CG_C" ]] || { log "RESTORE FAILED: ladder is '$(cg_argv "$c")', not Pg67c's"; return 1; }
  [[ "$(cur_work)" == "$PROD" ]] || { log "RESTORE FAILED: serving work is $(cur_work)"; return 1; }
  curl -s -m 60 $U/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"Reply with exactly: lane ok"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"thinking":false}}' | grep -q "lane ok" && log "smoke OK" || log "WARN smoke failed"
  grep -q '"trip": null' $PROD/wdp3.status.json 2>/dev/null && log "p3 watchdog ENFORCE trip null" || log "WARN watchdog status"
  docker logs ds41 2>&1 | grep -a 'MEGA_PEER PF1' > "$R/pf1-lines-restore.txt"; log "PF1 lines after restore: $(grep -ac ' ACTIVE ' "$R/pf1-lines-restore.txt") ACTIVE, $(grep -ac DISABLED "$R/pf1-lines-restore.txt") DISABLED"
  RESTORED=1; return 0; }
finish(){ local rc=$?
  [[ $ENDED -eq 1 ]] && exit $rc
  log "finish: rc=$rc restored=$RESTORED cur=$CUR winner=$WINNER dark=$(dark_min) min"
  if [[ $T_DARK0 -eq 0 && $ISOLATED -eq 0 ]]; then log "===== CARD F3 END (pre-flight exit; lane untouched) ====="; ENDED=1; exit $rc; fi
  if [[ $WINNER == flare && $CUR == flare ]] && bound; then
    isolate_off
    echo "card F3 END STATE (Milo, 2026-11-27): Flare gb300 core d683cc1a won one-user decode and was left serving per James ('whichever one wins c1'): flare on 127.0.0.1:30016, kwproxy on 127.0.0.1:30006 (loopback only). Pg67c is STOPPED. Bring Pg67c back: docker stop flare-gb300; pkill -f 'kwproxy.py 30006'; rm this file; env WORK=$PROD MODEL_DIR=$MODEL DS41_HOME=$GH bash $LAUNCH up" > $MAINT
    own "END: Flare (core d683cc1a, runtime 7db06e0c) LEFT SERVING on loopback :30016 + kwproxy :30006 by James's rule (it won C1); Pg67c stopped; MAINTENANCE kept with the bring-back command; dark $(dark_min) min"
    log "===== CARD F3 END (lane = FLARE on loopback; Pg67c stopped; winner flare) ====="; ENDED=1; exit 0
  fi
  restore || log "RESTORE FAILED -- lane state unknown, attendance required"
  if [[ $RESTORED -eq 1 ]]; then
    if check_abort && [[ $T_DARK0 -ne 0 ]]; then sleep 30; log "---- P3 (Pg67c restored; bracket)"; CELLSKIP="warm pc-nat" FULL=0 cells P3 $KV_PG 30006
      python3 $F/f3_verdict.py $R > "$R/verdict-final.out" 2>&1; log "final verdict (with P3 recorded): $(tail -1 $R/verdict-final.out)"; fi
    isolate_off
    [ -e $MAINT ] && mv $MAINT $MAINT.stale-F3-$(date +%Y%m%d-%H%M)
    own "END: Pg67c ($PROD) serving :30006; watchdog ENFORCE; MAINTENANCE archived; winner=$WINNER; dark $(dark_min) min"
    log "===== CARD F3 END (lane = Pg67c; winner $WINNER; dark $(dark_min) min) ====="
  else isolate_off; log "===== CARD F3 END (RESTORE FAILED; lane DARK -- ATTENDANCE REQUIRED) ====="; fi
  ENDED=1; exit $rc; }

# ---- pre-flight (nothing stopped) ---------------------------------------------------------------------
ISOLATED=0
log "===== CARD F3 START ====="
trap finish EXIT
trap 'exit 130' HUP INT TERM
log "waiting for f3_prep (download, disk cleanup) and the runtime image"
for i in $(seq 1 1080); do
  grep -q "F3 PREP END" "$P/prep.log" 2>/dev/null && [[ -f $P/img.status ]] && break
  check_abort || { log "ABORT while waiting for prep"; exit 9; }
  sleep 10; done
grep -q "F3 PREP END" "$P/prep.log" || { log "REFUSE: prep did not finish in 3 h"; exit 4; }
log "prep: $(grep 'F3 PREP END' $P/prep.log | tail -1 | cut -c1-200) | dl=$(cat $P/dl.status) img=$(cat $P/img.status)"
[[ "$(cat $P/dl.status)" == OK ]] || { log "REFUSE: core download not OK"; exit 4; }
[[ "$(cat $P/img.status)" == OK ]] || { log "REFUSE: runtime image not OK"; exit 4; }
for i in $(seq 1 60); do ps -eo args= | grep -E 'sha256sum -c|hf download|docker build' | grep -vq grep || break; sleep 10; done
docker run --rm -v $FL_DIR:/core:ro -w /core $FL_IMG inspect . --json --publisher-keys publisher.json > "$R/inspect.json" 2> "$R/inspect.err"
python3 - "$R/inspect.json" "$FL_CORE" > "$R/inspect.check" 2>&1 <<'PY' || { log "REFUSE: core inspect failed: $(cat $R/inspect.check | cut -c1-300) $(tail -2 $R/inspect.err | tr '\n' ' ' | cut -c1-300)"; exit 4; }
import json, sys
j = json.load(open(sys.argv[1])); c = j.get("core", j)
print("signatures_verified", c.get("signatures_verified"), "core_id", c.get("core_id"))
sys.exit(0 if c.get("signatures_verified") is True and c.get("core_id") == sys.argv[2] else 1)
PY
log "core verified: $(cat $R/inspect.check)"
[[ $(docker ps --format '{{.Names}}' | sort | tr '\n' ' ') == "ds41 ds41-sidecar " ]] || { log "REFUSE: unexpected containers $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; exit 4; }
[[ "$(cur_work)" == "$PROD" ]] || { log "REFUSE: serving work is $(cur_work), not $PROD"; exit 4; }
[[ "$(cg_argv "$(argv_of ds41)")" == "$CG_C" ]] || { log "REFUSE: live ladder is not Pg67c's"; exit 4; }
[[ -e $MAINT ]] && { log "REFUSE: MAINTENANCE present: $(cat $MAINT)"; exit 4; }
OTHER_RUNNER=$(ps -eo args= | awk '$1 ~ /(bash|sh|env)$/ && $2 ~ /runner\.sh$/ && $2 !~ /f3_runner/ {print}' | head -1)
[[ -n "$OTHER_RUNNER" ]] && { log "REFUSE: another runner is up: $OTHER_RUNNER"; exit 4; }
fs=$(df -Pk / | awk 'NR==2{print int($4/1048576)}'); log "root free ${fs} GiB; /models free $(df -BG --output=avail /models | tail -1 | tr -dc '0-9') GiB"; [[ $fs -ge 60 ]] || { log "REFUSE: root disk"; exit 4; }
for f in $F/f3_verdict.py $F/kwproxy.py $F/flare_api_probe.py $PCDEC_T $LIB/llm_decode_bench.py $LIBPY $LAUNCH; do [[ -e $f ]] || { log "REFUSE: missing $f"; exit 4; }; done
sha256sum $F/f3_verdict.py $F/f3_runner.sh $LIB/llm_decode_bench.py $PCDEC_T >&3
check_abort || { log "ABORT before the window"; exit 9; }

# ---- the window ---------------------------------------------------------------------------------------
idle_wait || exit 8
isolate_on || { log "REFUSE: could not install the isolation rule (nothing stopped)"; exit 5; }
ISOLATED=1; log "ISOLATE on (:30006 loopback only)"
echo "card F3 window (Milo; James 11/27 'test run'): Pg67c C1 cells, then Flare gb300 core d683cc1a, end state = C1 winner (f3_runner.sh)" > $MAINT
own "START: :30006 isolated for card F3 (P1 on live Pg67c, then DARK for Flare core d683cc1a @ runtime 7db06e0c); end state = C1 winner, else Pg67c restored; runner $R/runner.log"
log "---- P1 (live Pg67c, isolated)"
FULL=0 cells P1 $KV_PG 30006 || { log "P1 cells failed; ending with Pg67c in place"; exit 6; }
check_abort || { log "ABORT after P1"; exit 9; }
T_DARK0=$(date +%s); own "Pg67c down for the Flare arm (lane DARK)"
down_any
log "---- F26 (Flare core d683cc1a @ HF b55be385, runtime 7db06e0c, README command: MTP 7, Engram in RAM; lib direct :30016, per-class via kwproxy)"
if fl_up; then own "arm F26 BOUND (isolated; flare :30016, kwproxy :30006)"
  FULL=1 cells F26 $KV_FL 30016 || log "F26 cells did not complete cleanly"
else log "F26 boot FAILED"; fi
python3 $F/f3_verdict.py $R > "$R/verdict.out" 2>&1; WINNER=$(tail -1 "$R/verdict.out" | sed -n 's/^WINNER=//p'); [[ -n $WINNER ]] || WINNER=invalid
log "VERDICT: winner=$WINNER | $(sed -n 3p $R/F3-VERDICT.md 2>/dev/null | cut -c1-200)"
exit 0

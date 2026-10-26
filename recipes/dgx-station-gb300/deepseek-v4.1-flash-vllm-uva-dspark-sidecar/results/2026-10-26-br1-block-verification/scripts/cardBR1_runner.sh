#!/usr/bin/env bash
# cardBR1_runner.sh — Card BR1: DSpark BLOCK verification ("rejection_sample_method":"block", Sun et al.) vs the default
# sequential rejection sampler, on the production v25m-kvtier sidecar recipe, bracketed A -> B -> A2, lane :30006, GB300.
# ONE AXIS = SPEC_EXTRA=',"rejection_sample_method":"block"' spliced into --speculative-config by launch_br1.sh (empty on
# A/A2 => args byte-identical to production). Not a compute-hash factor => hash-hit boot (~519 s) expected on every arm.
# Block verification applies only to NON-GREEDY rows (af7f9488 rejection_sampler_utils.py: USE_BLOCK_VERIFICATION and not
# is_greedy) => the primary speed cells sample at temperature 1.0 (perclass_run_t.sh, ar_mixed_t.py); the stock temp-0
# window is a no-op/parity sanity cell (G2b). TF, gauntlet, replay_turns are temp-0/teacher-forced (unchanged instruments).
# Detached: cd ~/dsv41/m3/br1 && setsid nohup bash cardBR1_runner.sh > /dev/null 2>&1 < /dev/null & echo $! > cardBR1.pid
# Abort: touch ~/dsv41/m3/br1/results-cardBR1/ABORT (checked between cells/boots) or kill -TERM <pid>.
# Structure cloned from ~/dsv41/m3/kv3/cardKV3_runner.sh (DIFF-vs-cardKV3.txt). Production restore, watchdog argv capture +
# re-arm, ISOLATE, foreign-POST counts, SETTLE gate and the append-only LANE-OWNER own() line from ~/dsv41/m2/cardM2_runner.sh
# (DIFF-restore-vs-cardM2.txt). DECISIONS + open questions: BRIEF-BR1.md.
# ARM PLAN:
#   preflight (refusals leave the lane untouched) -> capture production watchdog argv+cwd -> stop it (log moved aside)
#     -> stop the v25m-kvtier pair (kept) -> ISOLATE (tagged REJECT on :30006 for non-loopback)
#   per arm X in A B A2: wait_quiet + stop_all + drop_caches + SETTLE + MEM_MIN -> launch_br1.sh sidecar -> server
#       (EXTRA = production KV flags; SPEC_EXTRA empty on A/A2, block on B) -> arm BR1 watchdog (enforce) -> wait_bind
#       -> boot facts (KV tokens, args/env vs production, hash-hit shape, block_in_args)
#     cells (each via bg() with a timeout; /metrics vllm:spec_decode_* at every cell start/end):
#       (1) pct1  : perclass_run_t.sh TEMP=1.0 SEED=42 -> X/perclass-t1   (PRIMARY, G2)
#       (2) pct0  : stock perclass_run.sh (temp 0, full r1+r2: it cannot skip r2) -> X/perclass-t0   (no-op sanity, G2b)
#       (3) c1-r1..3 : ar_mixed_t.py HELDOUT=1 C1 120 s at temperature 1.0 (G3)
#       (4) tf    : tf_logprob.py capture (teacher-forced; G4 after restore)
#       (5) gauntlet (KV3 gauntlet(), tags br1-X; G5)   (6) turns: replay_turns.py PASSES=2 (G6)
#       (7) errcount server/-rtx, foreign POSTs, BR1 watchdog collect (G7, G0b)
#     -> kill BR1 watchdog -> stop pair (keep)
#   END: restore production pair + fail-stop watchdog (M2 restore_lane; double-forked; PPID 1 checked; ISOLATE removed
#     first) -> TF compares + margin flips (CPU) -> verdict -> renames -> SUMMARY -> ledger -> LANE-OWNER append -> FINAL
set -uo pipefail
HM=${BR1_HOME:-~}   # desk dry-run override only; production = ~
DS=$HM/dsv41; D=$DS/m3; B1=$D/br1
R=$B1/results-cardBR1
U=http://127.0.0.1:30006; M=dsv41-flash-uva
IMAGE=vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063   # = launch_sidecar.sh default (af7f9488)
MODEL=${BR1_MODEL_DIR:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55; RTX=GPU-d07eb6e6-7053-42a7-6811-7bb833266182
M3HOOK=$D/el8/deepseek-v4.1-flash/m3/hook
ROWMAP=rowmap-mixed-ar-v1.json
LAUNCH=$B1/launch_br1.sh; LAUNCH_REF=$D/launch_sidecar.sh
VERDICT=$B1/br1_verdict.py
MIXED=$DS/ar/ar_mixed_v2.py; MIXED_T=$B1/ar_mixed_t.py
BENCH=$DS/bench; PERCLASS=$BENCH/perclass_run.sh; PCDEC=$BENCH/perclass_decode.py; PCTAB=$BENCH/perclass_table.py
PCPROMPTS=$BENCH/perclass_prompts.jsonl; X3=$DS/1017/agent_fixture_x3.py
PERCLASS_T=$B1/perclass_run_t.sh; PCDEC_T=$B1/perclass_decode_t.py
TF=$HM/pin-hot-experts/scripts/tf_logprob.py; CORPUS=$HM/pin-hot-experts/tf_corpus.jsonl   # CORPUS = cardR_1017_runner.sh line 15
FLIPS=$DS/1017/flips_margin_p2.py   # CLI from its header: REF.jsonl CAND.jsonl [--label X] [--calib R2B.jsonl]; last line GATE ...
TURNS=$D/replay_turns.py; TURNS_OUT=$D/results-cardSC1b; FIXTURE=$DS/transcript_fixture.json   # replay_turns.py hardcodes its out dir
WD=$D/sidecar_watchdog.py
V25=dsv41-vllm-1017-v25m-kvtier; V25R=$V25-rtx
WDLOG=$D/logs/sidecar_watchdog.kvtier.log   # production fail-stop watchdog log (M2)
CWDLOG=$D/logs/sidecar_watchdog.br1.log; CWDSTAT=$D/logs/sidecar_watchdog.br1.status.json; CWDPID=$D/logs/sidecar_watchdog.br1.pid   # card watchdog
N_A=dsv41-vllm-1017-BR1-A; N_B=dsv41-vllm-1017-BR1-B; N_A2=dsv41-vllm-1017-BR1-A2
ALLN="$N_A $N_A-rtx $N_B $N_B-rtx $N_A2 $N_A2-rtx"
KVX="--kv-cache-memory 9126805504 --kv-offloading-size 48 --kv-offloading-backend native"   # production EXTRA, every arm
SPEC_B=',"rejection_sample_method":"block"'
KV_EXPECT=4,400,013
GDIR=$HM/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
LEDGER=$DS/results/ledger.md
OWNER=$D/LANE-OWNER.txt
FF=' ERROR [0-9]{2}-[0-9]{2} |Traceback \(most recent call last\)|ValueError:|RuntimeError:|OutOfMemoryError|EngineCore failed'
# sha1 prefixes (8) — override via env only after re-reviewing the file (set-but-empty disables a check: desk dry-run only)
# >>>>> TODO-SHA (Milo, after review + rename of the five BR1 drafts): fill the 8-char sha1 prefixes below. TODO-SHA never
# >>>>> matches, so the runner REFUSES at preflight until these are filled.
SHA_LAUNCH=${SHA_LAUNCH-24cdb66e}; SHA_VERDICT=${SHA_VERDICT-46a2ad52}
SHA_PERCLASS_T=${SHA_PERCLASS_T-3dfeb6e8}; SHA_PCDEC_T=${SHA_PCDEC_T-f2b95a01}; SHA_MIXED_T=${SHA_MIXED_T-b12c5a66}
# measured on the box 2026-10-25 (desk, read-only sha1sum)
SHA_PERCLASS=${SHA_PERCLASS-682114ad}; SHA_PCDEC=${SHA_PCDEC-09c66432}; SHA_MIXED=${SHA_MIXED-ce5231c1}; SHA_WD=${SHA_WD-b17d9103}
SHA_TF=${SHA_TF-3638227d}; SHA_TURNS=${SHA_TURNS-5f7934a9}; SHA_ROWMAP=${SHA_ROWMAP-52e2861c}; SHA_SITE=${SHA_SITE-efe0bb6a}
SHA_PEER=${SHA_PEER-15d11022}
# extra pins (not in the card's list; same convention): diff reference, perclass table/x3/prompts, margin flips, corpora
SHA_LREF=${SHA_LREF-1796f1d9}; SHA_PCTAB=${SHA_PCTAB-6e6d3ff8}; SHA_X3=${SHA_X3-9d10d2e3}; SHA_PCPROMPTS=${SHA_PCPROMPTS-618898aa}
SHA_FLIPS=${SHA_FLIPS-affda07d}; SHA_CORPUS=${SHA_CORPUS-3672c1a7}; SHA_FIXTURE=${SHA_FIXTURE-e3f1aea4}
MIX_S=${MIX_S:-120}; C1_REPS=${C1_REPS:-3}; TEMP_T=${TEMP_T:-1.0}; SEED_T=${SEED_T:-42}; TOP_P_T=${TOP_P_T:-1.0}
T_PERCLASS=${T_PERCLASS:-2700}; T_MIXED=${T_MIXED:-600}; T_TF=${T_TF:-1800}; T_TURNS=${T_TURNS:-1500}
MEM_MIN=${MEM_MIN:-380}; MEM_GUARD=${MEM_GUARD:-30}; BOOT_MIN=${BOOT_MIN:-40}; WAIT_PID=${WAIT_PID:-}
SETTLE_S=${SETTLE_S:-90}; SETTLE_MAX_S=${SETTLE_MAX_S:-600}; ISOLATE=${ISOLATE:-1}; FOREIGN_QUIET_S=${FOREIGN_QUIET_S:-60}
ARMS="A B A2"   # locked order (not env-overridable)

if [[ -e $R ]]; then echo "$(date '+%F %T %Z') REFUSE: $R exists (refuse to mix runs; move it aside)" | tee -a "$B1/cardBR1.refused.log" >&2; exit 5; fi
mkdir -p "$R"
exec >> "$R/runner.log" 2>&1
exec 3>> "$R/runner.log"   # log() always writes here: a trap firing inside `bg ... > cell.log` inherits that redirect on fd 1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >&3; }
own(){ echo "- $(date '+%F %T %Z') $*" >> "$OWNER"; }   # LANE-OWNER.txt is APPEND-ONLY for this card (M2)
MEMINFO=${MEMINFO:-/proc/meminfo}
memavail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' "$MEMINFO"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; done; sleep 5; }
not_known(){ docker ps --format '{{.Names}}' | grep -vxE "$V25|$V25R" | tr '\n' ' '; }
any_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '; }
wd_pids(){ ps -eo pid=,comm=,args= | awk -v s="$V25" '$2=="python3" && index($0,"sidecar_watchdog.py") && index($0,"--server " s " ") {print $1}' | tr '\n' ' '; }   # python3 process only (its bash -c parent also carries the same argv text)
wd_orphan_parent(){ # pid of an orphaned bash (PPID 1, one child) holding the production watchdog: the M2 form
  # `( cd && setsid nohup ... & )` backgrounds the whole AND-list, so a bash copy carrying the PREVIOUS runner's argv
  # stays the watchdog's parent (seen 2026-10-25 after cardM3: ppid 1870623 'bash .../cardM3_runner.sh'). It is not a runner.
  local p pp; p=$(wd_pids); p=${p%% *}; [[ -n $p ]] || return 0
  pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [[ -n $pp && $pp != 1 ]] || return 0
  [[ "$(ps -o comm= -p "$pp" 2>/dev/null)" == bash && "$(ps -o ppid= -p "$pp" 2>/dev/null | tr -d ' ')" == 1 \
     && "$(ps --ppid "$pp" -o pid= 2>/dev/null | grep -c .)" == 1 ]] && echo "$pp"; return 0; }
other_runners(){ local skip; skip=$(wd_orphan_parent)
  ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh|perclass_run_t\.sh)' | grep -v "cardBR1_runner.sh" | grep -v grep \
    | awk -v s="${skip:-x}" '$1!=s' | cut -c1-120 | tr '\n' '|'; }
archive_busy(){ ps -eo args | grep -E 'ark_bf1017\.sh|archive_1017\.sh|archive_sums|rsync .*miloark|sha256sum|hf download|docker pull|docker save' | grep -v grep | head -3 | cut -c1-120 | tr '\n' '|'; }
wait_quiet(){ local stage=$1 i a f
  for ((i=0;i<60;i++)); do
    a=$(archive_busy); f=$(any_foreign)
    [[ -z $a && -z ${f// /} ]] && { [[ $i -gt 0 ]] && log "QUIET for $stage after $((i*30))s"; return 0; }
    (( i % 10 == 0 )) && log "WAIT $stage: archive=[${a}] foreign=[${f}]"
    sleep 30
  done; log "WAIT $stage: still busy after 30 min"; return 1; }
strays(){ log "strays: $(ps -eo pid,rss,args --sort=-rss | sed -n 2,5p | awk '{printf "%s %.1fGiB %s %s|", $1, $2/1048576, $3, $4}') MemAvailable=$(memavail)GiB"; }
is_up(){ [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
exists(){ docker ps -a --format '{{.Names}}' | grep -qx "$1"; }
rtx_used(){ nvidia-smi --id=$RTX --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
oom_count(){ { sudo -n dmesg 2>/dev/null || journalctl -k --no-pager -q 2>/dev/null; } | grep -ciE 'oom-kill|out of memory: kill' ; }
# ---- M2: host/GPU/shm captures + settle gate before every boot (nothing is deleted: a leftover mmap belongs to its owner)
shm_big(){ find /dev/shm -maxdepth 2 -type f -size +1G -printf '%s %p\n' 2>/dev/null | sort -k2; }
host_capture(){ local f=$1
  { echo "== $(date '+%F %T %Z') $2"; free -g; df -h /dev/shm; ls -la /dev/shm | grep -v ' sem\.mp-' | head -20; echo "-- /dev/shm objects > 1 GiB"; shm_big
    nvidia-smi --query-gpu=index,name,clocks.sm,clocks.mem,power.draw,pstate,memory.used --format=csv
    grep -E 'MemAvailable|Shmem:' /proc/meminfo; } >> "$f" 2>&1; }
settle(){ local stage=$1 f=$R/settle-$1.txt t0 ma new g
  host_capture "$f" "$stage (right after stop, before settle)"
  log "SETTLE $stage: sleeping ${SETTLE_S}s before the next boot"; sleep "$SETTLE_S"; t0=$(date +%s)
  while :; do
    host_capture "$f" "$stage"; ma=$(memavail)
    new=$(shm_big | awk 'FILENAME==ARGV[1]{seen[$2]=1; next} !($2 in seen){printf "%s(%.1fGiB) ", $2, $1/1073741824}' "$R/preflight-shm-big.txt" -)
    g=$(nvidia-smi --id=$GB300 --query-gpu=clocks.sm,clocks.mem,power.draw,pstate --format=csv,noheader 2>/dev/null)
    log "HOST $stage: MemAvailable=${ma}GiB shm_used=$(df -h --output=used /dev/shm | tail -1 | tr -d ' ') new_shm_gt1G=[${new}] GB300 sm,mem,W,pstate=[${g}] RTX used=$(rtx_used)MiB"
    if awk -v m="$ma" -v t="$MEM_MIN" 'BEGIN{exit !(m < t)}' || [[ -n $new ]]; then
      (( $(date +%s) - t0 >= SETTLE_MAX_S )) && { log "REFUSE boot $stage: MemAvailable ${ma} GiB (min $MEM_MIN) / new shm [${new}] after ${SETTLE_MAX_S}s — not deleting anything"; return 1; }
      log "HOLD $stage: MemAvailable ${ma} GiB < $MEM_MIN or new /dev/shm object(s) [${new}] — waiting 30 s"; sleep 30; continue
    fi
    return 0; done; }
# ---- M2: foreign clients. ISOLATE=1: REJECT non-loopback TCP to :30006 while dark; G0b counts non-127.0.0.1 POST /v1 per arm.
FW_TAG=cardBR1-isolate
isolate(){ [[ $ISOLATE == 1 ]] || { log "ISOLATE=0: foreign clients NOT blocked (detect-only)"; return 0; }
  local t ok=1; for t in iptables ip6tables; do
    sudo -n "$t" -C INPUT -p tcp --dport 30006 ! -i lo -m comment --comment "$FW_TAG" -j REJECT --reject-with tcp-reset 2>/dev/null \
      || sudo -n "$t" -I INPUT 1 -p tcp --dport 30006 ! -i lo -m comment --comment "$FW_TAG" -j REJECT --reject-with tcp-reset 2>/dev/null \
      || { [[ $t == iptables ]] && ok=0; } ; done
  [[ $ok == 1 ]] && log "ISOLATE on: non-loopback TCP :30006 rejected ($FW_TAG; removed before restore)" \
    || log "WARN ISOLATE failed (sudo -n iptables) — detect-only; G0b will invalidate any arm that serves a foreign client"; }
unisolate(){ local t i; for t in iptables ip6tables; do for i in 1 2 3; do
  sudo -n "$t" -D INPUT -p tcp --dport 30006 ! -i lo -m comment --comment "$FW_TAG" -j REJECT --reject-with tcp-reset 2>/dev/null || break; done; done
  sudo -n iptables -S INPUT 2>/dev/null | grep -q "$FW_TAG" && log "WARN: $FW_TAG rule still present (needs hands: sudo iptables -S INPUT)" || log "ISOLATE off ($FW_TAG removed)"; }
foreign(){ # name since -> "count sources" of non-127.0.0.1 POST /v1 access lines
  docker logs --since "$2" "$1" 2>&1 | tr '\r' '\n' | grep -oE 'INFO: +[0-9a-fA-F.:]+:[0-9]+ - "POST /v1/' | awk '{split($2,a,":"); if (a[1]!="127.0.0.1") {n++; s[a[1]]++}} END{printf "%d", n+0; for (k in s) printf " %s=%d", k, s[k]}'; }
foreign_log(){ local tag=$1 since=$2 f; f=$(foreign "$NAME" "$since"); echo "$f" > "$A/foreign-$tag.txt"; log "FOREIGN $BOOT $tag: ${f:-0} (non-127.0.0.1 POST /v1 since $since)"
  [[ ${f%% *} == 0 ]] || log "WARN $BOOT: foreign clients served during $tag — arm will be INVALID in the verdict (G0b)"; }
wait_foreign_quiet(){ local i n since=$1; n=$(foreign "$NAME" "$since"); [[ ${n%% *} == 0 ]] && return 0   # clean since boot: no wait
  for ((i=0;i<10;i++)); do since=$(date -u +%FT%TZ); sleep "$FOREIGN_QUIET_S"
    n=$(foreign "$NAME" "$since"); [[ ${n%% *} == 0 ]] && return 0; log "HOLD $BOOT: foreign POSTs in the last ${FOREIGN_QUIET_S}s: $n"; done
  log "WARN $BOOT: foreign traffic never quiet — measuring anyway (G0b marks the arm)"; return 0; }
wait_bind(){ # name maxmin since [nofastfail] [sidecar]
  local n=$1 maxmin=$2 since=$3 nff=${4:-} side=${5:-} i t0 lf=$R/.wb-$1.log; t0=$(date +%s)
  for ((i=0;i<maxmin*6;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $(( $(date +%s)-t0 ))s"; return 0; }
    is_up "$n" || { log "DIED $n after $(( $(date +%s)-t0 ))s"; return 1; }
    if [[ -n $side ]] && ! is_up "$side"; then log "DIED sidecar $side after $(( $(date +%s)-t0 ))s"; return 1; fi
    docker logs --since "$since" "$n" > "$lf" 2>&1
    if [[ -z $nff ]] && grep -qE "$FF" "$lf"; then
      log "FASTFAIL $n after $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < "$lf" | grep -E "$FF" | tail -3 | cut -c1-240 | tr '\n' '|')"; return 1; fi
    (( i % 30 == 0 )) && log "  $n waiting $((i/6)) min: MemAvail=$(memavail)GiB last=[$(tr '\r' '\n' < "$lf" | grep -E 'MEGA_PEER|Offload|offload|Autotun|Capturing|KV cache size|Loading safetensors|Loading weights|compile' | tail -1 | cut -c1-160)]"
    sleep 10
  done; log "TIMEOUT $n ($maxmin min)"; return 1; }
errcount(){ # tag name since — same narrow regex as the fast-fail
  docker logs --since "$3" "$2" > "$R/postlog-$1.log" 2>&1; grep -cE "$FF" "$R/postlog-$1.log" > "$R/errcount-$1.txt"
  log "ERRCOUNT $1 $(cat "$R/errcount-$1.txt") (since $3)"; }
health(){ local tag=$1 side=${2:-} code mc
  mc=$(curl -s -o /dev/null -w '%{http_code}' -m 10 $U/v1/models)
  code=$(curl -s -o "$R/health-$tag-chat.json" -w '%{http_code}' -m 180 -H 'Content-Type: application/json' $U/v1/chat/completions \
    -d '{"model":"'$M'","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":64,"temperature":0,"reasoning_effort":"low"}')
  log "HEALTH $tag: models_http=$mc chat_http=$code (reasoning_effort low) content=[$(python3 -c "import json;print(((json.load(open('$R/health-$tag-chat.json'))['choices'][0]['message'] or {}).get('content') or '')[:40])" 2>/dev/null)]${side:+ sidecar=$(docker inspect -f '{{.State.Running}}' "$side" 2>/dev/null)}"; }
gauntlet(){ local tag=$1
  rm -rf "$R/gauntlet-$tag"
  bg bash -c 'cd "$1" && shift && exec timeout 1200 python3 run_suite.py "$@"' _ "$GDIR" --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "br1-$tag" --cache-salt "br1-$tag" --sandbox-image "$GIMG" --workers 4 > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"; }
BOOT="-"; NAME=""; A=""
ph(){ printf '{"t": %s, "ev": "%s", "boot": "%s"%s}\n' "$(date +%s.%N)" "$1" "$BOOT" "${2:+, $2}" >> "$R/phases.jsonl"; }
setphase(){ printf '%s\n' "$1" > "$R/.phase.tmp" && mv "$R/.phase.tmp" "$R/.phase"; }
bg(){ "$@" & local p=$!; wait $p; }   # long children in background so TERM/INT traps fire immediately
check_abort(){ [[ -e $R/ABORT ]] && { log "ABORT file seen ($(head -1 "$R/ABORT" 2>/dev/null))"; exit 7; }; return 0; }

# ---- MemAvailable sampler (5 s): "<epoch> <phase> <GiB>" -> $R/mem.tsv; live guard writes ABORT below MEM_GUARD
mem_sampler(){ local m p g=0
  while :; do m=$(memavail); p=$(cat "$R/.phase" 2>/dev/null || echo -); echo "$(date +%s) $p $m" >> "$R/mem.tsv"
    if [[ $g == 0 ]] && awk -v m="$m" -v t="$MEM_GUARD" 'BEGIN{exit !(m < t)}'; then
      g=1; echo "MEM_GUARD: MemAvailable ${m} GiB < ${MEM_GUARD} during $p" > "$R/ABORT"; log "MEM_GUARD tripped: ${m} GiB during $p (ABORT written)"; fi
    sleep 5; done; }

# ---- BR1 card watchdog (one per boot; enforce). Killed BEFORE the pair is stopped (else it trips on the sidecar stop).
card_wd_pids(){ ps -eo pid=,comm=,args= | awk '$2=="python3" && index($0,"sidecar_watchdog.py") && index($0,"--server dsv41-vllm-1017-BR1-") {print $1}' | tr '\n' ' '; }
arm_wd(){ [[ -s $CWDLOG ]] && mv "$CWDLOG" "$R/watchdog-stale-$(date +%H%M%S).log"
  rm -f "$CWDSTAT"
  setsid nohup python3 "$WD" --server "$1" --mode enforce --interval 1 --stall 2 \
    --status "$CWDSTAT" --log "$CWDLOG" > /dev/null 2>&1 < /dev/null &
  echo $! > "$CWDPID"; sleep 3
  kill -0 "$(cat "$CWDPID")" 2>/dev/null && log "BR1 watchdog armed pid $(cat "$CWDPID") on $1 (enforce, stall 2 s)" || log "WARN: BR1 watchdog did not start on $1"; }
wd_collect(){ # arm -> $R/<arm>/wd.json ; must run BEFORE kill_wd
  local arm=$1 alive=false p; p=$(cat "$CWDPID" 2>/dev/null)
  [[ -n $p ]] && kill -0 "$p" 2>/dev/null && alive=true
  python3 - "$CWDSTAT" "$CWDLOG" "$alive" "$R/$arm/wd.json" <<'PY' 2>/dev/null || echo '{"error": "wd_collect failed"}' > "$R/$arm/wd.json"
import json, sys
stat, logf, alive, out = sys.argv[1:5]
d = {"alive_at_collect": alive == "true", "status": None, "timeouts": None, "trips": None}
try:
    st = json.load(open(stat)); d["status"] = st
    if st.get("words") is not None and st.get("timeouts_base") is not None:
        d["timeouts"] = st["words"][2] - st["timeouts_base"]
except Exception as e:
    d["status_error"] = repr(e)
try:
    L = open(logf).read().splitlines()
    d["trips"] = sum(1 for l in L if " TRIP " in l); d["log_lines"] = len(L); d["trip_lines"] = [l for l in L if " TRIP " in l][:5]
except Exception as e:
    d["log_error"] = repr(e)
json.dump(d, open(out, "w"), indent=1)
PY
  log "WATCHDOG $arm: $(python3 -c "import json;d=json.load(open('$R/$arm/wd.json'));print('alive',d.get('alive_at_collect'),'timeouts',d.get('timeouts'),'trips',d.get('trips'))" 2>&1 | tail -1)"; }
kill_wd(){ local p; p=$(cat "$CWDPID" 2>/dev/null)
  [[ -n $p ]] && kill "$p" 2>/dev/null && log "BR1 watchdog $p stopped"
  for p in $(card_wd_pids); do kill "$p" 2>/dev/null; done; sleep 2
  for p in $(card_wd_pids); do log "WARN: BR1 watchdog $p still alive; kill -9"; kill -9 "$p" 2>/dev/null; done
  rm -f "$CWDPID"; return 0; }
stop_pair(){ local n=$1
  kill_wd
  is_up "$n" && { log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; }
  is_up "$n-rtx" && { log "stop-and-keep $n-rtx"; docker stop -t 30 "$n-rtx" >/dev/null; }
  [[ -n $A && -s $CWDLOG ]] && mv "$CWDLOG" "$A/watchdog.log"
  ph stopped "\"name\": \"$n\""; return 0; }

# ---- production fail-stop watchdog (M2): argv captured from /proc at preflight; re-armed with the SAME argv (log pinned to $WDLOG)
WARGV=$R/watchdog-argv.txt; WCWD=$R/watchdog-cwd.txt; NEWWD=""
arm_prod_wd(){ local -a A=(); local i cwd p pp
  [[ -s $WARGV ]] || { log "WARN: no captured watchdog argv — watchdog NOT re-armed (needs hands)"; return 1; }
  mapfile -t A < "$WARGV"
  for ((i=0;i<${#A[@]};i++)); do [[ ${A[$i]} == --log && $((i+1)) -lt ${#A[@]} ]] && A[$((i+1))]=$WDLOG; done
  cwd=$(cat "$WCWD" 2>/dev/null); [[ -d $cwd ]] || cwd=$D
  log "re-arm watchdog: (cd $cwd) ${A[*]}"
  ( cd "$cwd" || exit 1; setsid nohup "${A[@]}" > /dev/null 2>&1 < /dev/null & )   # double-fork; ';' not '&&' so ONLY setsid is backgrounded (M2's '&&' form backgrounds the AND-list and leaves a bash copy as the parent, PPID != 1)
  sleep 4
  for p in $(wd_pids); do NEWWD=$p; done
  if [[ -n $NEWWD ]] && kill -0 "$NEWWD" 2>/dev/null; then
    echo "$NEWWD" > "$D/logs/sidecar_watchdog.pid"
    pp=$(ps -o ppid= -p "$NEWWD" 2>/dev/null | tr -d ' ')
    log "fail-stop watchdog armed pid $NEWWD (ppid $pp) on $V25: $(tr '\0' ' ' < /proc/$NEWWD/cmdline | cut -c1-300) | log tail: $(tail -1 "$WDLOG" 2>/dev/null | cut -c1-160)"
    if [[ $pp == 1 ]]; then log "watchdog PPID 1 OK (reparented to init; survives this runner's EXIT trap)"
    else log "WARN: watchdog PPID $pp != 1 ($(ps -o comm=,args= -p "$pp" 2>/dev/null | cut -c1-120))$([[ $pp == "$$" ]] && echo ' — IT IS THIS RUNNER: the EXIT trap will end it; needs hands')"; fi
    printf '{"pid": %s, "ppid": "%s", "ppid_is_1": %s}\n' "$NEWWD" "$pp" "$([[ $pp == 1 ]] && echo true || echo false)" > "$R/restore-watchdog.json"
  else log "WARN: watchdog did not start — lane serving WITHOUT fail-stop guard (needs hands)"; NEWWD=""
    echo '{"pid": null, "ppid": null, "ppid_is_1": false}' > "$R/restore-watchdog.json"; fi; }

LANE_TOUCHED=0; RESTORED=0; FINISHED=0; SAMPLER=""
restore_lane(){ # idempotent: BR1 pairs stop-and-keep, v25m-kvtier pair docker start (flags kept), fail-stop watchdog re-armed
  [[ $RESTORED == 1 ]] && return 0; RESTORED=1
  local n since t0 i
  BOOT=restore; setphase restore; ph restore_start
  kill_wd
  for n in $N_A $N_B $N_A2; do is_up "$n" && { log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; }
    is_up "$n-rtx" && { log "stop-and-keep $n-rtx"; docker stop -t 30 "$n-rtx" >/dev/null; }; done
  drop_caches; strays   # no stop_all here (M2): an abort before the pair stop must not bounce production
  unisolate   # production must be reachable from the LAN again before it binds
  t0=$(date +%s); since=$(date -u +%FT%TZ)
  log "RESTORE start: RTX used=$(rtx_used)MiB /dev/shm/vllm_peer_tier2=[$(ls -la /dev/shm/vllm_peer_tier2 2>&1 | cut -c1-120)] (reused, same layout; never deleted)"
  docker start "$V25R" >/dev/null 2>&1 || log "WARN docker start $V25R rc=$?"
  for ((i=0;i<36;i++)); do docker logs --since "$since" "$V25R" > "$R/.restore-rtx.log" 2>&1
    grep -qE 'serving|prepared 40 layers' "$R/.restore-rtx.log" && break; sleep 5; done
  log "RESTORE sidecar $V25R running=$(docker inspect -f '{{.State.Running}}' "$V25R" 2>/dev/null) after $(( $(date +%s)-t0 ))s: $(grep -m1 -E 'serving|prepared 40 layers' "$R/.restore-rtx.log" | cut -c1-120)"
  docker start "$V25" >/dev/null 2>&1 || log "WARN docker start $V25 rc=$?"
  if wait_bind "$V25" 45 "$since" nofastfail "$V25R"; then
    log "LANE DARK ends ($V25 bound after $(( $(date +%s)-t0 ))s)"
    docker logs --since "$since" "$V25" > "$R/restore-v25m-kvtier.log" 2>&1
    docker inspect -f '{{json .Args}}' "$V25" > "$R/args-after-$V25.json"; docker inspect -f '{{json .Config.Env}}' "$V25" > "$R/env-after-$V25.json"
    log "RESTORE FACTS rowmap40=$(grep -cE 'MEGA_PEER rowmap .*: 40 layers' "$R/restore-v25m-kvtier.log") peer2=$(grep -c 'peer mode 2, check 0' "$R/restore-v25m-kvtier.log") kvline=[$(tr '\r' '\n' < "$R/restore-v25m-kvtier.log" | grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens')] kvdecl=$(grep -c 9126805504 "$R/args-after-$V25.json") kvtier=$(grep -c 'kv-offloading-size' "$R/args-after-$V25.json") block=$(grep -o rejection_sample_method "$R/args-after-$V25.json" | wc -l | tr -d ' ') args_same=$(cmp -s "$R/args-before-$V25.json" "$R/args-after-$V25.json" && echo 1 || echo 0) env_same=$(cmp -s "$R/env-before-$V25.json" "$R/env-after-$V25.json" && echo 1 || echo 0)"
    health restore "$V25R"; arm_prod_wd
    own "cardBR1 done; lane restored to $V25 pair + watchdog pid ${NEWWD:-NONE} (ppid $(ps -o ppid= -p "${NEWWD:-0}" 2>/dev/null | tr -d ' ')); ISOLATE removed (receipts $R)"
  else
    log "RESTORE FAILED — lane dark, needs hands (BRIEF-BR1.md §Restore)"
    own "OWNER: cardBR1 RESTORE FAILED — lane dark; see $R/runner.log (manual: docker start $V25R; docker start $V25; re-arm watchdog from $WARGV)"
  fi
  rm -f "$D/logs/MAINTENANCE"; ph restore_end; }
on_exit(){ local rc=$?
  exec >> "$R/runner.log" 2>&1   # undo any cell-level redirect the signal arrived under
  pkill -P $$ 2>/dev/null
  [[ $LANE_TOUCHED == 1 ]] && unisolate
  if [[ $FINISHED == 0 ]]; then
    log "EXIT rc=$rc before END (lane_touched=$LANE_TOUCHED)"
    if [[ $LANE_TOUCHED == 1 ]]; then restore_lane
      python3 "$VERDICT" "$R" > "$R/VERDICT.txt" 2> "$R/verdict.err"; log "partial verdict rc=$? -> VERDICT.txt (not ledgered)"
      for n in $ALLN; do exists "$n" && docker rename "$n" "$n-ABORTED" 2>/dev/null && log "kept $n-ABORTED (stopped)"; done
    fi
    log "===== CARD BR1 END (ABORTED rc=$rc; lane = $(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ') watchdog=$(wd_pids)) ====="
  fi; }
trap on_exit EXIT
trap 'log "SIGNAL received"; exit 143' TERM INT HUP

boot_br1(){ # arm name spec_extra
  local arm=$1 name=$2 spec=$3 since t0 ma want
  BOOT=$arm; NAME=$name; A=$R/$arm; mkdir -p "$A/metrics"
  check_abort
  setphase "$arm:preboot"
  wait_quiet "boot-$arm" || log "WARN: host not quiet before boot $arm — proceeding (lane already dark)"
  stop_all; drop_caches; strays
  settle "pre-$arm" || return 1
  ma=$(memavail); if awk -v m="$ma" -v t="$MEM_MIN" 'BEGIN{exit !(m < t)}'; then log "ABORT boot $arm: MemAvailable ${ma} GiB < $MEM_MIN after stop+drop_caches (stray?)"; return 1; fi
  docker rm -f "$name" "$name-rtx" >/dev/null 2>&1 || true   # preflight guaranteed none pre-existed; clears only this run's half-made pair
  t0=$(date +%s); since=$(date -u +%FT%TZ); echo "$since" > "$A/boot_since.txt"
  setphase "$arm:boot"; ph boot_start "\"name\": \"$name\", \"spec_extra\": \"${spec//\"/\\\"}\""
  log "BOOT $arm: $name EXTRA=[$KVX] SPEC_EXTRA=[$spec] ROWMAP=$ROWMAP MemAvail=${ma}GiB"
  bg env NAME="$name" ROWMAP="$ROWMAP" EXTRA="$KVX" SPEC_EXTRA="$spec" bash "$LAUNCH" sidecar >> "$R/launch-$arm.txt" 2>&1 || { log "LAUNCH sidecar FAILED $arm: $(tail -2 "$R/launch-$arm.txt" | tr '\n' '|')"; return 1; }
  log "SIDECAR $arm serving: $(grep -m1 '^serving' "$D/logs/peer_server2.log" 2>/dev/null | cut -c1-120)"
  bg env NAME="$name" ROWMAP="$ROWMAP" EXTRA="$KVX" SPEC_EXTRA="$spec" bash "$LAUNCH" server >> "$R/launch-$arm.txt" 2>&1 || { log "LAUNCH server FAILED $arm: $(tail -2 "$R/launch-$arm.txt" | tr '\n' '|')"; return 1; }
  log "SERVER $arm launched: $(tail -1 "$R/launch-$arm.txt")"
  docker inspect -f '{{json .Args}}' "$name" > "$A/args.json" 2>&1; docker inspect -f '{{json .Config.Env}}' "$name" > "$A/env.json" 2>&1
  docker inspect -f '{{json .Args}}' "$name-rtx" > "$A/rtx-args.json" 2>&1; docker inspect -f '{{.Image}}' "$name" > "$A/image.txt" 2>&1
  arm_wd "$name"
  if ! wait_bind "$name" "$BOOT_MIN" "$since" "" "$name-rtx"; then docker logs "$name" > "$A/boot-FAIL.log" 2>&1; return 1; fi
  log "boot->ready $arm $(( $(date +%s)-t0 ))s ($name)"
  echo "$(date +%s) $arm:boot $(memavail)" >> "$R/mem.tsv"   # guarantees >=1 boot sample even on a fast boot
  docker logs --since "$since" "$name" > "$A/boot.log" 2>&1
  tr '\r' '\n' < "$A/boot.log" | grep -E 'KV cache size|Maximum concurrency|MEGA_PEER rowmap|peer mode|OffloadingConnector|KVConnector|speculative_config=' | cut -c1-400 > "$A/boot-facts.log"
  tr '\r' '\n' < "$A/boot.log" | grep -E 'Autotuning process (starts|ends)|Loaded [0-9]+ configs|Saved [0-9]+ configs' | cut -c1-300 > "$A/boot-autotune.log"
  tr '\r' '\n' < "$A/boot.log" | grep -iE 'block.{0,40}(verif|rejection)|(verif|rejection).{0,40}block' | cut -c1-300 > "$A/boot-block.log"
  local kv blk bl ffb ld sv as ae
  kv=$(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$A/boot-facts.log" | grep -oE '[0-9,]+' | head -1)
  blk=$(grep -o 'rejection_sample_method' "$A/args.json" | wc -l | tr -d ' ')
  bl=$(grep -c . "$A/boot-block.log"); ffb=$(tr '\r' '\n' < "$A/boot.log" | grep -cE "$FF")
  ld=$(grep -cE 'Loaded [0-9]+ configs' "$A/boot-autotune.log"); sv=$(grep -cE 'Saved [0-9]+ configs' "$A/boot-autotune.log")
  as=$(grep -c 'Autotuning process starts' "$A/boot-autotune.log"); ae=$(grep -c 'Autotuning process ends' "$A/boot-autotune.log")
  printf '{"arm": "%s", "name": "%s", "kv_tokens": "%s", "kv_expect": "%s", "block_in_args": %s, "block_boot_lines": %s, "ff_boot": %s, "loaded_lines": %s, "saved_lines": %s, "autotune_starts": %s, "autotune_ends": %s, "boot_s": %s, "bound_at": %s}\n' \
    "$arm" "$name" "${kv:-}" "$KV_EXPECT" "${blk:-0}" "${bl:-0}" "${ffb:-0}" "${ld:-0}" "${sv:-0}" "${as:-0}" "${ae:-0}" "$(( $(date +%s)-t0 ))" "$(date +%s)" > "$A/facts.json"
  want=0; [[ -n $spec ]] && want=1
  log "FACTS $arm: kv=[${kv:-MISSING}] expect=$KV_EXPECT block_in_args=$blk (want $want) block_boot_lines=$bl ff_boot=$ffb autotune loaded=$ld saved=$sv starts=$as ends=$ae args_eq_prod=$(cmp -s "$A/args.json" "$R/args-before-$V25.json" && echo 1 || echo 0)"
  [[ ${blk:-0} == "$want" ]] || log "WARN $arm: block_in_args=$blk, expected $want (G0 will FAIL; cells continue for receipts)"
  [[ ${kv:-} == "$KV_EXPECT" ]] || log "WARN $arm: GPU KV cache size [${kv:-MISSING}] != $KV_EXPECT (G0 will FAIL; cells continue for receipts)"
  health "$arm" "$name-rtx"
  foreign_log bind "$since"
  ph boot_ready "\"name\": \"$name\""
  return 0; }

alive(){ is_up "$NAME" && is_up "$NAME-rtx" && return 0; log "ARM $BOOT DEAD ($NAME up=$(is_up "$NAME" && echo y || echo n) rtx up=$(is_up "$NAME-rtx" && echo y || echo n)) — skipping remaining cells"; return 1; }
metrics(){ curl -s -m 10 $U/metrics 2>/dev/null | grep -E '^vllm:spec_decode_' > "$A/metrics/$1.txt"; }   # spec-decode counters at a cell boundary
cell(){ # tag cmd... -> runs with bg, logs one line; /metrics spec-decode snapshot at start and end
  local tag=$1; shift; check_abort; alive || return 1
  ph cell_start "\"cell\": \"$tag\""; metrics "$tag-start"
  bg "$@" > "$A/$tag.log" 2>&1; local rc=$?
  metrics "$tag-end"; ph cell_end "\"cell\": \"$tag\", \"rc\": $rc"
  log "CELL $BOOT $tag rc=$rc: $(tail -1 "$A/$tag.log" | cut -c1-330)"; return 0; }
run_cells(){ local arm=$1 sinceC r ta
  sinceC=$(date -u +%FT%TZ); ta=$(date +%s); setphase "$arm:cells"; ph cells_start
  wait_foreign_quiet "$(cat "$A/boot_since.txt" 2>/dev/null || echo "$sinceC")"
  # (1) PRIMARY: per-class window at temperature 1.0 (block verification is a no-op on greedy rows)
  cell pct1 bash -c 'cd "$1" && ARM="$2" OUTDIR="$3" TEMP="$4" SEED="$5" TOP_P="$6" DEC="$7" exec timeout "$8" bash "$9"' _ \
    "$BENCH" "BR1-$arm-t1" "$A/perclass-t1" "$TEMP_T" "$SEED_T" "$TOP_P_T" "$PCDEC_T" "$T_PERCLASS" "$PERCLASS_T"
  # (2) no-op sanity: the STOCK temp-0 window (perclass_run.sh has no r1-only switch -> full r1+r2)
  cell pct0 bash -c 'cd "$1" && ARM="$2" OUTDIR="$3" exec timeout "$4" bash perclass_run.sh' _ "$BENCH" "BR1-$arm-t0" "$A/perclass-t0" "$T_PERCLASS"
  # (3) mixed C1 x3 at temperature 1.0 (ar_mixed_v2 call shape + sampling flags)
  for r in $(seq 1 "$C1_REPS"); do
    cell "c1-r$r" timeout "$T_MIXED" env HELDOUT=1 python3 "$MIXED_T" "BR1-$arm-c1-r$r" "$A" 1 "$MIX_S" --temperature "$TEMP_T" --seed "$SEED_T" --top-p "$TOP_P_T" || break
  done
  # (4) teacher-forced TF capture (prefill only; temperature-independent) -> $A/tf-BR1-<arm>.jsonl
  cell tf bash -c 'cd "$1" && MAXPOS=4096 CONC=4 exec timeout "$2" python3 "$3" capture "$4" "$5"' _ "$A" "$T_TF" "$TF" "BR1-$arm" "$CORPUS"
  # (5) gauntlet (temp 0, seed 42, effort low: run_suite.py request_settings)
  check_abort; alive && { ph cell_start '"cell": "gauntlet"'; metrics gauntlet-start; gauntlet "$arm"; metrics gauntlet-end; ph cell_end '"cell": "gauntlet"'; }
  # (6) per-turn tool-call emission (replay_turns.py: temp 0, thinking off; writes $TURNS_OUT/turns-<TAG>.json)
  cell turns env TAG="BR1-$arm" PASSES=2 timeout "$T_TURNS" python3 "$TURNS"
  [[ -s $TURNS_OUT/turns-BR1-$arm.json ]] && mv "$TURNS_OUT/turns-BR1-$arm.json" "$A/" && log "TURNS $arm: moved turns-BR1-$arm.json into $A"
  # (7) post
  setphase "$arm:post"; ph cells_end
  errcount "$arm" "$NAME" "$sinceC"; errcount "$arm-rtx" "$NAME-rtx" "$sinceC"
  foreign_log cells "$sinceC"
  wd_collect "$arm"
  log "ARM $arm cells took $(( $(date +%s)-ta ))s"; }

# ================= preflight (refusals exit 4/5/6 with the lane untouched) =================
log "===== CARD BR1 START (arms=$ARMS spec_B=[$SPEC_B] temp_t=$TEMP_T seed_t=$SEED_T mix_s=$MIX_S c1_reps=$C1_REPS settle_s=$SETTLE_S isolate=$ISOLATE mem_min=$MEM_MIN boot_min=$BOOT_MIN R=$R) ====="
if [[ -n $WAIT_PID ]]; then log "WAIT for pid $WAIT_PID to exit"; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; log "pid $WAIT_PID gone"; fi
log "LANE-OWNER.txt (tail): $(tail -3 "$OWNER" 2>/dev/null | tr '\n' ' ' | cut -c1-500)"
wo=$(wd_orphan_parent); [[ -n $wo ]] && log "NOTE: ignoring pid $wo in the runner check: orphaned bash (PPID 1) whose only child is the production watchdog ($(ps -o args= -p "$wo" | cut -c1-100))"
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
x=$(not_known); [[ -n ${x// /} ]] && { log "REFUSE: running container(s) outside the production pair {$V25, $V25R}: $x"; exit 4; }
is_up "$V25" && is_up "$V25R" || { log "ABORT: $V25 pair not running — lane untouched"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: :30006 not bound — lane untouched"; exit 5; }
for n in $ALLN; do exists "$n" && { log "ABORT: BR1 container already exists: $n"; exit 5; }; done
ls "$TURNS_OUT"/turns-BR1-* >/dev/null 2>&1 && { log "ABORT: $TURNS_OUT/turns-BR1-* already exist (tag clash)"; exit 5; }
sudo -n iptables -S INPUT 2>/dev/null | grep -q "$FW_TAG" && { log "ABORT: a $FW_TAG firewall rule is already present (stale run?) — lane untouched"; exit 5; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: $IMAGE missing"; exit 5; }
docker image inspect "$GIMG" >/dev/null 2>&1 || { log "ABORT: gauntlet sandbox image missing"; exit 5; }
[[ -f $MODEL/config.json ]] || { log "ABORT: model missing $MODEL"; exit 5; }
[[ "$(docker inspect -f '{{.Image}}' "$V25" 2>/dev/null)" == "${IMAGE#*@}" ]] || { log "ABORT: $V25 does not run $IMAGE — the clone would not be production"; exit 5; }
chk(){ [[ -s $1 ]] || { log "ABORT: missing $1"; exit 5; }; [[ -z ${2:-} || "$(sha1sum "$1" | cut -c1-8)" == "$2" ]] || { log "ABORT: sha1 $(sha1sum "$1" | cut -c1-12) != $2 for $1"; exit 5; }; }
chk "$LAUNCH" "$SHA_LAUNCH"; chk "$VERDICT" "$SHA_VERDICT"; chk "$PERCLASS_T" "$SHA_PERCLASS_T"; chk "$PCDEC_T" "$SHA_PCDEC_T"; chk "$MIXED_T" "$SHA_MIXED_T"
chk "$PERCLASS" "$SHA_PERCLASS"; chk "$PCDEC" "$SHA_PCDEC"; chk "$MIXED" "$SHA_MIXED"; chk "$WD" "$SHA_WD"; chk "$TF" "$SHA_TF"; chk "$TURNS" "$SHA_TURNS"
chk "$M3HOOK/$ROWMAP" "$SHA_ROWMAP"; chk "$M3HOOK/sitecustomize.py" "$SHA_SITE"; chk "$D/local/peer_server2_local.py" "$SHA_PEER"
chk "$LAUNCH_REF" "$SHA_LREF"; chk "$PCTAB" "$SHA_PCTAB"; chk "$X3" "$SHA_X3"; chk "$PCPROMPTS" "$SHA_PCPROMPTS"; chk "$FLIPS" "$SHA_FLIPS"
chk "$CORPUS" "$SHA_CORPUS"; chk "$FIXTURE" "$SHA_FIXTURE"; chk "$GDIR/run_suite.py"
# launcher drift: the clone may differ from launch_sidecar.sh ONLY by comments, the SPEC_EXTRA definition and the splice
diff "$LAUNCH_REF" "$LAUNCH" > "$R/DIFF-launch-preflight.txt"
h=$(grep -cE '^[0-9]' "$R/DIFF-launch-preflight.txt")
dd=$(diff <(grep -v '^#' "$LAUNCH_REF") <(grep -v '^#' "$LAUNCH" | grep -vxF 'SPEC_EXTRA=${SPEC_EXTRA:-}' | sed 's/\$SPEC_EXTRA}/}/') | grep -cE '^[<>]')
sd=$(grep -cxF 'SPEC_EXTRA=${SPEC_EXTRA:-}' "$LAUNCH"); sp=$(grep -F -- '--speculative-config' "$LAUNCH" | grep -cF '$SPEC_EXTRA}')
[[ $h == 3 && $dd == 0 && $sd == 1 && $sp == 1 ]] || { log "ABORT: $LAUNCH drifted from $LAUNCH_REF (hunks=$h non-SPEC_EXTRA changed lines=$dd SPEC_EXTRA defs=$sd splices=$sp; want 3/0/1/1) — see DIFF-launch-preflight.txt"; exit 5; }
grep -q 'DEC=' "$PERCLASS_T" && grep -q -- '--temperature "\$TEMP"' "$PERCLASS_T" || { log "ABORT: $PERCLASS_T does not pass --temperature \$TEMP to DEC"; exit 5; }
u=$(rtx_used)
{ for f in "$0" "$LAUNCH" "$LAUNCH_REF" "$VERDICT" "$PERCLASS_T" "$PCDEC_T" "$MIXED_T" "$PERCLASS" "$PCDEC" "$PCTAB" "$X3" "$PCPROMPTS" "$MIXED" "$WD" "$TF" "$FLIPS" "$CORPUS" \
      "$TURNS" "$FIXTURE" "$GDIR/run_suite.py" "$M3HOOK/$ROWMAP" "$M3HOOK/sitecustomize.py" "$D/local/peer_server2_local.py"; do echo "sha1 $(sha1sum "$f" | cut -c1-12) $f"; done
  echo "image $IMAGE"; echo "gauntlet image $GIMG"; echo "all arms EXTRA=[$KVX]"; echo "B SPEC_EXTRA=[$SPEC_B]"
  echo "sampling for pct1/mixed: temperature=$TEMP_T seed=$SEED_T (+row/prompt index) top_p=$TOP_P_T"
  echo "iptables :30006 rules before: $(sudo -n iptables -S INPUT 2>/dev/null | grep 30006 | tr '\n' '|')"; } > "$R/instruments.txt"
docker ps -a > "$R/preflight-docker-ps-a.txt" 2>&1; nvidia-smi > "$R/preflight-nvidia-smi.txt" 2>&1; free -g > "$R/preflight-free-g.txt" 2>&1
df -h /dev/shm >> "$R/preflight-free-g.txt" 2>&1
shm_big > "$R/preflight-shm-big.txt"; host_capture "$R/preflight-host.txt" preflight
docker inspect -f '{{json .Args}}' "$V25" > "$R/args-before-$V25.json"; docker inspect -f '{{json .Config.Env}}' "$V25" > "$R/env-before-$V25.json"
docker inspect -f '{{json .Args}}' "$V25R" > "$R/args-before-$V25R.json"
grep -q 9126805504 "$R/args-before-$V25.json" && grep -q 'kv-offloading-size' "$R/args-before-$V25.json" && ! grep -q rejection_sample_method "$R/args-before-$V25.json" \
  || { log "ABORT: production args are not the expected v25m-kvtier recipe (kv decl + tier, no rejection_sample_method)"; exit 5; }
WP=$(wd_pids); WP=${WP%% *}
[[ -n $WP ]] || { log "ABORT: no live sidecar_watchdog.py --server $V25 — refusing (cannot mirror its restore); lane untouched"; exit 5; }
tr '\0' '\n' < "/proc/$WP/cmdline" | sed '/^$/d' > "$WARGV"; readlink "/proc/$WP/cwd" > "$WCWD"
grep -qx -- "--server" "$WARGV" && grep -qx -- "$V25" "$WARGV" || { log "ABORT: captured watchdog argv does not name $V25: $(tr '\n' ' ' < "$WARGV")"; exit 5; }
log "watchdog pid $WP (ppid $(ps -o ppid= -p "$WP" | tr -d ' ')) argv: $(tr '\n' ' ' < "$WARGV") cwd=$(cat "$WCWD")"
OOM0=$(oom_count); echo "${OOM0:-}" > "$R/oom-start.txt"
log "preflight OK: MemAvailable=$(memavail)GiB RTX=${u}MiB oom_lines=${OOM0:-unreadable} (sha1s in instruments.txt; docker ps -a / nvidia-smi / free -g in preflight-*.txt)"
log "lane before: $(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')"
own "cardBR1 runner pid $$ preflight OK — about to take :30006 dark (A=$N_A, B=$N_B, A2=$N_A2; ~3 h); restores $V25 pair + watchdog at END. Receipts $R"
wait_quiet preflight || { log "ABORT: host never quiet; lane untouched"; exit 6; }
check_abort

# ================= lane dark: watchdog off (log moved aside), v25m-kvtier pair stopped-and-kept =================
LANE_TOUCHED=1
echo "cardBR1 $(date '+%F %T %Z')" > "$D/logs/MAINTENANCE"
setphase pre; mem_sampler & SAMPLER=$!
for p in $(wd_pids); do kill "$p" 2>/dev/null && log "fail-stop watchdog $p stopped (cardBR1)"; done
for i in 1 2 3 4 5; do [[ -z $(wd_pids) ]] && break; sleep 1; done
[[ -n $(wd_pids) ]] && { log "ABORT: sidecar watchdog still alive ($(wd_pids)) — refusing to stop $V25 under a fail-stop watchdog"; exit 6; }
echo "$(date '+%F %T %Z') watchdog stopped by cardBR1 (re-armed after restore)" >> "$WDLOG"
mv "$WDLOG" "$WDLOG.pre-cardBR1-$(date +%Y%m%d-%H%M%S)"
BOOT=pre; ph lane_dark
log "LANE DARK begins (stopping $V25 pair, kept)"
own "cardBR1: lane :30006 DARK (watchdog $WP stopped, $V25 + $V25R stopped-and-kept)"
docker stop -t 60 "$V25" "$V25R" >/dev/null 2>&1; log "stop-and-keep $V25 $V25R"
isolate

# ================= arms: A (control) -> B (block verification) -> A2 (control) =================
for arm in $ARMS; do
  case $arm in
    A) n=$N_A; sx="";;
    B) n=$N_B; sx="$SPEC_B";;
    A2) n=$N_A2; sx="";;
    *) log "BUG unknown arm $arm"; continue;;
  esac
  if boot_br1 "$arm" "$n" "$sx"; then
    run_cells "$arm"
  else
    log "ARM $arm FAILED TO BOOT — its cells are MISSING (verdict: B -> FAIL, control -> INVALID)"
    [[ -d $A ]] && { wd_collect "$arm"; foreign_log boot "$(cat "$A/boot_since.txt" 2>/dev/null || date -u +%FT%TZ)"; }
  fi
  stop_pair "$n"
  check_abort
done

# ================= END: restore, TF compares (CPU), verdict, renames, ledger =================
restore_lane
setphase end; kill "$SAMPLER" 2>/dev/null
OOM1=$(oom_count); echo "${OOM1:-}" > "$R/oom-end.txt"
# tf_logprob.py compare takes FILENAMES (tf-<tag>.jsonl), never tags; outputs tf-compare-<ref>-vs-<cand>.json in the cwd
for pr in "A B" "A2 B" "A A2"; do set -- $pr
  if [[ -s $R/$1/tf-BR1-$1.jsonl && -s $R/$2/tf-BR1-$2.jsonl ]]; then
    ( cd "$R" && timeout 600 python3 "$TF" compare "$1/tf-BR1-$1.jsonl" "$2/tf-BR1-$2.jsonl" ) > "$R/tfcmp-$2-vs-$1.log" 2>&1; log "TFCMP $2 vs $1: $(tail -1 "$R/tfcmp-$2-vs-$1.log" | cut -c1-240)"
  else log "TFCMP $2 vs $1: UNMEASURED (missing tf-BR1-$1.jsonl or tf-BR1-$2.jsonl)"; fi; done
# margin-weighted flips (flips_margin_p2.py REF CAND --label --calib); B vs A calibrated on A2 (and vice versa); floor A2 vs A
fm(){ local ref=$1 cand=$2 cal=${3:-} out=$R/flipsm-$2-vs-$1.txt
  [[ -s $R/$ref/tf-BR1-$ref.jsonl && -s $R/$cand/tf-BR1-$cand.jsonl ]] || { log "FLIPSM $cand vs $ref: UNMEASURED (missing capture)"; return 0; }
  ( cd "$R" && timeout 600 python3 "$FLIPS" "$ref/tf-BR1-$ref.jsonl" "$cand/tf-BR1-$cand.jsonl" --label "$cand-vs-$ref" ${cal:+--calib "$cal/tf-BR1-$cal.jsonl"} ) > "$out" 2>&1
  log "FLIPSM $cand vs $ref: $(tail -1 "$out" | cut -c1-200)"; }
fm A B A2; fm A2 B A; fm A A2
python3 "$VERDICT" "$R" > "$R/VERDICT.txt" 2> "$R/verdict.err"; vrc=$?
SUF=$(python3 -c "import json;print(json.load(open('$R/verdict.json'))['suffix'])" 2>/dev/null); SUF=${SUF:-BR1-NOVERDICT}
VL=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["result"],"|",d["suffix"],"|",json.dumps(dict((k,v[0]) for k,v in d["gates"].items())))' "$R/verdict.json" 2>&1 | tail -1 | cut -c1-900)
log "VERDICT rc=$vrc $VL"
for n in $ALLN; do exists "$n" && docker rename "$n" "$n-$SUF" 2>/dev/null && log "kept $n-$SUF (stopped)"; done
grep -hE ' (=====|CELL|BOOT|SIDECAR|SERVER|FACTS|HEALTH|GAUNTLET|TURNS|TFCMP|FLIPSM|WATCHDOG|BR1 watchdog|boot->ready|BOUND|FASTFAIL|DIED|TIMEOUT|ERRCOUNT|FOREIGN|ISOLATE|SETTLE|HOST|HOLD|RESTORE|LANE|VERDICT|MEM_GUARD|WARN|ABORT|REFUSE|ARM|kept|stop-and-keep|fail-stop|re-arm|watchdog) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card BR1: DSpark block verification (rejection_sample_method block) vs sequential, A/B/A2 on the v25m-kvtier recipe, :30006 — $(python3 -c "import json;d=json.load(open('$R/verdict.json'));print(d['result'], d['suffix'])" 2>/dev/null)"
  cat "$R/VERDICT.txt" 2>/dev/null; echo "Receipts: $R. Production restored to $V25 pair + watchdog ${NEWWD:-NONE}. NOT promoted — James decides."; } >> "$LEDGER"
FINISHED=1
log "FINAL dsv41 running=[$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')] bound=$(curl -s -m 3 $U/v1/models | grep -c $M) watchdog=${NEWWD:-NONE} ppid=$(ps -o ppid= -p "${NEWWD:-0}" 2>/dev/null | tr -d ' ') isolate_rule=$(sudo -n iptables -S INPUT 2>/dev/null | grep -c "$FW_TAG")"
log "===== CARD BR1 END (lane = $V25) ====="

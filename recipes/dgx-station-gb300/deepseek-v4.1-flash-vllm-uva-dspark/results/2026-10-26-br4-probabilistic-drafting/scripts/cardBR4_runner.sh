#!/usr/bin/env bash
# cardBR4_runner.sh — Card BR4: DSpark PROBABILISTIC drafting ("draft_sample_method":"probabilistic"), alone and with BLOCK
# verification ("rejection_sample_method":"block") on top, vs the production greedy-draft/sequential config, on the GB300-ONLY
# v26 "Seeded Adaptive" recipe (Grace UVA offload, no RTX sidecar), bracketed A -> P -> PB -> A2, lane :30006, GB300.
# Clone of ~/dsv41/m3/br3/cardBR3_runner.sh @58ea2774 (DIFF-br4-runner-vs-br3.txt shows every change).
# WHY: BR3 (block alone) was NULL on this lane (tokens/verify-step B/ctrl 0.999): the lane drafts GREEDY and block verification only
# reweights a draft DISTRIBUTION. Probabilistic drafting is the untested lever at temperature > 0; block may add on top (BR1: +2.0 %).
# CONTROL = exactly the stopped-kept container dsv41-vllm-1017-v26seed-BOUND-REF (launch_ar6.sh, image 0909, hook v26ar2,
# ROWMAP rowmap-mixed-ar-v1.json, --offload-backend uva --cpu-offload-gb 60, greedy drafting, k-schedule [[1,8,5],[9,24,1]],
# seqs 24). AXES via SPEC_EXTRA spliced into --speculative-config by the REUSED BR3 launcher ~/dsv41/m3/br3/launch_br3.sh
# (absolute path, sha1 pinned 8066045c, not copied):
#   A/A2 : empty                                   (Args byte-identical to v26seed)
#   P    : ,"draft_sample_method":"probabilistic"
#   PB   : ,"draft_sample_method":"probabilistic","rejection_sample_method":"block"
# Neither key is a compute-hash factor (0909 config/speculative.py compute_hash) => hash-hit boot (~474-484 s) expected.
# Probabilistic drafting allocates a draft-logits cache (max_num_reqs x k x vocab, bf16 ~31 MB) in the speculator __init__,
# BEFORE KV profiling => P/PB may show slightly fewer GPU KV tokens than A (predicted ~ -0.6 %; see BRIEF-BR4.md, G0 rule).
# At temperature 0 both keys are no-ops on output (gumbel_noised_argmax = plain argmax at temp 0; the greedy rejection path
# ignores draft logits; block only runs on non-greedy rows) => the stock temp-0 window stays a PARITY cell (G2b).
# The primary speed cells sample at temperature 1.0 (perclass_run_t.sh, ar_mixed_t.py — the BR1 instruments, by absolute path).
# ADAPTIVE RESIDENCY (production regime, M2/AR6 GO+ctrl handling): the hook boots FROZEN (PIN_ADAPTIVE_FREEZE=1, worker parked on
# /ar/GO). After bind every arm gets: touch <run>/GO -> ctrl.json = {"freeze": false, "baseline": true} (byte-identical to the
# v26seed production ~/dsv41/ar/soak/ctrl.json) -> fixed ADAPT_SETTLE_S (90 s) -> cells. Each arm has its OWN /ar run
# dir ($R/<arm>/ar): the production soak dir is never written. swaps.jsonl / telemetry.jsonl line counts at every cell start/end
# -> <arm>/ar-lines.tsv (annotation).
# Detached: cd ~/dsv41/m3/br4 && setsid nohup bash cardBR4_runner.sh > /dev/null 2>&1 < /dev/null & echo $! > cardBR4.pid
# Abort: touch ~/dsv41/m3/br4/results-cardBR4/ABORT (checked between cells/boots) or kill -TERM <pid>.
# ARM PLAN:
#   preflight (refusals leave the lane untouched) -> capture production (v27) watchdog argv+cwd -> stop it (log moved aside)
#     -> stop the v27-block pair (kept) -> ISOLATE -> REFUSE (and restore) if the RTX still holds > 2048 MiB (BR4 is Grace-only)
#   per arm X in A P PB A2: wait_quiet + stop_all + drop_caches + SETTLE + MEM_MIN + RTX idle + GB300 idle -> launch_br3.sh
#       (single step, no sidecar, no card watchdog) -> wait_bind -> boot facts (KV, args/env/binds vs v26seed, hook guard lines,
#       hash-hit shape, prob_in_args, block_in_args) -> GO + unfreeze + 90 s settle
#     cells (each via bg() with a timeout; /metrics vllm:spec_decode_* incl. per-position + /ar line counts at every cell start/end):
#       (1) pct1  : perclass_run_t.sh TEMP=1.0 SEED=42 -> X/perclass-t1   (PRIMARY, G2)
#       (2) pct0  : stock perclass_run.sh (temp 0, full r1+r2) -> X/perclass-t0   (parity, G2b)
#       (3) c1-r1..3 : ar_mixed_t.py HELDOUT=1 C1 120 s at temperature 1.0 (G3)
#       (4) tf    : tf_logprob.py capture (teacher-forced; G4 after restore)
#       (5) gauntlet (tags br4-X; G5)   (6) turns: replay_turns.py PASSES=2 (G6)
#       (7) errcount server, foreign POSTs (G7, G0b)
#     -> stop arm (keep)
#   END: restore production v27 pair + fail-stop watchdog (BR1/BR3 restore_lane; -rtx first, wait 'serving', server; double-forked
#     watchdog, PPID 1 checked; ISOLATE removed first) -> TF compares + margin flips (CPU) -> verdict -> renames -> SUMMARY
#     -> ledger -> LANE-OWNER append -> FINAL
set -uo pipefail
HM=${BR4_HOME:-~}   # desk dry-run override only; production = ~
DS=$HM/dsv41; D=$DS/m3; B1=$D/br1; B3=$D/br3; B4=$D/br4
R=$B4/results-cardBR4
U=http://127.0.0.1:30006; M=dsv41-flash-uva
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909                                         # = launch_ar6.sh / v26seed
IMAGE_ID=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d       # docker image inspect, 2026-10-26
PROD_IMAGE_ID=sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063  # v27 pair (af7f9488), restored at END
MODEL=${BR4_MODEL_DIR:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55; RTX=GPU-d07eb6e6-7053-42a7-6811-7bb833266182
HOOK=$HM/pin-hot-experts/v26ar2/hook
ROWMAP=rowmap-mixed-ar-v1.json
LAUNCH=$B3/launch_br3.sh; LAUNCH_REF=$DS/ar/launch_ar6.sh   # BR3 launcher REUSED by absolute path (not copied)
VERDICT=$B4/br4_verdict.py
MIXED=$DS/ar/ar_mixed_v2.py; MIXED_T=$B1/ar_mixed_t.py
BENCH=$DS/bench; PERCLASS=$BENCH/perclass_run.sh; PCDEC=$BENCH/perclass_decode.py; PCTAB=$BENCH/perclass_table.py
PCPROMPTS=$BENCH/perclass_prompts.jsonl; X3=$DS/1017/agent_fixture_x3.py
PERCLASS_T=$B1/perclass_run_t.sh; PCDEC_T=$B1/perclass_decode_t.py
TF=$HM/pin-hot-experts/scripts/tf_logprob.py; CORPUS=$HM/pin-hot-experts/tf_corpus.jsonl
FLIPS=$DS/1017/flips_margin_p2.py   # CLI: REF.jsonl CAND.jsonl [--label X] [--calib R2B.jsonl]; last line GATE ...
TURNS=$D/replay_turns.py; TURNS_OUT=$D/results-cardSC1b; FIXTURE=$DS/transcript_fixture.json   # replay_turns.py hardcodes its out dir
WD=$D/sidecar_watchdog.py   # PRODUCTION watchdog only (re-armed at restore); BR4 arms run no card watchdog (no sidecar)
REF=dsv41-vllm-1017-v26seed-BOUND-REF   # control ground truth: STOPPED-and-kept, never started/stopped/renamed by this card
V27=dsv41-vllm-1017-v27-block-BOUND-REF; V27R=$V27-rtx
WDLOG=$D/logs/sidecar_watchdog.v27.log   # production fail-stop watchdog log (v27 promotion 2026-10-26 04:08)
N_A=dsv41-vllm-1017-BR4-A; N_P=dsv41-vllm-1017-BR4-P; N_PB=dsv41-vllm-1017-BR4-PB; N_A2=dsv41-vllm-1017-BR4-A2
ALLN="$N_A $N_P $N_PB $N_A2"
SPEC_P=',"draft_sample_method":"probabilistic"'
SPEC_PB=',"draft_sample_method":"probabilistic","rejection_sample_method":"block"'
GO_CTRL='{"freeze": false, "baseline": true}'   # byte-identical to the v26seed production soak/ctrl.json (35 bytes)
RTX_IDLE_MIB=2048; GB_IDLE_MIB=8192
GDIR=$HM/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
LEDGER=$DS/results/ledger.md
OWNER=$D/LANE-OWNER.txt
FF=' ERROR [0-9]{2}-[0-9]{2} |Traceback \(most recent call last\)|ValueError:|RuntimeError:|OutOfMemoryError|EngineCore failed'
# sha1 prefixes (8) — override via env only after re-reviewing the file (set-but-empty disables a check: desk dry-run only)
# launch_br3.sh is the REVIEWED BR3 launcher (used by the finished BR3 run), reused by absolute path: pin filled.
# >>>>> TODO-SHA (Milo, after review + rename of the BR4 drafts): fill the 8-char sha1 prefix of br4_verdict.py below.
# >>>>> TODO-SHA never matches, so the runner REFUSES at preflight until it is filled.
SHA_LAUNCH=${SHA_LAUNCH-8066045c}; SHA_VERDICT=${SHA_VERDICT-41448c5c}
# reused BR1 instruments (absolute paths in br1/, measured on the box 2026-10-26, desk read-only sha1sum)
SHA_PERCLASS_T=${SHA_PERCLASS_T-3dfeb6e8}; SHA_PCDEC_T=${SHA_PCDEC_T-f2b95a01}; SHA_MIXED_T=${SHA_MIXED_T-b12c5a66}
SHA_PERCLASS=${SHA_PERCLASS-682114ad}; SHA_PCDEC=${SHA_PCDEC-09c66432}; SHA_MIXED=${SHA_MIXED-ce5231c1}; SHA_WD=${SHA_WD-b17d9103}
SHA_TF=${SHA_TF-3638227d}; SHA_TURNS=${SHA_TURNS-5f7934a9}
# v26ar2 hook dir (mounted as /w): rowmap, sitecustomize, hook, adaptive module
SHA_ROWMAP=${SHA_ROWMAP-52e2861c}; SHA_SITE=${SHA_SITE-e65a93d7}; SHA_HOOKPY=${SHA_HOOKPY-69bf0777}; SHA_E4=${SHA_E4-7ea55cbd}
SHA_LREF=${SHA_LREF-f9df7b3f}; SHA_PCTAB=${SHA_PCTAB-6e6d3ff8}; SHA_X3=${SHA_X3-9d10d2e3}; SHA_PCPROMPTS=${SHA_PCPROMPTS-618898aa}
SHA_FLIPS=${SHA_FLIPS-affda07d}; SHA_CORPUS=${SHA_CORPUS-3672c1a7}; SHA_FIXTURE=${SHA_FIXTURE-e3f1aea4}; SHA_SUITE=${SHA_SUITE-e5f575dc}
MIX_S=${MIX_S:-120}; C1_REPS=${C1_REPS:-3}; TEMP_T=${TEMP_T:-1.0}; SEED_T=${SEED_T:-42}; TOP_P_T=${TOP_P_T:-1.0}
T_PERCLASS=${T_PERCLASS:-2700}; T_MIXED=${T_MIXED:-600}; T_TF=${T_TF:-1800}; T_TURNS=${T_TURNS:-1500}
MEM_MIN=${MEM_MIN:-380}; MEM_GUARD=${MEM_GUARD:-30}; BOOT_MIN=${BOOT_MIN:-40}; WAIT_PID=${WAIT_PID:-}
SETTLE_S=${SETTLE_S:-90}; SETTLE_MAX_S=${SETTLE_MAX_S:-600}; ISOLATE=${ISOLATE:-1}; FOREIGN_QUIET_S=${FOREIGN_QUIET_S:-60}
ADAPT_SETTLE_S=${ADAPT_SETTLE_S:-90}
ARMS="A P PB A2"   # locked order (not env-overridable)

if [[ -e $R ]]; then echo "$(date '+%F %T %Z') REFUSE: $R exists (refuse to mix runs; move it aside)" | tee -a "$B4/cardBR4.refused.log" >&2; exit 5; fi
mkdir -p "$R"
exec >> "$R/runner.log" 2>&1
exec 3>> "$R/runner.log"   # log() always writes here: a trap firing inside `bg ... > cell.log` inherits that redirect on fd 1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >&3; }
own(){ echo "- $(date '+%F %T %Z') $*" >> "$OWNER"; }   # LANE-OWNER.txt is APPEND-ONLY for this card (M2)
MEMINFO=${MEMINFO:-/proc/meminfo}
memavail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' "$MEMINFO"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; done; sleep 5; }
not_known(){ docker ps --format '{{.Names}}' | grep -vxE "$V27|$V27R" | tr '\n' ' '; }
any_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '; }
wd_pids(){ ps -eo pid=,comm=,args= | awk -v s="$V27" '$2=="python3" && index($0,"sidecar_watchdog.py") && index($0,"--server " s " ") {print $1}' | tr '\n' ' '; }   # python3 process only
wd_orphan_parent(){ # pid of an orphaned bash (PPID 1, one child) holding the production watchdog (BR1)
  local p pp; p=$(wd_pids); p=${p%% *}; [[ -n $p ]] || return 0
  pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [[ -n $pp && $pp != 1 ]] || return 0
  [[ "$(ps -o comm= -p "$pp" 2>/dev/null)" == bash && "$(ps -o ppid= -p "$pp" 2>/dev/null | tr -d ' ')" == 1 \
     && "$(ps --ppid "$pp" -o pid= 2>/dev/null | grep -c .)" == 1 ]] && echo "$pp"; return 0; }
other_runners(){ local skip; skip=$(wd_orphan_parent)
  ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh|perclass_run_t\.sh)' | grep -v "cardBR4_runner.sh" | grep -v grep \
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
gb_used(){ nvidia-smi --id=$GB300 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
oom_count(){ { sudo -n dmesg 2>/dev/null || journalctl -k --no-pager -q 2>/dev/null; } | grep -ciE 'oom-kill|out of memory: kill' ; }
envcmp(){ python3 -c 'import json,sys;print(int(sorted(json.load(open(sys.argv[1])))==sorted(json.load(open(sys.argv[2])))))' "$1" "$2" 2>/dev/null || echo 0; }   # ORDER-INSENSITIVE (docker reorders Env)
spec_el(){ python3 -c 'import json,sys;a=json.load(open(sys.argv[1]));print(a[a.index("--speculative-config")+1])' "$1" 2>/dev/null; }
lc(){ if [[ -f $1 ]]; then grep -c . "$1"; else echo 0; fi; }
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
FW_TAG=cardBR4-isolate
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
wait_bind(){ # name maxmin since [nofastfail] [sidecar]   (sidecar arg used only by the v27 restore)
  local n=$1 maxmin=$2 since=$3 nff=${4:-} side=${5:-} i t0 lf=$R/.wb-$1.log; t0=$(date +%s)
  for ((i=0;i<maxmin*6;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $(( $(date +%s)-t0 ))s"; return 0; }
    is_up "$n" || { log "DIED $n after $(( $(date +%s)-t0 ))s"; return 1; }
    if [[ -n $side ]] && ! is_up "$side"; then log "DIED sidecar $side after $(( $(date +%s)-t0 ))s"; return 1; fi
    docker logs --since "$since" "$n" > "$lf" 2>&1
    if [[ -z $nff ]] && grep -qE "$FF" "$lf"; then
      log "FASTFAIL $n after $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < "$lf" | grep -E "$FF" | tail -3 | cut -c1-240 | tr '\n' '|')"; return 1; fi
    (( i % 30 == 0 )) && log "  $n waiting $((i/6)) min: MemAvail=$(memavail)GiB last=[$(tr '\r' '\n' < "$lf" | grep -E 'PIN_HOT|MEGA_PEER|Offload|offload|Autotun|Capturing|KV cache size|Loading safetensors|Loading weights|compile' | tail -1 | cut -c1-160)]"
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
  bg bash -c 'cd "$1" && shift && exec timeout 1200 python3 run_suite.py "$@"' _ "$GDIR" --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "br4-$tag" --cache-salt "br4-$tag" --sandbox-image "$GIMG" --workers 4 > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"; }
BOOT="-"; NAME=""; A=""; RUN=""
ph(){ printf '{"t": %s, "ev": "%s", "boot": "%s"%s}\n' "$(date +%s.%N)" "$1" "$BOOT" "${2:+, $2}" >> "$R/phases.jsonl"; }
setphase(){ printf '%s\n' "$1" > "$R/.phase.tmp" && mv "$R/.phase.tmp" "$R/.phase"; }
bg(){ "$@" & local p=$!; wait $p; }   # long children in background so TERM/INT traps fire immediately
check_abort(){ [[ -e $R/ABORT ]] && { log "ABORT file seen ($(head -1 "$R/ABORT" 2>/dev/null))"; exit 7; }; return 0; }
# ---- AR conventions (M2/AR6): RUN = the boot's /ar mount; ctrl() atomic tmp+mv then 3 s (hook polls 1 s, mtime-gated)
ctrl(){ printf '%s' "$1" > "$RUN/ctrl.tmp" && mv "$RUN/ctrl.tmp" "$RUN/ctrl.json" && touch "$RUN/ctrl.json"
  ph ctrl "\"ctrl\": $1${2:+, \"why\": \"$2\"}"; log "CTRL $BOOT ${2:-} $1"; sleep 3; }
arlines(){ # tag -> one annotation line: /ar swaps.jsonl + telemetry.jsonl line counts, steps.bin bytes
  [[ -n $RUN && -n $A ]] || return 0
  printf '%s\t%s\tswaps=%s\ttelemetry=%s\tsteps_bytes=%s\n' "$(date +%s)" "$1" "$(lc "$RUN/swaps.jsonl")" "$(lc "$RUN/telemetry.jsonl")" \
    "$(stat -c %s "$RUN/steps.bin" 2>/dev/null || echo 0)" >> "$A/ar-lines.tsv"; }

# ---- MemAvailable sampler (5 s): "<epoch> <phase> <GiB>" -> $R/mem.tsv; live guard writes ABORT below MEM_GUARD
mem_sampler(){ local m p g=0
  while :; do m=$(memavail); p=$(cat "$R/.phase" 2>/dev/null || echo -); echo "$(date +%s) $p $m" >> "$R/mem.tsv"
    if [[ $g == 0 ]] && awk -v m="$m" -v t="$MEM_GUARD" 'BEGIN{exit !(m < t)}'; then
      g=1; echo "MEM_GUARD: MemAvailable ${m} GiB < ${MEM_GUARD} during $p" > "$R/ABORT"; log "MEM_GUARD tripped: ${m} GiB during $p (ABORT written)"; fi
    sleep 5; done; }

# ---- BR4 arms are single GB300-only containers: no sidecar, no card watchdog. stop_arm never touches any -rtx container.
stop_arm(){ local n=$1
  arlines stop
  is_up "$n" && { log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; }
  ph stopped "\"name\": \"$n\""; return 0; }

# ---- production fail-stop watchdog (BR1/M2): argv captured from /proc at preflight; re-armed with the SAME argv (log pinned to $WDLOG)
WARGV=$R/watchdog-argv.txt; WCWD=$R/watchdog-cwd.txt; NEWWD=""
arm_prod_wd(){ local -a A=(); local i cwd p pp
  [[ -s $WARGV ]] || { log "WARN: no captured watchdog argv — watchdog NOT re-armed (needs hands)"; return 1; }
  mapfile -t A < "$WARGV"
  for ((i=0;i<${#A[@]};i++)); do [[ ${A[$i]} == --log && $((i+1)) -lt ${#A[@]} ]] && A[$((i+1))]=$WDLOG; done
  cwd=$(cat "$WCWD" 2>/dev/null); [[ -d $cwd ]] || cwd=$D
  log "re-arm watchdog: (cd $cwd) ${A[*]}"
  ( cd "$cwd" || exit 1; setsid nohup "${A[@]}" > /dev/null 2>&1 < /dev/null & )   # double-fork; ';' not '&&' so ONLY setsid is backgrounded (BR1 fork_test)
  sleep 4
  for p in $(wd_pids); do NEWWD=$p; done
  if [[ -n $NEWWD ]] && kill -0 "$NEWWD" 2>/dev/null; then
    echo "$NEWWD" > "$D/logs/sidecar_watchdog.pid"
    pp=$(ps -o ppid= -p "$NEWWD" 2>/dev/null | tr -d ' ')
    log "fail-stop watchdog armed pid $NEWWD (ppid $pp) on $V27: $(tr '\0' ' ' < /proc/$NEWWD/cmdline | cut -c1-300) | log tail: $(tail -1 "$WDLOG" 2>/dev/null | cut -c1-160)"
    if [[ $pp == 1 ]]; then log "watchdog PPID 1 OK (reparented to init; survives this runner's EXIT trap)"
    else log "WARN: watchdog PPID $pp != 1 ($(ps -o comm=,args= -p "$pp" 2>/dev/null | cut -c1-120))$([[ $pp == "$$" ]] && echo ' — IT IS THIS RUNNER: the EXIT trap will end it; needs hands')"; fi
    printf '{"pid": %s, "ppid": "%s", "ppid_is_1": %s}\n' "$NEWWD" "$pp" "$([[ $pp == 1 ]] && echo true || echo false)" > "$R/restore-watchdog.json"
  else log "WARN: watchdog did not start — lane serving WITHOUT fail-stop guard (needs hands)"; NEWWD=""
    echo '{"pid": null, "ppid": null, "ppid_is_1": false}' > "$R/restore-watchdog.json"; fi; }

LANE_TOUCHED=0; RESTORED=0; FINISHED=0; SAMPLER=""
restore_lane(){ # idempotent: BR4 arms stop-and-keep, v27-block pair docker start (flags kept), fail-stop watchdog re-armed
  [[ $RESTORED == 1 ]] && return 0; RESTORED=1
  local n since t0 i
  BOOT=restore; setphase restore; ph restore_start
  for n in $ALLN; do is_up "$n" && { log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; }; done
  drop_caches; strays   # no stop_all here (M2): an abort before the pair stop must not bounce production
  unisolate   # production must be reachable from the LAN again before it binds
  t0=$(date +%s); since=$(date -u +%FT%TZ)
  log "RESTORE start: RTX used=$(rtx_used)MiB GB300 used=$(gb_used)MiB /dev/shm/vllm_peer_tier2=[$(ls -la /dev/shm/vllm_peer_tier2 2>&1 | cut -c1-120)] (reused, same layout; never deleted)"
  docker start "$V27R" >/dev/null 2>&1 || log "WARN docker start $V27R rc=$?"
  for ((i=0;i<36;i++)); do docker logs --since "$since" "$V27R" > "$R/.restore-rtx.log" 2>&1
    grep -qE 'serving|prepared 40 layers' "$R/.restore-rtx.log" && break; sleep 5; done
  log "RESTORE sidecar $V27R running=$(docker inspect -f '{{.State.Running}}' "$V27R" 2>/dev/null) after $(( $(date +%s)-t0 ))s: $(grep -m1 -E 'serving|prepared 40 layers' "$R/.restore-rtx.log" | cut -c1-120)"
  docker start "$V27" >/dev/null 2>&1 || log "WARN docker start $V27 rc=$?"
  if wait_bind "$V27" 45 "$since" nofastfail "$V27R"; then
    log "LANE DARK ends ($V27 bound after $(( $(date +%s)-t0 ))s)"
    docker logs --since "$since" "$V27" > "$R/restore-v27-block.log" 2>&1
    docker inspect -f '{{json .Args}}' "$V27" > "$R/args-after-$V27.json"; docker inspect -f '{{json .Config.Env}}' "$V27" > "$R/env-after-$V27.json"
    log "RESTORE FACTS rowmap40=$(grep -cE 'MEGA_PEER rowmap .*: 40 layers' "$R/restore-v27-block.log") peer2=$(grep -c 'peer mode 2, check 0' "$R/restore-v27-block.log") kvline=[$(tr '\r' '\n' < "$R/restore-v27-block.log" | grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens')] kvdecl=$(grep -c 9126805504 "$R/args-after-$V27.json") kvtier=$(grep -c 'kv-offloading-size' "$R/args-after-$V27.json") block=$(grep -o rejection_sample_method "$R/args-after-$V27.json" | wc -l | tr -d ' ')(want 1) args_same=$(cmp -s "$R/args-before-$V27.json" "$R/args-after-$V27.json" && echo 1 || echo 0) env_same_sorted=$(envcmp "$R/env-before-$V27.json" "$R/env-after-$V27.json")"
    health restore "$V27R"; arm_prod_wd
    own "cardBR4 done; lane restored to $V27 pair + watchdog pid ${NEWWD:-NONE} (ppid $(ps -o ppid= -p "${NEWWD:-0}" 2>/dev/null | tr -d ' ')); ISOLATE removed (receipts $R)"
  else
    log "RESTORE FAILED — lane dark, needs hands (BRIEF-BR4.md §Restore)"
    own "OWNER: cardBR4 RESTORE FAILED — lane dark; see $R/runner.log (manual: docker start $V27R; wait 'serving'; docker start $V27; re-arm watchdog from $WARGV)"
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
    log "===== CARD BR4 END (ABORTED rc=$rc; lane = $(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ') watchdog=$(wd_pids)) ====="
  fi; }
trap on_exit EXIT
trap 'log "SIGNAL received"; exit 143' TERM INT HUP

# ---- GO + unfreeze + fixed settle: the v26seed production regime (seed map + adaptive unfrozen), identical on every arm
go_unfreeze(){ local since i gs uf tg tu
  since=$(date -u +%FT%TZ); setphase "$BOOT:go"; ph go
  arlines pre-go
  touch "$RUN/GO"
  for ((i=0;i<30;i++)); do grep -q '"baseline GO"' "$RUN/telemetry.jsonl" 2>/dev/null && break; sleep 2; done
  ctrl "$GO_CTRL" unfreeze
  for ((i=0;i<15;i++)); do grep -q '"ctrl freeze=0"' "$RUN/telemetry.jsonl" 2>/dev/null && break; sleep 2; done
  docker logs --since "$since" "$NAME" > "$A/go.log" 2>&1
  gs=$(tr '\r' '\n' < "$A/go.log" | grep -c 'ADAPT GO seen'); uf=$(tr '\r' '\n' < "$A/go.log" | grep -c 'ADAPT ctrl freeze=0')
  tg=$(grep -c '"baseline GO"' "$RUN/telemetry.jsonl" 2>/dev/null); tu=$(grep -c '"ctrl freeze=0"' "$RUN/telemetry.jsonl" 2>/dev/null)
  printf '{"go_seen_log": %s, "unfreeze_log": %s, "baseline_go_tel": %s, "unfreeze_tel": %s, "ctrl": %s, "settle_s": %s}\n' \
    "${gs:-0}" "${uf:-0}" "${tg:-0}" "${tu:-0}" "$(cat "$RUN/ctrl.json" 2>/dev/null || echo null)" "$ADAPT_SETTLE_S" > "$A/go.json"
  log "GO $BOOT: log 'GO seen'=$gs 'ctrl freeze=0'=$uf | telemetry 'baseline GO'=$tg 'ctrl freeze=0'=$tu | ctrl.json=$(cat "$RUN/ctrl.json" 2>/dev/null)"
  [[ ${gs:-0} -ge 1 && ${uf:-0} -ge 1 ]] || log "WARN $BOOT: GO/unfreeze not acknowledged by the hook — arm NOT in the production regime (G0 will FAIL; cells continue for receipts)"
  setphase "$BOOT:adapt-settle"; log "ADAPT SETTLE $BOOT: ${ADAPT_SETTLE_S}s (unfrozen, no traffic)"; sleep "$ADAPT_SETTLE_S"
  arlines settled; }

boot_br4(){ # arm name spec_extra
  local arm=$1 name=$2 spec=$3 since t0 ma want u g
  BOOT=$arm; NAME=$name; A=$R/$arm; RUN=$A/ar; mkdir -p "$A/metrics" "$RUN"
  check_abort
  setphase "$arm:preboot"
  wait_quiet "boot-$arm" || log "WARN: host not quiet before boot $arm — proceeding (lane already dark)"
  stop_all; drop_caches; strays
  settle "pre-$arm" || return 1
  ma=$(memavail); if awk -v m="$ma" -v t="$MEM_MIN" 'BEGIN{exit !(m < t)}'; then log "ABORT boot $arm: MemAvailable ${ma} GiB < $MEM_MIN after stop+drop_caches (stray?)"; return 1; fi
  u=$(rtx_used); [[ -n $u && $u -lt $RTX_IDLE_MIB ]] || { log "REFUSE boot $arm: RTX busy (${u:-?} MiB >= $RTX_IDLE_MIB) — BR4 arms are Grace-only"; return 1; }
  g=$(gb_used); [[ -n $g && $g -lt $GB_IDLE_MIB ]] || { log "REFUSE boot $arm: GB300 busy (${g:-?} MiB >= $GB_IDLE_MIB)"; return 1; }
  docker rm -f "$name" >/dev/null 2>&1 || true   # preflight guaranteed none pre-existed; clears only this run's half-made container
  rm -f "$RUN/GO" "$RUN/ctrl.json" "$RUN/ctrl.tmp" "$RUN/telemetry.jsonl" "$RUN/swaps.jsonl" "$RUN/steps.bin"   # this run's own dir only
  t0=$(date +%s); since=$(date -u +%FT%TZ); echo "$since" > "$A/boot_since.txt"
  setphase "$arm:boot"; ph boot_start "\"name\": \"$name\", \"spec_extra\": \"${spec//\"/\\\"}\""
  log "BOOT $arm: $name HOOK=$HOOK ROWMAP=$ROWMAP RUN=$RUN SPEC_EXTRA=[$spec] MemAvail=${ma}GiB RTX=${u}MiB GB300=${g}MiB"
  bg env NAME="$name" RUN="$RUN" HOOK="$HOOK" ROWMAP="$ROWMAP" PIN_STEP_RING=65536 PIN_SWAP_CUDA_EVENTS=1 SPEC_EXTRA="$spec" bash "$LAUNCH" >> "$R/launch-$arm.txt" 2>&1 \
    || { log "LAUNCH FAILED $arm: $(tail -2 "$R/launch-$arm.txt" | tr '\n' '|')"; return 1; }
  log "SERVER $arm launched: $(tail -1 "$R/launch-$arm.txt")"
  docker inspect -f '{{json .Args}}' "$name" > "$A/args.json" 2>&1; docker inspect -f '{{json .Config.Env}}' "$name" > "$A/env.json" 2>&1
  docker inspect -f '{{json .HostConfig.Binds}}' "$name" > "$A/binds.json" 2>&1; docker inspect -f '{{.Image}}' "$name" > "$A/image.txt" 2>&1
  if ! wait_bind "$name" "$BOOT_MIN" "$since"; then docker logs "$name" > "$A/boot-FAIL.log" 2>&1; return 1; fi
  log "boot->ready $arm $(( $(date +%s)-t0 ))s ($name)"
  echo "$(date +%s) $arm:boot $(memavail)" >> "$R/mem.tsv"   # guarantees >=1 boot sample even on a fast boot
  docker logs --since "$since" "$name" > "$A/boot.log" 2>&1
  tr '\r' '\n' < "$A/boot.log" | grep -E 'KV cache size|Maximum concurrency|speculative_config=' | cut -c1-400 > "$A/boot-facts.log"
  tr '\r' '\n' < "$A/boot.log" | grep -E 'Autotuning process (starts|ends)|Loaded [0-9]+ configs|Saved [0-9]+ configs' | cut -c1-300 > "$A/boot-autotune.log"
  tr '\r' '\n' < "$A/boot.log" | grep -iE 'block.{0,40}(verif|rejection)|(verif|rejection).{0,40}block' | cut -c1-300 > "$A/boot-block.log"
  tr '\r' '\n' < "$A/boot.log" | grep -iE 'probabilistic|draft_sample|draft logits' | cut -c1-300 > "$A/boot-prob.log"
  tr '\r' '\n' < "$A/boot.log" | grep -E 'PIN_HOT|pin-hot|hook FAILED' | cut -c1-300 > "$A/boot-pin.log"
  local kv blk prb bl pl ffb ld sv as ae rl rd l9 pab hf km k ae_ref ee_ref be_ref im ru wantp
  kv=$(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$A/boot-facts.log" | grep -oE '[0-9,]+' | head -1)
  blk=$(grep -o 'rejection_sample_method' "$A/args.json" | wc -l | tr -d ' ')
  prb=$(grep -o 'draft_sample_method' "$A/args.json" | wc -l | tr -d ' ')
  bl=$(grep -c . "$A/boot-block.log"); pl=$(grep -c . "$A/boot-prob.log"); ffb=$(tr '\r' '\n' < "$A/boot.log" | grep -cE "$FF")
  ld=$(grep -cE 'Loaded [0-9]+ configs' "$A/boot-autotune.log"); sv=$(grep -cE 'Saved [0-9]+ configs' "$A/boot-autotune.log")
  as=$(grep -c 'Autotuning process starts' "$A/boot-autotune.log"); ae=$(grep -c 'Autotuning process ends' "$A/boot-autotune.log")
  # hook guard lines (v26ar2 pin_hot_experts_hook.py _budget_ok: 'PIN_HOT abort <where>: MemAvailable ... < 10 GiB' raises)
  rl=$(grep -c 'PIN_HOT rehome layer=' "$A/boot-pin.log"); rd=$(grep -c 'PIN_HOT rehome done layers=40' "$A/boot-pin.log")
  l9=$(grep -m1 'PIN_HOT rehome layer=9 ' "$A/boot-pin.log" | grep -oE 'host_avail=[0-9.]+' | cut -d= -f2)
  pab=$(grep -c 'PIN_HOT abort' "$A/boot-pin.log"); hf=$(grep -c 'hook FAILED' "$A/boot-pin.log")
  km=""; for k in "atomic counter ready" "atomic mode ready" "swap barrier" "v26ar.2 telemetry step_ring=65536" "cuda_events=1" "rowmap=/w/$ROWMAP" "freeze=True"; do
    grep -qF "$k" "$A/boot-pin.log" || km="$km${km:+;}$k"; done
  ae_ref=$(cmp -s "$A/args.json" "$R/args-ref-$REF.json" && echo 1 || echo 0); ee_ref=$(envcmp "$A/env.json" "$R/env-ref-$REF.json")
  be_ref=$(python3 -c 'import json,sys
r=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); run=sys.argv[3]
n=lambda L: sorted(("<AR>:/ar" if b.endswith(":/ar") else b) for b in (L or []))
print(int(n(r) == n(a) and (run + ":/ar") in (a or [])))' "$R/binds-ref-$REF.json" "$A/binds.json" "$RUN" 2>/dev/null || echo 0)
  im=$(cat "$A/image.txt" 2>/dev/null); ru=$(rtx_used)
  python3 -c 'import json,sys
d={}
for a in sys.argv[2:]:
    k,v=a.split("=",1); d[k]=int(v) if v.lstrip("-").isdigit() else v
json.dump(d, open(sys.argv[1],"w"), indent=1)' "$A/facts.json" arm="$arm" name="$name" kv_tokens="${kv:-}" kv_ref="$(cat "$R/kv-ref.txt" 2>/dev/null)" \
    block_in_args="${blk:-0}" prob_in_args="${prb:-0}" block_boot_lines="${bl:-0}" prob_boot_lines="${pl:-0}" spec_extra="$spec" ff_boot="${ffb:-0}" loaded_lines="${ld:-0}" saved_lines="${sv:-0}" \
    autotune_starts="${as:-0}" autotune_ends="${ae:-0}" rehome_layers="${rl:-0}" rehome_done="${rd:-0}" host_avail_l9_gib="${l9:-}" \
    pin_abort="${pab:-0}" hook_failed="${hf:-0}" hook_keys_missing="$km" args_eq_ref="$ae_ref" env_eq_ref_sorted="$ee_ref" \
    binds_eq_ref_except_ar="$be_ref" image="$im" rtx_used_after_bind_mib="${ru:-}" boot_s="$(( $(date +%s)-t0 ))" bound_at="$(date +%s)"
  case $arm in A|A2) wantp=0; want=0;; P) wantp=1; want=0;; PB) wantp=1; want=1;; *) wantp=x; want=x;; esac   # prob 0/1/1/0, block 0/0/1/0
  log "FACTS $arm: kv=[${kv:-MISSING}] (v26seed log: $(cat "$R/kv-ref.txt" 2>/dev/null)) prob_in_args=$prb (want $wantp) block_in_args=$blk (want $want) prob_boot_lines=$pl block_boot_lines=$bl ff_boot=$ffb autotune loaded=$ld saved=$sv starts=$as ends=$ae | rehome layers=$rl done40=$rd host_avail@L9=${l9:-?}GiB pin_abort=$pab hook_failed=$hf keys_missing=[${km}] | args_eq_v26seed=$ae_ref env_eq_v26seed(sorted)=$ee_ref binds_eq_v26seed(except /ar)=$be_ref image=${im:0:19} rtx_after_bind=${ru}MiB"
  [[ ${blk:-0} == "$want" ]] || log "WARN $arm: block_in_args=$blk, expected $want (G0 will FAIL; cells continue for receipts)"
  [[ ${prb:-0} == "$wantp" ]] || log "WARN $arm: prob_in_args=$prb, expected $wantp (G0 will FAIL; cells continue for receipts)"
  if [[ $arm != A ]]; then local kva; kva=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("kv_tokens",""))' "$R/A/facts.json" 2>/dev/null)
    [[ -n $kv && $kv == "$kva" ]] || log "WARN $arm: GPU KV cache size [${kv:-MISSING}] != arm A [${kva:-MISSING}] (G0 rule: A2 == A exactly; P == PB exactly; P/PB may sit <= 1.5 % below A — the draft-logits cache is allocated before KV profiling)"; fi
  [[ $rd == 1 && $pab == 0 && $hf == 0 && -z $km ]] || log "WARN $arm: hook wiring incomplete (rehome_done=$rd abort=$pab failed=$hf missing=[$km]) — G0 will FAIL"
  health "$arm"
  go_unfreeze
  foreign_log bind "$since"
  ph boot_ready "\"name\": \"$name\""
  return 0; }

alive(){ is_up "$NAME" && return 0; log "ARM $BOOT DEAD ($NAME up=n) — skipping remaining cells"; return 1; }
metrics(){ curl -s -m 10 $U/metrics 2>/dev/null | grep -E '^vllm:spec_decode_' > "$A/metrics/$1.txt"; }   # spec-decode counters at a cell boundary
cell(){ # tag cmd... -> runs with bg, logs one line; /metrics spec-decode + /ar line counts at start and end
  local tag=$1; shift; check_abort; alive || return 1
  ph cell_start "\"cell\": \"$tag\""; metrics "$tag-start"; arlines "$tag-start"
  bg "$@" > "$A/$tag.log" 2>&1; local rc=$?
  metrics "$tag-end"; arlines "$tag-end"; ph cell_end "\"cell\": \"$tag\", \"rc\": $rc"
  log "CELL $BOOT $tag rc=$rc: $(tail -1 "$A/$tag.log" | cut -c1-330)"; return 0; }
run_cells(){ local arm=$1 sinceC r ta
  sinceC=$(date -u +%FT%TZ); ta=$(date +%s); setphase "$arm:cells"; ph cells_start
  wait_foreign_quiet "$(cat "$A/boot_since.txt" 2>/dev/null || echo "$sinceC")"
  # (1) PRIMARY: per-class window at temperature 1.0 (probabilistic drafting and block verification are no-ops on greedy rows)
  cell pct1 bash -c 'cd "$1" && ARM="$2" OUTDIR="$3" TEMP="$4" SEED="$5" TOP_P="$6" DEC="$7" exec timeout "$8" bash "$9"' _ \
    "$BENCH" "BR4-$arm-t1" "$A/perclass-t1" "$TEMP_T" "$SEED_T" "$TOP_P_T" "$PCDEC_T" "$T_PERCLASS" "$PERCLASS_T"
  # (2) parity: the STOCK temp-0 window (perclass_run.sh has no r1-only switch -> full r1+r2)
  cell pct0 bash -c 'cd "$1" && ARM="$2" OUTDIR="$3" exec timeout "$4" bash perclass_run.sh' _ "$BENCH" "BR4-$arm-t0" "$A/perclass-t0" "$T_PERCLASS"
  # (3) mixed C1 x3 at temperature 1.0 (ar_mixed_v2 call shape + sampling flags)
  for r in $(seq 1 "$C1_REPS"); do
    cell "c1-r$r" timeout "$T_MIXED" env HELDOUT=1 python3 "$MIXED_T" "BR4-$arm-c1-r$r" "$A" 1 "$MIX_S" --temperature "$TEMP_T" --seed "$SEED_T" --top-p "$TOP_P_T" || break
  done
  # (4) teacher-forced TF capture (prefill only; temperature-independent) -> $A/tf-BR4-<arm>.jsonl
  cell tf bash -c 'cd "$1" && MAXPOS=4096 CONC=4 exec timeout "$2" python3 "$3" capture "$4" "$5"' _ "$A" "$T_TF" "$TF" "BR4-$arm" "$CORPUS"
  # (5) gauntlet (temp 0, seed 42, effort low: run_suite.py request_settings)
  check_abort; alive && { ph cell_start '"cell": "gauntlet"'; metrics gauntlet-start; arlines gauntlet-start; gauntlet "$arm"; metrics gauntlet-end; arlines gauntlet-end; ph cell_end '"cell": "gauntlet"'; }
  # (6) per-turn tool-call emission (replay_turns.py: temp 0, thinking off; writes $TURNS_OUT/turns-<TAG>.json)
  cell turns env TAG="BR4-$arm" PASSES=2 timeout "$T_TURNS" python3 "$TURNS"
  [[ -s $TURNS_OUT/turns-BR4-$arm.json ]] && mv "$TURNS_OUT/turns-BR4-$arm.json" "$A/" && log "TURNS $arm: moved turns-BR4-$arm.json into $A"
  # (7) post
  setphase "$arm:post"; ph cells_end
  errcount "$arm" "$NAME" "$sinceC"
  foreign_log cells "$sinceC"
  log "ARM $arm cells took $(( $(date +%s)-ta ))s; /ar lines: $(tail -1 "$A/ar-lines.tsv" 2>/dev/null | tr '\t' ' ')"; }

# ================= preflight (refusals exit 4/5/6 with the lane untouched) =================
log "===== CARD BR4 START (arms=$ARMS spec_P=[$SPEC_P] spec_PB=[$SPEC_PB] temp_t=$TEMP_T seed_t=$SEED_T mix_s=$MIX_S c1_reps=$C1_REPS settle_s=$SETTLE_S adapt_settle_s=$ADAPT_SETTLE_S isolate=$ISOLATE mem_min=$MEM_MIN boot_min=$BOOT_MIN R=$R) ====="
if [[ -n $WAIT_PID ]]; then log "WAIT for pid $WAIT_PID to exit"; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; log "pid $WAIT_PID gone"; fi
log "LANE-OWNER.txt (tail): $(tail -3 "$OWNER" 2>/dev/null | tr '\n' ' ' | cut -c1-500)"
wo=$(wd_orphan_parent); [[ -n $wo ]] && log "NOTE: ignoring pid $wo in the runner check: orphaned bash (PPID 1) whose only child is the production watchdog ($(ps -o args= -p "$wo" | cut -c1-100))"
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
x=$(not_known); [[ -n ${x// /} ]] && { log "REFUSE: running container(s) outside the production pair {$V27, $V27R}: $x"; exit 4; }
is_up "$V27" && is_up "$V27R" || { log "ABORT: $V27 pair not running — lane untouched"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: :30006 not bound — lane untouched"; exit 5; }
exists "$REF" || { log "ABORT: control ground-truth container $REF missing"; exit 5; }
is_up "$REF" && { log "ABORT: $REF is RUNNING (expected stopped-and-kept) — lane untouched"; exit 5; }
[[ "$(docker inspect -f '{{.Image}}' "$REF" 2>/dev/null)" == "$IMAGE_ID" ]] || { log "ABORT: $REF does not run $IMAGE ($IMAGE_ID)"; exit 5; }
for n in $ALLN; do exists "$n" && { log "ABORT: BR4 container already exists: $n"; exit 5; }; exists "$n-rtx" && { log "ABORT: unexpected $n-rtx exists"; exit 5; }; done
ls "$TURNS_OUT"/turns-BR4-* >/dev/null 2>&1 && { log "ABORT: $TURNS_OUT/turns-BR4-* already exist (tag clash)"; exit 5; }
sudo -n iptables -S INPUT 2>/dev/null | grep -q "$FW_TAG" && { log "ABORT: a $FW_TAG firewall rule is already present (stale run?) — lane untouched"; exit 5; }
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null)" == "$IMAGE_ID" ]] || { log "ABORT: $IMAGE missing or not $IMAGE_ID"; exit 5; }
docker image inspect "$GIMG" >/dev/null 2>&1 || { log "ABORT: gauntlet sandbox image missing"; exit 5; }
[[ -f $MODEL/config.json ]] || { log "ABORT: model missing $MODEL"; exit 5; }
[[ "$(docker inspect -f '{{.Image}}' "$V27" 2>/dev/null)" == "$PROD_IMAGE_ID" ]] || { log "ABORT: $V27 is not on $PROD_IMAGE_ID — restore target is not the expected v27"; exit 5; }
chk(){ [[ -s $1 ]] || { log "ABORT: missing $1"; exit 5; }; [[ -z ${2:-} || "$(sha1sum "$1" | cut -c1-8)" == "$2" ]] || { log "ABORT: sha1 $(sha1sum "$1" | cut -c1-12) != $2 for $1"; exit 5; }; }
chk "$LAUNCH" "$SHA_LAUNCH"; chk "$VERDICT" "$SHA_VERDICT"; chk "$PERCLASS_T" "$SHA_PERCLASS_T"; chk "$PCDEC_T" "$SHA_PCDEC_T"; chk "$MIXED_T" "$SHA_MIXED_T"
chk "$PERCLASS" "$SHA_PERCLASS"; chk "$PCDEC" "$SHA_PCDEC"; chk "$MIXED" "$SHA_MIXED"; chk "$WD" "$SHA_WD"; chk "$TF" "$SHA_TF"; chk "$TURNS" "$SHA_TURNS"
chk "$HOOK/$ROWMAP" "$SHA_ROWMAP"; chk "$HOOK/sitecustomize.py" "$SHA_SITE"; chk "$HOOK/pin_hot_experts_hook.py" "$SHA_HOOKPY"; chk "$HOOK/e4_adaptive.py" "$SHA_E4"
chk "$LAUNCH_REF" "$SHA_LREF"; chk "$PCTAB" "$SHA_PCTAB"; chk "$X3" "$SHA_X3"; chk "$PCPROMPTS" "$SHA_PCPROMPTS"; chk "$FLIPS" "$SHA_FLIPS"
chk "$CORPUS" "$SHA_CORPUS"; chk "$FIXTURE" "$SHA_FIXTURE"; chk "$GDIR/run_suite.py" "$SHA_SUITE"
# launcher drift: the clone may differ from launch_ar6.sh ONLY by comments, the SPEC_EXTRA definition and the one spliced line
REF_SPEC_LINE=$(cat <<'EOF'
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,8,5],[9,24,1]]}' \
EOF
)
NEW_SPEC_LINE=$(cat <<'EOF'
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":[[1,8,5],[9,24,1]]$SPEC_EXTRA}" \
EOF
)
normlaunch(){ grep -v '^#' "$1" | grep -vxF 'SPEC_EXTRA=${SPEC_EXTRA:-}' | SPECL="$2" awk '$0==ENVIRON["SPECL"]{print "@@SPEC@@"; next} {print}'; }
diff "$LAUNCH_REF" "$LAUNCH" > "$R/DIFF-launch-preflight.txt"
h=$(grep -cE '^[0-9]' "$R/DIFF-launch-preflight.txt")
dd=$(diff <(normlaunch "$LAUNCH_REF" "$REF_SPEC_LINE") <(normlaunch "$LAUNCH" "$NEW_SPEC_LINE") | grep -cE '^[<>]')
sd=$(grep -cxF 'SPEC_EXTRA=${SPEC_EXTRA:-}' "$LAUNCH"); rs=$(grep -cxF -- "$REF_SPEC_LINE" "$LAUNCH_REF"); ns=$(grep -cxF -- "$NEW_SPEC_LINE" "$LAUNCH")
[[ $h == 3 && $dd == 0 && $sd == 1 && $rs == 1 && $ns == 1 ]] || { log "ABORT: $LAUNCH drifted from $LAUNCH_REF (hunks=$h non-SPEC changed lines=$dd SPEC_EXTRA defs=$sd ref spec lines=$rs new spec lines=$ns; want 3/0/1/1/1) — see DIFF-launch-preflight.txt"; exit 5; }
grep -q 'DEC=' "$PERCLASS_T" && grep -q -- '--temperature "\$TEMP"' "$PERCLASS_T" || { log "ABORT: $PERCLASS_T does not pass --temperature \$TEMP to DEC"; exit 5; }
u=$(rtx_used)
{ for f in "$0" "$LAUNCH" "$LAUNCH_REF" "$VERDICT" "$PERCLASS_T" "$PCDEC_T" "$MIXED_T" "$PERCLASS" "$PCDEC" "$PCTAB" "$X3" "$PCPROMPTS" "$MIXED" "$WD" "$TF" "$FLIPS" "$CORPUS" \
      "$TURNS" "$FIXTURE" "$GDIR/run_suite.py" "$HOOK/$ROWMAP" "$HOOK/sitecustomize.py" "$HOOK/pin_hot_experts_hook.py" "$HOOK/e4_adaptive.py"; do echo "sha1 $(sha1sum "$f" | cut -c1-12) $f"; done
  echo "arm image $IMAGE ($IMAGE_ID)"; echo "production image $PROD_IMAGE_ID"; echo "gauntlet image $GIMG"; echo "P SPEC_EXTRA=[$SPEC_P]"; echo "PB SPEC_EXTRA=[$SPEC_PB]"
  echo "GO ctrl $GO_CTRL; adapt settle ${ADAPT_SETTLE_S}s"
  echo "sampling for pct1/mixed: temperature=$TEMP_T seed=$SEED_T (+row/prompt index) top_p=$TOP_P_T"
  echo "iptables :30006 rules before: $(sudo -n iptables -S INPUT 2>/dev/null | grep 30006 | tr '\n' '|')"; } > "$R/instruments.txt"
docker ps -a > "$R/preflight-docker-ps-a.txt" 2>&1; nvidia-smi > "$R/preflight-nvidia-smi.txt" 2>&1; free -g > "$R/preflight-free-g.txt" 2>&1
df -h /dev/shm >> "$R/preflight-free-g.txt" 2>&1
shm_big > "$R/preflight-shm-big.txt"; host_capture "$R/preflight-host.txt" preflight
# control ground truth (read-only inspect of the stopped v26seed container)
docker inspect -f '{{json .Args}}' "$REF" > "$R/args-ref-$REF.json"; docker inspect -f '{{json .Config.Env}}' "$REF" > "$R/env-ref-$REF.json"
docker inspect -f '{{json .HostConfig.Binds}}' "$REF" > "$R/binds-ref-$REF.json"
docker logs "$REF" 2>&1 | tr '\r' '\n' | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1 | grep -oE '[0-9,]+' > "$R/kv-ref.txt"
grep -q '"--offload-backend","uva"' "$R/args-ref-$REF.json" && grep -qF '[[1,8,5],[9,24,1]]' "$R/args-ref-$REF.json" && grep -q '"--max-num-seqs","24"' "$R/args-ref-$REF.json" \
  && ! grep -q rejection_sample_method "$R/args-ref-$REF.json" && ! grep -q draft_sample_method "$R/args-ref-$REF.json" \
  && grep -q '"PIN_ADAPTIVE_FREEZE=1"' "$R/env-ref-$REF.json" && grep -qF "\"PIN_ROWMAP=/w/$ROWMAP\"" "$R/env-ref-$REF.json" && grep -qF "\"$HOOK:/w:ro\"" "$R/binds-ref-$REF.json" \
  || { log "ABORT: $REF args/env/binds are not the expected v26seed recipe (uva, k [[1,8,5],[9,24,1]], seqs 24, greedy draft, no block, freeze=1, $ROWMAP, /w=$HOOK)"; exit 5; }
# rendered spec element (launcher line evaluated with SPEC_EXTRA) vs the v26seed Args element
render_spec(){ SPEC_EXTRA=$1 bash -c 'l=$(grep -E -- "^ +--speculative-config " "$1"); l=${l#*--speculative-config }; l=${l% \\}; eval "set -- $l"; printf "%s" "$1"' _ "$LAUNCH"; }
se=$(spec_el "$R/args-ref-$REF.json"); r0=$(render_spec ""); rp=$(render_spec "$SPEC_P"); rpb=$(render_spec "$SPEC_PB")
[[ -n $se && $r0 == "$se" && $rp == "${se%\}}$SPEC_P}" && $rpb == "${se%\}}$SPEC_PB}" ]] \
  || { log "ABORT: launch_br3.sh renders spec [$r0] / P [$rp] / PB [$rpb]; v26seed has [$se]"; exit 5; }
python3 -c 'import json,sys
se,r0,rp,rpb=(json.loads(x) for x in sys.argv[1:5])
ok = r0 == se and rp == dict(se, draft_sample_method="probabilistic") and rpb == dict(se, draft_sample_method="probabilistic", rejection_sample_method="block")
sys.exit(0 if ok else 1)' "$se" "$r0" "$rp" "$rpb" || { log "ABORT: rendered P/PB spec JSON is not v26seed + the expected keys (json-level)"; exit 5; }
log "spec render OK: A/A2 == v26seed [$se]; P == v26seed + probabilistic [$rp]; PB == v26seed + probabilistic + block [$rpb]"
docker inspect -f '{{json .Args}}' "$V27" > "$R/args-before-$V27.json"; docker inspect -f '{{json .Config.Env}}' "$V27" > "$R/env-before-$V27.json"
docker inspect -f '{{json .Args}}' "$V27R" > "$R/args-before-$V27R.json"
grep -q 9126805504 "$R/args-before-$V27.json" && grep -q 'kv-offloading-size' "$R/args-before-$V27.json" && grep -qF '\"rejection_sample_method\":\"block\"' "$R/args-before-$V27.json" \
  || { log "ABORT: production args are not the expected v27-block recipe (kv decl + tier + rejection_sample_method block)"; exit 5; }
WP=$(wd_pids); WP=${WP%% *}
[[ -n $WP ]] || { log "ABORT: no live sidecar_watchdog.py --server $V27 — refusing (cannot mirror its restore); lane untouched"; exit 5; }
tr '\0' '\n' < "/proc/$WP/cmdline" | sed '/^$/d' > "$WARGV"; readlink "/proc/$WP/cwd" > "$WCWD"
grep -qx -- "--server" "$WARGV" && grep -qx -- "$V27" "$WARGV" || { log "ABORT: captured watchdog argv does not name $V27: $(tr '\n' ' ' < "$WARGV")"; exit 5; }
log "watchdog pid $WP (ppid $(ps -o ppid= -p "$WP" | tr -d ' ')) argv: $(tr '\n' ' ' < "$WARGV") cwd=$(cat "$WCWD")"
OOM0=$(oom_count); echo "${OOM0:-}" > "$R/oom-start.txt"
log "preflight OK: MemAvailable=$(memavail)GiB RTX=${u}MiB (production -rtx up by design; checked again after the pair stops) kv_ref=[$(cat "$R/kv-ref.txt")] oom_lines=${OOM0:-unreadable}"
log "lane before: $(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')"
own "cardBR4 runner pid $$ preflight OK — about to take :30006 dark (A=$N_A, P=$N_P, PB=$N_PB, A2=$N_A2; GB300-only, ~3.2 h); restores $V27 pair + watchdog at END. Receipts $R"
wait_quiet preflight || { log "ABORT: host never quiet; lane untouched"; exit 6; }
check_abort

# ================= lane dark: watchdog off (log moved aside), v27-block pair stopped-and-kept =================
LANE_TOUCHED=1
echo "cardBR4 $(date '+%F %T %Z')" > "$D/logs/MAINTENANCE"
setphase pre; mem_sampler & SAMPLER=$!
for p in $(wd_pids); do kill "$p" 2>/dev/null && log "fail-stop watchdog $p stopped (cardBR4)"; done
for i in 1 2 3 4 5; do [[ -z $(wd_pids) ]] && break; sleep 1; done
[[ -n $(wd_pids) ]] && { log "ABORT: sidecar watchdog still alive ($(wd_pids)) — refusing to stop $V27 under a fail-stop watchdog"; exit 6; }
echo "$(date '+%F %T %Z') watchdog stopped by cardBR4 (re-armed after restore)" >> "$WDLOG"
mv "$WDLOG" "$WDLOG.pre-cardBR4-$(date +%Y%m%d-%H%M%S)"
BOOT=pre; ph lane_dark
log "LANE DARK begins (stopping $V27 pair, kept)"
own "cardBR4: lane :30006 DARK (watchdog $WP stopped, $V27 + $V27R stopped-and-kept)"
docker stop -t 60 "$V27" "$V27R" >/dev/null 2>&1; log "stop-and-keep $V27 $V27R"
isolate
# Grace-only gate: with the production pair down, the RTX must be idle before the first boot (else refuse -> EXIT trap restores)
for i in $(seq 1 12); do u=$(rtx_used); [[ -n $u && $u -lt $RTX_IDLE_MIB ]] && break; sleep 5; done
if [[ -z $u || $u -ge $RTX_IDLE_MIB ]]; then
  log "REFUSE: RTX still holds ${u:-?} MiB (>= $RTX_IDLE_MIB) 60 s after the production pair stopped — BR4 is Grace-only; restoring production"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/rtx-busy.txt" 2>&1; exit 6; fi
log "RTX idle after pair stop: ${u} MiB"

# ================= arms: A (control) -> P (probabilistic draft) -> PB (probabilistic + block) -> A2 (control) =================
for arm in $ARMS; do
  case $arm in
    A) n=$N_A; sx="";;
    P) n=$N_P; sx="$SPEC_P";;
    PB) n=$N_PB; sx="$SPEC_PB";;
    A2) n=$N_A2; sx="";;
    *) log "BUG unknown arm $arm"; continue;;
  esac
  if boot_br4 "$arm" "$n" "$sx"; then
    run_cells "$arm"
  else
    log "ARM $arm FAILED TO BOOT — its cells are MISSING (verdict: candidate P/PB -> FAIL, control -> INVALID)"
    [[ -d $A ]] && exists "$n" && foreign_log boot "$(cat "$A/boot_since.txt" 2>/dev/null || date -u +%FT%TZ)"
  fi
  stop_arm "$n"
  check_abort
done

# ================= END: restore, TF compares (CPU), verdict, renames, ledger =================
restore_lane
setphase end; kill "$SAMPLER" 2>/dev/null
OOM1=$(oom_count); echo "${OOM1:-}" > "$R/oom-end.txt"
# tf_logprob.py compare takes FILENAMES (tf-<tag>.jsonl), never tags; outputs tf-compare-<ref>-vs-<cand>.json in the cwd
# (P PB = annotation only: probabilistic vs probabilistic+block)
for pr in "A P" "A2 P" "A PB" "A2 PB" "A A2" "P PB"; do set -- $pr
  if [[ -s $R/$1/tf-BR4-$1.jsonl && -s $R/$2/tf-BR4-$2.jsonl ]]; then
    ( cd "$R" && timeout 600 python3 "$TF" compare "$1/tf-BR4-$1.jsonl" "$2/tf-BR4-$2.jsonl" ) > "$R/tfcmp-$2-vs-$1.log" 2>&1; log "TFCMP $2 vs $1: $(tail -1 "$R/tfcmp-$2-vs-$1.log" | cut -c1-240)"
  else log "TFCMP $2 vs $1: UNMEASURED (missing tf-BR4-$1.jsonl or tf-BR4-$2.jsonl)"; fi; done
# margin-weighted flips (flips_margin_p2.py REF CAND --label --calib); each candidate vs A calibrated on A2 (and vice versa); floor A2 vs A
fm(){ local ref=$1 cand=$2 cal=${3:-} out=$R/flipsm-$2-vs-$1.txt
  [[ -s $R/$ref/tf-BR4-$ref.jsonl && -s $R/$cand/tf-BR4-$cand.jsonl ]] || { log "FLIPSM $cand vs $ref: UNMEASURED (missing capture)"; return 0; }
  ( cd "$R" && timeout 600 python3 "$FLIPS" "$ref/tf-BR4-$ref.jsonl" "$cand/tf-BR4-$cand.jsonl" --label "$cand-vs-$ref" ${cal:+--calib "$cal/tf-BR4-$cal.jsonl"} ) > "$out" 2>&1
  log "FLIPSM $cand vs $ref: $(tail -1 "$out" | cut -c1-200)"; }
fm A P A2; fm A2 P A; fm A PB A2; fm A2 PB A; fm A A2
python3 "$VERDICT" "$R" > "$R/VERDICT.txt" 2> "$R/verdict.err"; vrc=$?
SUF=$(python3 -c "import json;print(json.load(open('$R/verdict.json'))['suffix'])" 2>/dev/null); SUF=${SUF:-BR4-NOVERDICT}
VL=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["result"],"|",d["suffix"],"|",json.dumps(dict((k,v[0]) for k,v in d["gates"].items())))' "$R/verdict.json" 2>&1 | tail -1 | cut -c1-900)
log "VERDICT rc=$vrc $VL"
for n in $ALLN; do exists "$n" && docker rename "$n" "$n-$SUF" 2>/dev/null && log "kept $n-$SUF (stopped)"; done
grep -hE ' (=====|CELL|BOOT|SERVER|FACTS|GO|CTRL|ADAPT|HEALTH|GAUNTLET|TURNS|TFCMP|FLIPSM|boot->ready|BOUND|FASTFAIL|DIED|TIMEOUT|ERRCOUNT|FOREIGN|ISOLATE|SETTLE|HOST|HOLD|RESTORE|LANE|RTX|VERDICT|MEM_GUARD|WARN|ABORT|REFUSE|ARM|kept|stop-and-keep|fail-stop|re-arm|watchdog|spec) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card BR4: DSpark probabilistic drafting (P) and probabilistic + block verification (PB) vs greedy/sequential, A/P/PB/A2 on the GB300-only v26seed (Seeded Adaptive) recipe, :30006 — $(python3 -c "import json;d=json.load(open('$R/verdict.json'));print(d['result'], d['suffix'])" 2>/dev/null)"
  cat "$R/VERDICT.txt" 2>/dev/null; echo "Receipts: $R. Production restored to $V27 pair + watchdog ${NEWWD:-NONE}. NOT promoted — James decides."; } >> "$LEDGER"
FINISHED=1
log "FINAL dsv41 running=[$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')] bound=$(curl -s -m 3 $U/v1/models | grep -c $M) watchdog=${NEWWD:-NONE} ppid=$(ps -o ppid= -p "${NEWWD:-0}" 2>/dev/null | tr -d ' ') isolate_rule=$(sudo -n iptables -S INPUT 2>/dev/null | grep -c "$FW_TAG")"
log "===== CARD BR4 END (lane = $V27) ====="

#!/usr/bin/env bash
# cardP2f_runner.sh — Card P2f: PROMOTION-GATE measurement of config C (vLLM nightly-39983a5e + pin-hot hook, FLASHMLA_MEGA_ATTN_DSV41,
#   --no-swa-bounded-replay, off60, fp8_ds_mla, alloc env, filled FI store) vs v23, same window, bracketed A -> C -> D.
#   Armed 2026-10-18 ~19:20 CDT by Milo (hermes milo) subagent claude-opus-5-5. Template: cardP2e_runner.sh (same helpers).
# ARMS (one launch, this order):
#   A = live v23 REF (dsv41-vllm-1017-v23-pinexact-BOUND-REF), no reboot. Per-class pre-control = the 18:29 baseline
#       ~/dsv41/1017/results-perclass-baseline (NOT rerun).
#   C = docker start dsv41-vllm-1017-P2b-n39983a5e-noreplay-EXP-FAIL (store-loaded third boot of config C; bind budget 30 min).
#   D = docker start v23 REF and LEFT SERVING (post-control).
# CELLS
#   A: GSM8K-200 (lil-bench v0.7.7 c71ec1f2, --test-profile gsm8k --profile-runs 200, C30, temp 0 — the Card D method, run from the
#      box at 127.0.0.1 instead of the Mac; compare-baseline = Card D 1017 json) ; BFCL dev (simple_python+multiple, 600) ;
#      BFCL heldout (live_simple+live_multiple, 1,311; bfcl_gate.py says: run only at promotion = this card) ; RT C1x3 (TTFT 8K) ;
#      knee6 x2 ; knee r1/r2 ; gauntlet ; cold 64K N=1
#   C: per-class (perclass_run.sh ARM=candC, CTL=v23 baseline) ; then the same cells as A in the same order (GSM8K compare-baseline = A)
#   D: per-class (perclass_run.sh ARM=v23post, no CTL) ; RT C1x3 ; knee6 x2 ; knee r1/r2 ; cold 64K N=1 ; gauntlet ;
#      then REPORT-ONLY fidelity (GSM8K vs A, BFCL dev + heldout) — lane is serving v23 again by then.
#   Then perclass_table.py candC vs v23post (second control) and v23post vs v23 (drift view).
# PRE-REGISTERED GATE (evaluated by cardP2f_verdict.py; fixed before launch):
#   GATE PASS iff ALL of:
#     (1) per-class decode wall-clock: every class x {C1, C8} candC mean >= bar against BOTH controls (v23 18:29 baseline AND v23post),
#         bar = ctl mean x (1 - max(2%, 1.5 x ctl same-config spread))  [20 comparisons]
#     (2) GSM8K-200 C >= A - 1.5 pt
#     (3) BFCL dev C >= A - 1 pt AND BFCL heldout C >= A - 1 pt
#     (4) gauntlet C 12/12
#     (5) RT C1 (kbench real-text decode_agg, mean of 3) C >= CTL - 2%, CTL = mean(A, D)
#     (6) knee6 C8: mean of 4 raw reps (2 knee6 runs x 2 reps; C8 is bimodal) C >= CTL - 2%, CTL = mean(A 4-rep mean, D 4-rep mean)
#   JUDGMENT_CALL iff (2)-(6) pass and every failing (1) comparison is a CODE cell with candC >= ctl mean - 5%
#     (reported with the acceptance-tax vs kernel-gain arithmetic). Anything else = FAIL. A missing cell is FAIL (never a number).
#   TRIGGER (reported, not gated): per-class /metrics acceptance moves > 2 pts vs either control.
#   Validity: rehome done layers=40, no "Decoder SWA bounded replay:" line, MegaAttn backend line, prompts sha1 618898aa.
# NO PROMOTION under any result. Stop-and-keep; the C container gets suffix -P2f-PASS / -P2f-FAIL / -P2f-JUDGMENT.
# Never :30003, never glmf-*/dsf*. Refuses foreign GPU containers / other runners. Does not touch run_window_T1T2.sh or
#   $HOME/pin-hot-experts/hook/.
set -uo pipefail
D=$HOME/dsv41/m3; W=$HOME/dsv41/1017; R=$W/results-cardP2f; mkdir -p "$R"
BK=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v23-pinexact-BOUND-REF
CN=dsv41-vllm-1017-P2b-n39983a5e-noreplay-EXP-FAIL
IMAGE=vllm/vllm-openai@sha256:a3bc7f09c1e2e72e068021feb75118c92e98c538706caa0c5d1acc0bb9252623
PCACHE=$HOME/dsv41/vllm-cache-p2-39983a5e
BENCH=$HOME/dsv41/bench; PROMPTS=$BENCH/perclass_prompts.jsonl
PC_BASE=$W/results-perclass-baseline; PC_C=$W/results-perclass-candC; PC_D=$W/results-perclass-v23post
LB=$W/lilbench/app; LBREF=$W/lilbench/cardD-t2-gsm8k200-1017.json
S=$HOME/pin-hot-experts/scripts; BFCL=$HOME/pin-hot-experts/bfcl_data
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
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh)' | grep -v "cardP2f_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
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
store_keys(){ find "$PCACHE/fi-autotune-v2" -mindepth 1 -maxdepth 3 -type d 2>/dev/null | sort; }
hostmem(){ # tag phase [container]
  local tag=$1 ph=$2 c=${3:-} f="$R/hostmem-$1-$2.txt" pid
  { echo "date $(date '+%F %T %Z') tag=$tag phase=$ph container=${c:--}"
    grep -E '^(MemTotal|MemFree|MemAvailable|Cached|Shmem|Mlocked|Unevictable|AnonPages|Mapped):' /proc/meminfo | awk '{printf "meminfo %s %.2f GiB\n",$1,$2/1048576}'
    if [[ -n $c ]] && docker ps --format '{{.Names}}' | grep -qx "$c"; then
      pid=$(docker top "$c" -eo pid,rss,args 2>/dev/null | grep 'VLLM::EngineCore' | awk '{print $1}' | head -1)
      if [[ -n $pid ]]; then
        echo "enginecore_pid $pid"
        sudo -n awk '/^[0-9a-f]+-[0-9a-f]+ /{dz=($0 ~ /\/dev\/zero/); if(dz)n++} dz&&/^Size:/{s+=$2} dz&&/^Rss:/{r+=$2} END{printf "pinned_devzero maps=%d size_gib=%.2f rss_gib=%.2f\n",n,s/1048576,r/1048576}' /proc/$pid/smaps
      fi
    fi
  } > "$f" 2>&1
  log "HOSTMEM $tag $ph: $(grep -E 'MemAvailable|pinned_devzero' "$f" | tr '\n' ';' | tr -s ' ' | cut -c1-240)"; }
# wait_bind: dump logs to a FILE then grep (never `docker logs | grep -q` under pipefail). Fast-fail only on hard errors.
wait_bind(){ # name maxmin [since] [nofastfail]
  local n=$1 maxmin=$2 since=${3:-} nff=${4:-} i t0 lf=$R/.wb-$1.log; t0=$(date +%s)
  for ((i=0;i<maxmin*6;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $(( $(date +%s)-t0 ))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n after $(( $(date +%s)-t0 ))s"; return 1; }
    if [[ -n $since ]]; then docker logs --since "$since" "$n" > "$lf" 2>&1; else docker logs "$n" > "$lf" 2>&1; fi
    if [[ -z $nff ]] && grep -qE 'ValueError:|Traceback|ERROR|RuntimeError:' "$lf"; then
      log "FASTFAIL $n after $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < "$lf" | grep -E 'ValueError:|Traceback|ERROR|RuntimeError:' | tail -3 | cut -c1-240 | tr '\n' '|')"; return 1; fi
    (( i % 30 == 0 )) && log "  $n waiting $((i/6)) min: store_entries=$(store_entries) MemAvail=$(memavail)GiB last=[$(tr '\r' '\n' < "$lf" | grep -E 'rehome|Autotun|Preloaded|Capturing|KV cache size|Decoder SWA|Loading safetensors|Loading weights' | tail -1 | cut -c1-160)]"
    sleep 10
  done; log "TIMEOUT $n ($maxmin min)"; return 1; }
boot_facts(){ # tag name [since]
  local tag=$1 name=$2 since=${3:-}
  if [[ -n $since ]]; then docker logs --since "$since" "$name" > "$R/boot-$tag.log" 2>&1; else docker logs "$name" > "$R/boot-$tag.log" 2>&1; fi
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -E "Decoder SWA bounded replay|SWA bounded replay|Using V[12] Model Runner|KV cache format|data type to store kv cache|attention backend for the DeepSeek-V4 DSpark drafter|Using .* backend|MEGA_ATTN|MegaAttn|SPARSE_DSV41|Using FlashInfer (managed )?autotune cache|Managed autotune cache|Preloaded [0-9]+/[0-9]+|Autotuning process (starts|ends)|Loaded [0-9]+ configs|Saved [0-9]+ configs|GPU KV cache size|Available KV cache memory|Setting kv cache block size|Graph capturing finished|PIN_HOT installed|PIN_HOT abort|PIN_HOT hook FAILED|rehome layer=9 |rehome done|Model loading took|model loading took" | grep -vE 'it/s\]|non-default args|Initializing a V1 LLM engine' | cut -c1-400 > "$R/facts-$tag.txt"
  local ts te; ts=$(grep -m1 'Autotuning process starts' "$R/facts-$tag.txt" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1); te=$(grep 'Autotuning process ends' "$R/facts-$tag.txt" | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "TUNEWINDOW ${ts:-?} ${te:-?}" >> "$R/facts-$tag.txt"; echo "STORE_ENTRIES $(store_entries)" >> "$R/facts-$tag.txt"
  log "FACTS $tag: replay_line=$(grep -c 'Decoder SWA bounded replay:' "$R/facts-$tag.txt") drafter_inherit_line=$(grep -c 'attention backend for the DeepSeek-V4 DSpark drafter' "$R/facts-$tag.txt") mega_lines=$(grep -ciE 'MEGA_ATTN|MegaAttn' "$R/facts-$tag.txt") kvfmt=[$(grep -oE "Using DeepSeek's [a-z0-9_]+ KV cache format" "$R/facts-$tag.txt" | sort -u | tr '\n' ' ')] rehome=$(grep -c 'rehome done layers=40' "$R/facts-$tag.txt") | autotune: starts=$(grep -c 'Autotuning process starts' "$R/facts-$tag.txt") window=${ts:-?}->${te:-?} managed=$(grep -oE 'Preloaded [0-9]+/[0-9]+' "$R/facts-$tag.txt" | tail -1) loaded=$(grep -oE 'Loaded [0-9]+ configs' "$R/facts-$tag.txt" | tail -1) saved=$(grep -oE 'Saved [0-9]+ configs' "$R/facts-$tag.txt" | tail -1) store_entries=$(store_entries) | $(grep 'GPU KV cache size' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //' | cut -c1-60) | guard: $(grep 'rehome layer=9 ' "$R/facts-$tag.txt" | tail -1 | grep -oE 'host_avail=[0-9.]+GiB' | head -1) done:$(grep 'rehome done' "$R/facts-$tag.txt" | tail -1 | grep -oE 'host_avail=[0-9.]+GiB' | head -1)"; }
# ---------------- cells ----------------
gsm8k(){ # tag baseline_json
  local tag=$1 base=$2
  ( cd "$LB" && LLM_BENCH_NO_UPDATE_CHECK=1 timeout 1500 python3 llm_decode_bench.py --display-mode plain --host 127.0.0.1 --port 30006 --model $M \
      --test-profile gsm8k --profile-runs 200 --compare-baseline "$base" --output "$R/gsm8k200-$tag.json" ) > "$R/gsm8k200-$tag.stdout.txt" 2>&1 || log "gsm8k $tag nonzero rc"
  log "GSM8K $tag: $(python3 - "$R/gsm8k200-$tag.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); a=d['accuracy']; c=d.get('comparison') or {}
    print(f"acc={a['accuracy']} ({a['correct']}/{a['scored']}) errors={a['errors']} trunc={a['truncated_no_answer']} | vs base {c.get('baseline_accuracy')} delta_pp={c.get('delta_pp')} mcnemar_p={c.get('mcnemar_exact_p')} ctok_ratio={(c.get('completion_tokens') or {}).get('ratio_mean')}")
except Exception as e: print('ERR',e)
PY
)"; }
bfcl(){ # tag suite
  local tag=$1 suite=$2
  ( cd "$R" && SUITE=$suite CONC=8 MAXTOK=2048 THINKING=0 timeout 2400 python3 $S/bfcl_gate.py "$tag-$suite" "$BFCL" "$R" ) > "$R/bfcl-$tag-$suite.log" 2>&1 || log "bfcl $tag $suite nonzero rc"
  log "BFCL $tag $suite: $(grep -h '^BFCL' "$R/bfcl-$tag-$suite.log" | tail -1 | cut -c1-220)"; }
rt_c1(){ local tag=$1 N O r
  N="p2f-$1-$(date +%s)"; O="$R/rt-$1"; mkdir -p "$O"
  for r in 1 2 3; do timeout 900 python3 $BK $U $M $P/eval-c1-r$r-try1.jsonl $O/c1-r$r.json 1 --warm $P/eval-c1-r$r-try1-warm.jsonl --nonce $N-c1-r$r > "$O/c1-r$r.stdout" 2>&1; done
  log "RT $tag: $(python3 - "$O" <<'PY'
import json,sys
O=sys.argv[1]; out=[]
for k in ('c1-r1','c1-r2','c1-r3'):
    try:
        s=json.load(open(f'{O}/{k}.json'))['summary']; out.append(f"{k} agg={s['decode_agg']} ttft={s['median_ttft_s']} acc={s.get('acceptance_len')}")
    except Exception as e: out.append(f'{k} ERR {e}')
print(' ; '.join(out))
PY
)"; }
knee6x2(){ local tag=$1 k
  for k in 1 2; do
    ( cd $HOME/dsv41 && timeout 900 bash knee6.sh "P2f-$tag-k6r$k" ) > "$R/knee6-$tag-r$k.txt" 2>&1 || log "knee6 $tag r$k nonzero rc"
    mv -f "$HOME/dsv41/knee-P2f-$tag-k6r$k.json" "$R/" 2>/dev/null
    log "KNEE6 $tag r$k: $(grep -E 'conc= ?(1|6|8):' "$R/knee6-$tag-r$k.txt" | sed 's/  */ /g' | tr '\n' ';' | cut -c1-400)"
  done; }
knee12(){ local tag=$1 r
  for r in r1 r2; do
    ( cd $HOME/dsv41 && timeout 900 bash knee.sh "P2f-$tag-$r" ) > "$R/knee-$tag-$r.txt" 2>&1 || log "knee $tag $r nonzero rc"
    mv -f "$HOME/dsv41/knee-P2f-$tag-$r.json" "$R/" 2>/dev/null
    log "KNEE $tag $r: $(grep -E 'conc= ?(1|8|16):' "$R/knee-$tag-$r.txt" | sed 's/  */ /g' | tr '\n' ';' | cut -c1-400)"
  done; }
cold64(){ local tag=$1
  BASE_URL=$U/v1 MODEL=$M API_KEY=none SIZES="65536" N=1 THINKING=0 timeout 900 python3 $HOME/dsv41/cold_prefill_probe.py > "$R/cold-$tag.txt" 2>&1 || log "cold $tag nonzero rc"
  log "COLD64K $tag: $(grep PREFILL "$R/cold-$tag.txt" | cut -c1-200 | tr '\n' ';')"; }
gauntlet(){ local tag=$1
  rm -rf "$R/gauntlet-$tag"
  (cd "$GDIR" && timeout 1200 python3 run_suite.py --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "p2f-$tag" --cache-salt "p2f-$tag" --sandbox-image "$GIMG" --workers 4) > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"; }
perclass(){ # arm outdir [ctl_dir ctl]
  local arm=$1 od=$2 cdir=${3:-} ct=${4:-}
  log "PERCLASS $arm start -> $od ${ct:+(CTL=$ct)}"
  if [[ -n $ct ]]; then ( cd "$BENCH" && ARM=$arm OUTDIR=$od CTL_DIR=$cdir CTL=$ct timeout 2700 bash perclass_run.sh ) || log "perclass $arm nonzero rc"
  else ( cd "$BENCH" && ARM=$arm OUTDIR=$od timeout 2700 bash perclass_run.sh ) || log "perclass $arm nonzero rc"; fi
  log "PERCLASS $arm done: sha=$(grep -m1 -oE 'sha1=[0-9a-f]+' "$od/runner.log") $(grep -h '^GATE' "$od"/perclass-table-*-vs-*.md 2>/dev/null | tr '\n' ';') fails=[$(grep -h 'FAIL |' "$od"/perclass-table-*-vs-*.md 2>/dev/null | awk -F'|' '{print $2"/"$3" "$8}' | tr -s ' ' | tr '\n' ';')]"; }
metrics(){ curl -s $U/metrics | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$R/metrics-$1.txt" 2>/dev/null; }
errcount(){ # tag name since
  docker logs --since "$3" "$2" > "$R/postlog-$1.log" 2>&1; grep -cE 'ValueError:|Traceback|ERROR|RuntimeError:' "$R/postlog-$1.log" > "$R/errcount-$1.txt"
  log "ERRCOUNT $1 $(cat "$R/errcount-$1.txt") (since $3)"; }
speed_cells(){ local tag=$1; rt_c1 "$tag"; knee6x2 "$tag"; knee12 "$tag"; }

# ================= preflight =================
log "===== CARD P2f START (REF=$REF CAND=$CN) ====="
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running for A"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: REF not bound on :30006"; exit 5; }
docker ps -a --format '{{.Names}}' | grep -qx "$CN" || { log "ABORT: candidate container $CN missing"; exit 5; }
docker ps --format '{{.Names}}' | grep -qx "$CN" && { log "ABORT: candidate $CN is running (expected stopped)"; exit 5; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: image not present"; exit 5; }
[[ "$(sha1sum "$PROMPTS" | cut -c1-8)" == "618898aa" ]] || { log "ABORT: prompts sha1 != 618898aa"; exit 5; }
for f in perclass-v23-r1-c1.json perclass-v23-r1-c8.json perclass-v23-r2-c1.json perclass-v23-r2-c8.json perclass-table-v23.json; do
  [[ -s $PC_BASE/$f ]] || { log "ABORT: baseline $PC_BASE/$f missing"; exit 5; }; done
for d in "$PC_C" "$PC_D"; do [[ -e $d ]] && { log "ABORT: $d already exists (refuse to mix runs)"; exit 5; }; done
[[ -f $LB/llm_decode_bench.py && -s $LBREF && -f $S/bfcl_gate.py && -f $GDIR/run_suite.py && -f $W/cardP2f_verdict.py && -f $BENCH/perclass_run.sh ]] || { log "ABORT: instruments missing"; exit 5; }
log "preflight OK: MemAvailable=$(memavail)GiB store_entries=$(store_entries) sha: perclass_decode=$(sha1sum $BENCH/perclass_decode.py | cut -c1-8) perclass_run=$(sha1sum $BENCH/perclass_run.sh | cut -c1-8) perclass_table=$(sha1sum $BENCH/perclass_table.py | cut -c1-8) prompts=$(sha1sum $PROMPTS | cut -c1-8) lilbench=$(sha1sum $LB/llm_decode_bench.py | cut -c1-8) bfcl_gate=$(sha1sum $S/bfcl_gate.py | cut -c1-8) knee6=$(sha1sum $HOME/dsv41/knee6.sh | cut -c1-8) knee=$(sha1sum $HOME/dsv41/knee.sh | cut -c1-8) verdict=$(sha1sum $W/cardP2f_verdict.py | cut -c1-8)"
wait_quiet A || { log "ABORT: host never quiet; lane untouched (REF serving)"; exit 6; }

# ================= A: live v23 =================
sinceA=$(date -u +%FT%TZ)
boot_facts A "$REF" "$(docker inspect -f '{{.State.StartedAt}}' "$REF")"; strays; hostmem A live "$REF"
log "--- MEASURE A ($REF) ---"
gsm8k A "$LBREF"; bfcl A dev; bfcl A heldout
speed_cells A; gauntlet A; cold64 A; metrics A
errcount A "$REF" "$sinceA"; log "--- END MEASURE A ---"

# ================= C: config C store-loaded restart =================
wait_quiet C || log "WARN: host not quiet before C — proceeding"
log "LANE DARK begins (stopping REF for C)"
stop_all; drop_caches; strays; hostmem C pre ""
CBOUND=0
ma=$(memavail)
if awk -v m="$ma" 'BEGIN{exit !(m < 380)}'; then log "ABORT C: MemAvailable ${ma} GiB < 380 after stop+drop_caches (stray?) — skipping to D"
else
  STORE0=$(store_entries); store_keys > "$R/store-keys-before-C.txt"; log "store entries before C: $STORE0"
  t0=$(date +%s); sinceC=$(date -u +%FT%TZ); docker start "$CN" >/dev/null
  if wait_bind "$CN" 30 "$sinceC"; then
    CBOUND=1; log "boot->ready C $(( $(date +%s)-t0 ))s ($CN)"; store_keys > "$R/store-keys-after-C.txt"
    boot_facts C "$CN" "$sinceC"; hostmem C bound "$CN"; log "store entries after C bind: $(store_entries) (before $STORE0)"
    grep -q 'Decoder SWA bounded replay:' "$R/facts-C.txt" && log "WARN C: replay line PRESENT -> C INVALID"
    grep -q 'rehome done layers=40' "$R/facts-C.txt" || log "WARN C: hook did NOT re-home -> C INVALID"
    grep -qiE 'MEGA_ATTN|MegaAttn' "$R/facts-C.txt" || log "WARN C: no MegaAttn line in boot facts — check facts-C.txt"
    log "--- MEASURE C ($CN) ---"
    perclass candC "$PC_C" "$PC_BASE" v23
    gsm8k C "$R/gsm8k200-A.json"; bfcl C dev; bfcl C heldout
    speed_cells C; gauntlet C; cold64 C; metrics C; hostmem C steady "$CN"
    errcount C "$CN" "$sinceC"; log "--- END MEASURE C ---"
  else
    docker logs --since "$sinceC" "$CN" > "$R/boot-C-FAIL.log" 2>&1
    log "BOOTFAIL C tail: $(tr '\r' '\n' < "$R/boot-C-FAIL.log" | grep -E 'PIN_HOT abort|Error:|ERROR|EngineCore failed|OutOfMemory' | tail -3 | cut -c1-220 | tr '\n' '|')"
  fi
  docker ps --format '{{.Names}}' | grep -qx "$CN" && { docker stop "$CN" >/dev/null; log "stop-and-keep $CN"; }
fi

# ================= D: v23 REF back, left serving =================
wait_quiet D || log "WARN: host not quiet before D — booting REF anyway (lane must return)"
stop_all; drop_caches; strays
t0=$(date +%s); sinceD=$(date -u +%FT%TZ); docker start "$REF" >/dev/null
if wait_bind "$REF" 40 "$sinceD" nofastfail; then
  log "boot->ready D $(( $(date +%s)-t0 ))s"; log "LANE DARK ends (REF bound)"
  boot_facts D "$REF" "$sinceD"; hostmem D bound "$REF"
  log "--- MEASURE D ($REF) ---"
  perclass v23post "$PC_D"
  speed_cells D; cold64 D; gauntlet D
  log "D report-only fidelity (not gated)"; gsm8k D "$R/gsm8k200-A.json"; bfcl D dev; bfcl D heldout
  metrics D; errcount D "$REF" "$sinceD"; log "--- END MEASURE D ---"
else
  log "D REF FAILED TO BIND — lane dark, needs hands"
  docker start "$REF" >/dev/null 2>&1
fi

# ================= tables, verdict, suffix (no promotion) =================
if [[ -d $PC_C && -d $PC_D ]]; then
  ( cd "$BENCH" && python3 perclass_table.py --outdir "$PC_C" --arm candC --control-dir "$PC_D" --control v23post ) > "$R/perclass-candC-vs-v23post.stdout" 2>&1
  ( cd "$BENCH" && python3 perclass_table.py --outdir "$PC_D" --arm v23post --control-dir "$PC_BASE" --control v23 ) > "$R/perclass-v23post-vs-v23.stdout" 2>&1
  log "PERCLASS2 candC vs v23post: $(grep -h '^GATE' "$PC_C/perclass-table-candC-vs-v23post.md" 2>/dev/null) | drift view v23post vs v23: $(grep -h '^GATE' "$PC_D/perclass-table-v23post-vs-v23.md" 2>/dev/null)"
fi
cp -f "$PC_C"/perclass-table-*.md "$PC_D"/perclass-table-*.md "$R/" 2>/dev/null
python3 $W/cardP2f_verdict.py "$R" "$PC_BASE" "$PC_C" "$PC_D" > "$R/VERDICT.md" 2> "$R/verdict.err"
log "VERDICT: $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md")"
s=$(grep -m1 "^SUFFIX " "$R/VERDICT.md" | awk '{print $2}')
if docker ps -a --format '{{.Names}}' | grep -qx "$CN"; then
  docker ps --format '{{.Names}}' | grep -qx "$CN" && docker stop "$CN" >/dev/null 2>&1
  docker rename "$CN" "$CN-${s:-P2f-DONE}" && log "kept $CN-${s:-P2f-DONE} (stopped)"; fi
grep -hE ' (GSM8K|BFCL|RT|KNEE6|KNEE|COLD64K|GAUNTLET|PERCLASS|PERCLASS2|FACTS|HOSTMEM|ERRCOUNT|boot->ready|BOUND|BOOTFAIL|FASTFAIL|DIED|TIMEOUT|WARN|ABORT|VERDICT|LANE|store entries|kept) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card P2f: promotion gate on config C (nightly-39983a5e + hook, MegaAttn, replay off) vs v23 — A live v23 → C store-loaded restart → D v23 restart — $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md" 2>/dev/null)"
  grep -vE '^SUFFIX|^GATE_RESULT' "$R/VERDICT.md"; echo; echo "Runner summary lines:"; sed 's/^/    /' "$R/SUMMARY.txt"; echo "Receipts: $R, $PC_C, $PC_D (Mac copy: el8-cards-2026-10-18/P2/receipts-P2f/). NOT promoted — James decides."; } >> "$LEDGER"
log "FINAL dsv41 running=[$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')] bound=$(curl -s -m 3 $U/v1/models | grep -c $M)"
log "===== CARD P2f END (lane = $REF) ====="

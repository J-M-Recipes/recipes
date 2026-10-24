#!/usr/bin/env bash
# cardKV1_runner.sh — R1 of ~/hermes/research/dsv41-kv-20261022/KV-ROUTES.md:
#   boot v25 sidecar with an EXPLICIT --kv-cache-memory (8.5 GiB) instead of letting util 0.97 decide,
#   then re-run the SC5 "fit" cell (1M needle + post probe + depth decode) and a C1 load pre/post.
# Authorized: James said "ok; go" 2026-10-22 ~18:35 CDT for R6+R1; LANE-OWNER.txt requires his word to touch :30006.
# Rollback = the stopped-and-kept ...-kv4.9-KEPT pair (start it, ~1 min) or v23 RETIRED-REF (~5 min).
# ws v25 sidecar (af7f9488 + el8 hook + overlay-58132, rowmap-static-v1, 295/89, spins 2e9, fail-stop watchdog).
set -uo pipefail
D=~/dsv41/m3; R=$D/kv1/results-cardKV1
LAUNCH=$D/launch_sidecar.sh; WD=$D/sidecar_watchdog.py; WDF=$D/wd_failstop.sh
SC=$D/sc5; PROBE=$SC/sc5_probes.py
LANE=dsv41-vllm-1017-v25-sidecar-BOUND-REF; RTXC=$LANE-rtx
KEPT=dsv41-vllm-1017-v25-sidecar-kv4.9-KEPT; KEPTR=$KEPT-rtx
OLDR=dsv41-vllm-1017-v23-pinexact-RETIRED-REF
U=http://127.0.0.1:30006; M=dsv41-flash-uva
ROWMAP=rowmap-static-v1.json; MAXLEN=1048576
KVBYTES=9126805504          # 8.5 GiB
BASELINE_TOK=2537081        # this lane's KV tokens at util 0.97 (SC5 F_boot, 2026-10-22)
P=$D/bench/k3pool/long; POOL=$P/eval-c16-try1.jsonl
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55; RTX=GPU-d07eb6e6-7053-42a7-6811-7bb833266182
LEDGER=~/dsv41/results/ledger.md
FF=' ERROR [0-9]{2}-[0-9]{2} |Traceback \(most recent call last\)|ValueError:|RuntimeError:|OutOfMemoryError|EngineCore failed|No available memory for the cache blocks'

mkdir -p "$R"
[[ -e $R/runner.log ]] && { echo "ABORT: $R/runner.log exists" >&2; exit 5; }
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
state(){ echo "$*" > "$R/STATE"; log "STATE $*"; }
memavail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo; }
rtx_used(){ nvidia-smi --id=$RTX --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
is_up(){ [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
names(){ docker ps -a --format '{{.Names}}'; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
any_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '; }
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh)' | grep -v cardKV1_runner.sh | grep -v grep | cut -c1-120 | tr '\n' '|'; }

restore_kept(){ # put the 4.9 pair back and leave it serving
  log "RESTORE: bringing $KEPT back as $LANE"
  docker stop "$LANE" "$RTXC" >/dev/null 2>&1
  docker rename "$LANE" "dsv41-vllm-1017-v25-kv8.5-FAIL" 2>/dev/null
  docker rename "$RTXC" "dsv41-vllm-1017-v25-kv8.5-FAIL-rtx" 2>/dev/null
  docker rename "$KEPT" "$LANE"; docker rename "$KEPTR" "$RTXC"
  docker start "$LANE" "$RTXC" >/dev/null
  for i in $(seq 1 90); do curl -s -m 2 $U/v1/models | grep -q "$M" && { log "RESTORE bound: $LANE serving (v25 kv4.9)"; return 0; }; sleep 10; done
  log "RESTORE: did not bind in 15 min — starting $OLDR as last resort"
  docker start "$OLDR" >/dev/null 2>&1
  for i in $(seq 1 60); do curl -s -m 2 $U/v1/models | grep -q "$M" && { log "RESTORE bound: $OLDR (v23) serving"; return 1; }; sleep 10; done
  log "RESTORE: NOTHING BOUND — NEEDS HANDS"; return 2; }

rollback(){ log "ROLLBACK: $1"
  docker logs "$LANE" > "$R/fail-server.log" 2>&1; docker logs "$RTXC" > "$R/fail-rtx.log" 2>&1
  docker kill "$LANE" >/dev/null 2>&1; docker stop "$RTXC" >/dev/null 2>&1
  restore_kept; rc=$?
  bash "$WDF" >> "$R/watchdog-rearm.log" 2>&1
  state "FAIL"; echo "FAIL: $1" > "$R/verdict.txt"; touch "$R/DONE"; exit 3; }

wait_bind(){ local n=$1 maxmin=$2 side=$3 since=$4 i t0 lf=$R/boot-wait.log; t0=$(date +%s)
  for ((i=0;i<maxmin*6;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $(( $(date +%s)-t0 ))s"; return 0; }
    is_up "$n" || { log "DIED $n after $(( $(date +%s)-t0 ))s"; return 1; }
    is_up "$side" || { log "DIED sidecar $side after $(( $(date +%s)-t0 ))s"; return 1; }
    # NB: do NOT filter with `docker logs --since` — a naive host timestamp filtered to 1 line on 2026-10-22
    # (container logs are stamped in UTC). The container is fresh, so its whole log IS this boot.
    docker logs "$n" > "$lf" 2>&1
    if grep -qE "$FF" "$lf"; then log "FASTFAIL: $(tr '\r' '\n' < "$lf" | grep -E "$FF" | tail -2 | cut -c1-200 | tr '\n' '|')"; return 1; fi
    (( i % 30 == 0 )) && log "  waiting $((i/6)) min: MemAvail=$(memavail)GiB RTX=$(rtx_used)MiB last=[$(tr '\r' '\n' < "$lf" | grep -E 'MEGA_PEER|Autotun|Capturing|KV cache size|Decoder SWA|Loading weights|Preloaded|Saved [0-9]+ configs' | tail -1 | cut -c1-140)]"
    sleep 10
  done
  log "TIMEOUT $n ($maxmin min)"; return 1; }

log "===== CARD KV1 (explicit --kv-cache-memory 8.5 GiB) START ====="
# ---- preflight
o=$(other_runners); [[ -n $o ]] && { log "ABORT: other runner live: $o"; exit 4; }
f=$(any_foreign); [[ -n ${f// /} ]] && { log "ABORT: foreign container: $f"; exit 4; }
is_up "$LANE" || { log "ABORT: $LANE not running"; exit 5; }
is_up "$RTXC" || { log "ABORT: $RTXC not running"; exit 5; }
names | grep -qx "$KEPT" && { log "ABORT: $KEPT already exists"; exit 5; }
names | grep -qx "$OLDR" || { log "ABORT: last-resort $OLDR missing"; exit 5; }
[[ -f $POOL ]] || { log "ABORT: pool missing $POOL"; exit 5; }
pgrep -f sidecar_watchdog.py >/dev/null || log "WARN: no watchdog running before start"
log "preflight: MemAvailable=$(memavail)GiB RTX=$(rtx_used)MiB LANE up; owner note written"
cat > "$D/LANE-OWNER.txt" <<EOF
OWNER: Milo cardKV1 (R1 explicit --kv-cache-memory 8.5 GiB) since $(date '+%F %T %Z'). Authorized by James "ok; go" 2026-10-22.
Lane: v25 sidecar (af7f9488 + el8 hook + overlay-58132, rowmap-static-v1 295/89, spins 2e9).
Do NOT start/stop :30006 containers while this is live. Rollback stock: $(echo $KEPT) (+-rtx), then v23 RETIRED-REF.
EOF
state BOOTING
# ---- disarm watchdog (fail-stop arming returns at the end)
P0=$(cat $D/logs/sidecar_watchdog.pid 2>/dev/null); [[ -n ${P0:-} ]] && kill "$P0" 2>/dev/null
for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -f sidecar_watchdog.py >/dev/null || break; sleep 1; done
pgrep -f sidecar_watchdog.py >/dev/null && { log "ABORT: watchdog still running"; exit 6; }
log "watchdog (pid ${P0:-none}) disarmed"

# ---- stop-and-keep the 4.9 lane, boot the candidate under the SAME service name
log "stop-and-keep $LANE -> $KEPT  (LANE DARK begins)"
docker stop "$LANE" "$RTXC" >/dev/null; docker rename "$LANE" "$KEPT"; docker rename "$RTXC" "$KEPTR"
drop_caches
t0=$(date +%s); SINCE=$(date '+%F %T')
NAME=$LANE ROWMAP=$ROWMAP bash "$LAUNCH" sidecar || rollback "sidecar launch failed"
PEER_SPINS=2000000000 REPLAY_OVERLAY=1 EXTRA="--kv-cache-memory $KVBYTES" MEGA_PEER_CHECK=0 MEGA_COLD_TRT=0 \
  NAME=$LANE ROWMAP=$ROWMAP MAXLEN=$MAXLEN bash "$LAUNCH" server || rollback "server launch failed"
wait_bind "$LANE" 45 "$RTXC" "$SINCE" || rollback "no bind / died during boot"
log "LANE DARK ends after $(( $(date +%s) - t0 ))s"
state BOUND

# ---- boot facts (full container log: the container is fresh, so all of it is this boot)
docker logs "$LANE" > "$R/boot.log" 2>&1
tr '\r' '\n' < "$R/boot.log" | grep -E "GPU KV cache size|Maximum concurrency|non-default args|Decoder SWA bounded replay|peer mode 2|MEGA_PEER rowmap|Available KV cache memory|as specified by kv_cache_memory_bytes" | cut -c1-400 > "$R/facts.txt"   # non-default args line is truncated at 400 chars; the flag shows up in the gpu_worker 'reserved ... kv_cache_memory_bytes' line
KVTOK=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$R/facts.txt" | tail -1 | tr -dc '0-9')
CONC=$(grep -oE 'Maximum concurrency for 1,048,576 tokens per request: [0-9.]+x' "$R/facts.txt" | tail -1 | grep -oE '[0-9.]+x')
log "FACTS: kv_tokens=${KVTOK:-none} concurrency=${CONC:-none} peer2=$(grep -c 'peer mode 2, check 0' "$R/facts.txt") rowmap40=$(grep -c "MEGA_PEER rowmap /w/$ROWMAP: 40 layers" "$R/facts.txt") replay=$(grep -c 'Decoder SWA bounded replay: in eager' "$R/facts.txt") kvflag=$(grep -c "kv_cache_memory" "$R/facts.txt") RTX=$(rtx_used)MiB"
grep -q 'peer mode 2, check 0' "$R/facts.txt" || rollback "boot facts: peer mode 2 missing"
grep -q "MEGA_PEER rowmap /w/$ROWMAP: 40 layers" "$R/facts.txt" || rollback "boot facts: rowmap not 40 layers"
[[ -n ${KVTOK:-} ]] || rollback "boot facts: no KV cache size line"

# ---- probes: C1 load (pre) -> 1M fit -> C1 load (post)
python3 "$PROBE" load --base $U --model $M --conc 1 --duration 150 --ramp 5 --pool "$POOL" \
  --phase-name kv1_c1_pre --phase-file "$R/phases.tsv" --out "$R/load-pre.json" > "$R/load-pre.log" 2>&1 || log "WARN load pre rc"
log "LOAD pre: $(python3 -c "import json;d=json.load(open('$R/load-pre.json'));print('ok',d['ok'],'err',d['errors'],'tok/s %.1f'%d['output_tok_s_window_completed'])" 2>&1 | cut -c1-160)"
state FIT
timeout 3000 python3 "$PROBE" fit --base $U --model $M --tag kv1 --out "$R/fit.json" --target 1000000 --maxlen "$MAXLEN" > "$R/fit.log" 2>&1 || log "WARN fit nonzero rc"
log "FIT: $(tail -1 "$R/fit.log" | cut -c1-300)"
python3 "$PROBE" load --base $U --model $M --conc 1 --duration 150 --ramp 5 --pool "$POOL" \
  --phase-name kv1_c1_post --phase-file "$R/phases.tsv" --out "$R/load-post.json" > "$R/load-post.log" 2>&1 || log "WARN load post rc"
log "LOAD post: $(python3 -c "import json;d=json.load(open('$R/load-post.json'));print('ok',d['ok'],'err',d['errors'],'tok/s %.1f'%d['output_tok_s_window_completed'])" 2>&1 | cut -c1-160)"

# ---- verdict
python3 - "$R" "$KVTOK" "$BASELINE_TOK" <<'PY' > "$R/verdict.txt" 2> "$R/verdict.err"
import json,sys,os
R,kv,baseline=sys.argv[1],int(sys.argv[2] or 0),int(sys.argv[3])
fit=json.load(open(f"{R}/fit.json")); pre=json.load(open(f"{R}/load-pre.json")); post=json.load(open(f"{R}/load-post.json"))
n=fit["needle"]; pp=fit["post_probe"]; dd=fit.get("depth_decode") or {}
boot=open(f"{R}/facts.txt").read()
gates={}
gates["capacity"]=["PASS" if kv>=4000000 else "FAIL", f"KV {kv:,} tokens (baseline {baseline:,}; bar >= 4,000,000 = +58%)"]
gates["flag"]=["PASS" if "kv_cache_memory" in boot else "FAIL", "explicit --kv-cache-memory visible in non-default args"]
gates["fit_1m"]=["PASS" if (n["http"]==200 and (n.get("prompt_tokens") or 0)>=950000 and n["recall"]) else "FAIL",
                 f"needle http={n['http']} pt={n.get('prompt_tokens')} ttft={n.get('ttft_s')}s recall={n['recall']} finish={n.get('finish_reason')}"]
gates["fit_after"]=["PASS" if pp["ok"] else "FAIL", f"post probe {pp.get('content')!r}"]
gates["depth_decode"]=["PASS" if (dd.get("http")==200 and (dd.get("decode_tok_s") or 0)>0) else "FAIL",
                       f"1M-depth decode {dd.get('decode_tok_s')} tok/s ttft={dd.get('ttft_s')}s"]
sp=post["output_tok_s_window_completed"]/max(pre["output_tok_s_window_completed"],1e-9)
gates["decode_self"]=["PASS" if sp>=0.97 else "FAIL", f"C1 after/before 1M = {sp:.3f} (bar >= 0.97)"]
gates["errors"]=["PASS" if (pre["errors"]==0 and post["errors"]==0 and post["ok"]>0) else "FAIL", f"load err pre={pre['errors']} post={post['errors']}"]
res="PASS" if all(v[0]=="PASS" for v in gates.values()) else "FAIL"
print(f"KV1 RESULT {res}"); print(f"KV tokens {kv:,} (baseline {baseline:,}); concurrency 1M = {kv/1048576:.2f}x")
for k,v in gates.items(): print(f"  {k:12s} {v[0]:4s}  {v[1]}")
json.dump({"result":res,"kv_tokens":kv,"baseline":baseline,"gates":gates,"fit":fit,"load_pre":pre,"load_post":post},
          open(f"{R}/verdict.json","w"), indent=1)
PY
RES=$(head -1 "$R/verdict.txt" | awk '{print $3}')
log "VERDICT: $RES"; sed 's/^/    /' "$R/verdict.txt" | head -12
docker logs "$LANE" > "$R/server-final.log" 2>&1
grep -cE "$FF" "$R/server-final.log" | sed 's/^/engine-error lines: /'

if [[ "$RES" == "PASS" ]]; then
  bash "$WDF" >> "$R/watchdog-rearm.log" 2>&1
  log "WATCHDOG: $(tail -2 "$R/watchdog-rearm.log" | tr '\n' ' ' | cut -c1-200)"
  printf "Lane = BOUND-REF with --kv-cache-memory 8.5 GiB (%s tokens, %s at 1M). Rollback pair stopped-and-kept as %s (+-rtx). Watchdog fail-stop re-armed.\n" "$KVTOK" "$CONC" "$KEPT" > "$D/LANE-OWNER.txt"
  state PASS
else
  log "NOT promoted: keeping the candidate boot for inspection is NOT safe (KV sizing change) — restoring 4.9"
  docker kill "$LANE" >/dev/null 2>&1; docker stop "$RTXC" >/dev/null 2>&1
  restore_kept
  bash "$WDF" >> "$R/watchdog-rearm.log" 2>&1
  printf "Lane restored to v25 kv4.9 (%s). cardKV1 verdict FAIL: %s\n" "$LANE" "$(sed -n 3,9p "$R/verdict.txt" | tr '\n' ';' | cut -c1-300)" > "$D/LANE-OWNER.txt"
  state FAIL
fi
{ echo; echo "## $(date '+%F %H:%M %Z') — Card KV1 (explicit --kv-cache-memory 8.5 GiB) — v25 sidecar lane — $RES"
  grep -vE '^SUFFIX' "$R/verdict.txt" | head -12; echo "Receipts: $R."; } >> "$LEDGER"
log "===== CARD KV1 DONE ($RES) ====="
touch "$R/DONE"

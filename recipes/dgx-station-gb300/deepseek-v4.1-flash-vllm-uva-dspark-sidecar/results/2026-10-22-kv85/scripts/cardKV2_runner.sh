#!/usr/bin/env bash
# cardKV2_runner.sh — (A) stress the LIVE 8.5 GiB lane (no reboot), then (B) R1b: boot 9.5 GiB, same stress + speed, keep only if
# it is no slower. James 2026-10-22 ~19:10 CDT: "yes plz stress test. also try 1" (1 = R1b 8.5 -> 9.5 GiB). Constraint: NO speed loss.
# Cells per arm:
#   speed  kbench C1 x6 (r1,r2,r3 pools x2, warm, nonce) + C16 x1   -> decode_agg
#   S1     warm-grow MIX, concurrent: 1M needle (+post probe, depth decode) + cold 128K prefill + kbench C16 (1024 out)
#   S2     CAPACITY, concurrent: 4 independent ~1M needles (distinct seeds, no prefix sharing) = ~4.0M tokens live
#   checks 0 engine-error lines, watchdog no TRIP, sidecar timeouts words[2] delta 0, server+sidecar up, chat health, GB300 mem peak
# Gates for B (9.5) to be KEPT: KV >= 4,850,000; S1+S2 pass; C1 mean >= 0.98 x A; C16 >= 0.97 x A. Else restore the 8.5 pair.
# A failing does not reboot anything (it is today's lane); it only reports. B is skipped if A fails.
set -uo pipefail
D=~/dsv41/m3; DS=~/dsv41; R=$D/kv2/results-cardKV2
LAUNCH=$D/launch_sidecar.sh; WDF=$D/wd_failstop.sh; WDLOG=$D/logs/sidecar_watchdog.log
PROBE=$D/sc5/sc5_probes.py; BK=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; COLD=$DS/cold_prefill_probe.py
LANE=dsv41-vllm-1017-v25-sidecar-BOUND-REF; RTXC=$LANE-rtx
K85=dsv41-vllm-1017-v25-sidecar-kv8.5-KEPT
U=http://127.0.0.1:30006; M=dsv41-flash-uva; ROWMAP=rowmap-static-v1.json; MAXLEN=1048576
KV85=9126805504; KV95=10200547328
SHM=/dev/shm/vllm_peer_tier2; GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55; RTX=GPU-d07eb6e6-7053-42a7-6811-7bb833266182
LEDGER=$DS/results/ledger.md
FF=' ERROR [0-9]{2}-[0-9]{2} |Traceback \(most recent call last\)|ValueError:|RuntimeError:|OutOfMemoryError|EngineCore failed|No available memory for the cache blocks|CUDA out of memory'

mkdir -p "$R"; [[ -e $R/runner.log ]] && { echo "ABORT: $R/runner.log exists" >&2; exit 5; }
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
state(){ echo "$*" > "$R/STATE"; log "STATE $*"; }
is_up(){ [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
names(){ docker ps -a --format '{{.Names}}'; }
memavail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo; }
w2(){ python3 -c "import struct;print(struct.unpack('<7q',open('$SHM','rb').read(56))[2])" 2>/dev/null || echo NA; }
trips(){ local c; c=$(grep -c 'TRIP' "$WDLOG" 2>/dev/null || true); echo "${c:-0}"; }   # grep -c prints 0 AND exits 1 on no match
bound(){ curl -s -m 3 $U/v1/models 2>/dev/null | grep -q "$M"; }
wait_bind(){ local maxmin=$1 i t0=$(date +%s)
  for ((i=0;i<maxmin*6;i++)); do
    bound && { log "BOUND after $(( $(date +%s)-t0 ))s"; return 0; }
    is_up "$LANE" || { log "DIED server"; return 1; }; is_up "$RTXC" || { log "DIED sidecar"; return 1; }
    docker logs "$LANE" > "$R/.wb.log" 2>&1
    grep -qE "$FF" "$R/.wb.log" && { log "FASTFAIL: $(tr '\r' '\n' < "$R/.wb.log" | grep -E "$FF" | tail -2 | cut -c1-200 | tr '\n' '|')"; return 1; }
    sleep 10; done; log "TIMEOUT bind"; return 1; }
arm_wd(){ bash "$WDF" >> "$R/watchdog-rearm.log" 2>&1; sleep 2; pgrep -f sidecar_watchdog.py >/dev/null && log "watchdog armed (fail-stop)" || log "WARN watchdog not running"; }
disarm_wd(){ local p; p=$(cat $D/logs/sidecar_watchdog.pid 2>/dev/null); [[ -n ${p:-} ]] && kill "$p" 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -f sidecar_watchdog.py >/dev/null || break; sleep 1; done
  pgrep -f sidecar_watchdog.py >/dev/null && { log "ABORT: watchdog still running"; exit 6; }; log "watchdog disarmed"; }
# recover from a watchdog trip / dead lane by restarting the CURRENT pair together (docker start keeps args)
recover_pair(){ log "RECOVER: restarting $LANE + $RTXC"; docker kill "$LANE" >/dev/null 2>&1; docker stop "$RTXC" >/dev/null 2>&1
  sleep 5; docker start "$RTXC" >/dev/null; sleep 20; docker start "$LANE" >/dev/null; wait_bind 20 && { arm_wd; return 0; }; log "RECOVER FAILED — NEEDS HANDS"; return 1; }

speed(){ local tag=$1 r k
  for k in 1 2; do for r in 1 2 3; do
    timeout 600 python3 $BK $U $M $P/eval-c1-r$r-try1.jsonl "$R/$tag-c1-$k-r$r.json" 1 --warm $P/eval-c1-r$r-try1-warm.jsonl --nonce "kv2-$tag-c1-$k-$r-$(date +%s)" > "$R/$tag-c1-$k-r$r.stdout" 2>&1
  done; done
  timeout 900 python3 $BK $U $M $P/eval-c16-try2.jsonl "$R/$tag-c16.json" 16 --nonce "kv2-$tag-c16-$(date +%s)" --maxhit 100 --max-tokens 1024 > "$R/$tag-c16.stdout" 2>&1
  python3 - "$R" "$tag" <<'PY' > "$R/$tag-speed.txt"
import json,glob,sys,statistics as st
R,t=sys.argv[1:3]
c1=[json.load(open(f))["summary"]["decode_agg"] for f in sorted(glob.glob(f"{R}/{t}-c1-*-r*.json"))]
c16=json.load(open(f"{R}/{t}-c16.json"))["summary"]
print(json.dumps({"c1":c1,"c1_mean":round(st.mean(c1),2),"c1_sd":round(st.pstdev(c1),2),"c16_agg":c16["decode_agg"],"c16_ok":c16["ok"],"c16_np":c16["np"]}))
PY
  log "SPEED $tag: $(cat "$R/$tag-speed.txt")"; }

stress(){ local tag=$1 since t0 w0 tr0 pids=() i
  since=$(date -u +%FT%TZ); w0=$(w2); tr0=$(trips); t0=$(date +%s)
  ( while :; do nvidia-smi --id=$GB300 --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > "$R/$tag-gpumem.txt" 2>/dev/null & local smp=$!
  log "S1 $tag: 1M needle + cold 128K + C16 concurrently (w2=$w0 trips=$tr0 MemAvail=$(memavail))"
  timeout 2400 python3 "$PROBE" fit --base $U --model $M --tag "$tag-s1" --out "$R/$tag-s1-fit.json" --target 1000000 --maxlen $MAXLEN > "$R/$tag-s1-fit.log" 2>&1 & pids+=($!)
  sleep 8
  BASE_URL=$U/v1 MODEL=$M API_KEY=none SIZES=161800 N=1 THINKING=0 timeout 1200 python3 $COLD > "$R/$tag-s1-cold128k.txt" 2>&1 & pids+=($!)
  timeout 1200 python3 $BK $U $M $P/eval-c16-try1.jsonl "$R/$tag-s1-c16.json" 16 --nonce "kv2-$tag-s1c16-$(date +%s)" --maxhit 100 --max-tokens 1024 > "$R/$tag-s1-c16.stdout" 2>&1 & pids+=($!)
  for i in "${pids[@]}"; do wait "$i"; done
  log "S1 $tag done in $(( $(date +%s)-t0 ))s: $(tail -1 "$R/$tag-s1-fit.log" | cut -c1-220) | 128K: $(grep PREFILL "$R/$tag-s1-cold128k.txt" | cut -c1-120) | C16: $(python3 -c "import json;s=json.load(open('$R/$tag-s1-c16.json'))['summary'];print(s['ok'],'/',s['np'],'agg',s['decode_agg'])" 2>&1 | tail -1)"
  pids=(); t0=$(date +%s)
  log "S2 $tag: 4 concurrent ~1M needles (distinct seeds)"
  for i in 1 2 3 4; do
    timeout 2400 python3 "$PROBE" fit --base $U --model $M --tag "$tag-s2-$i" --out "$R/$tag-s2-fit$i.json" --target 1000000 --maxlen $MAXLEN --seed $((1000+i)) --depth-decode 0 > "$R/$tag-s2-fit$i.log" 2>&1 & pids+=($!)
    sleep 3; done
  for i in "${pids[@]}"; do wait "$i"; done
  log "S2 $tag done in $(( $(date +%s)-t0 ))s"; for i in 1 2 3 4; do log "   $(tail -1 "$R/$tag-s2-fit$i.log" | cut -c1-200)"; done
  kill $smp 2>/dev/null
  docker logs --since "$since" "$LANE" > "$R/$tag-stress-server.log" 2>&1
  local errs; errs=$(grep -cE "$FF" "$R/$tag-stress-server.log")
  local w1 tr1 peak up hc; w1=$(w2); tr1=$(trips); peak=$(sort -n "$R/$tag-gpumem.txt" | tail -1)
  is_up "$LANE" && is_up "$RTXC" && up=1 || up=0
  hc=$(curl -s -m 60 $U/v1/chat/completions -H 'content-type: application/json' -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"Reply with the word OK."}],"max_tokens":16,"chat_template_kwargs":{"thinking":false}}' | grep -c '"finish_reason"')
  echo "{\"errs\": $errs, \"w2_before\": \"$w0\", \"w2_after\": \"$w1\", \"trips_before\": $tr0, \"trips_after\": $tr1, \"gpu_peak_mib\": ${peak:-0}, \"up\": $up, \"health\": $hc}" > "$R/$tag-stress-checks.json"
  log "STRESS CHECKS $tag: $(cat "$R/$tag-stress-checks.json") $([[ $errs -gt 0 ]] && tr '\r' '\n' < "$R/$tag-stress-server.log" | grep -E "$FF" | head -2 | cut -c1-200 | tr '\n' '|')"; }

judge(){ python3 - "$R" "$1" "${2:-}" "${3:-0}" <<'PY'
import json,sys
R,t,base,kv=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4] or 0)
g={}
def L(p):
    try: return json.load(open(p))
    except Exception as e: return {"_err":str(e)}
s1=L(f"{R}/{t}-s1-fit.json"); n=s1.get("needle",{})
g["S1_1m"]=n.get("http")==200 and (n.get("prompt_tokens") or 0)>=950000 and bool(n.get("recall")) and s1.get("post_probe",{}).get("ok",False)
c16=L(f"{R}/{t}-s1-c16.json").get("summary",{}); g["S1_c16"]=c16.get("ok")==c16.get("np") and (c16.get("np") or 0)>0
import re
m=re.search(r"prompt_tokens=(\d+)",open(f"{R}/{t}-s1-cold128k.txt").read()); g["S1_128k"]=bool(m) and int(m.group(1))>=120000
s2=[L(f"{R}/{t}-s2-fit{i}.json").get("needle",{}) for i in (1,2,3,4)]
g["S2_4x1m"]=all(x.get("http")==200 and (x.get("prompt_tokens") or 0)>=950000 and x.get("recall") for x in s2)
ck=L(f"{R}/{t}-stress-checks.json")
g["no_errors"]=ck.get("errs")==0; g["no_trip"]=ck.get("trips_after")==ck.get("trips_before")
g["no_timeouts"]=ck.get("w2_after")==ck.get("w2_before") and ck.get("w2_after")!="NA"; g["alive"]=ck.get("up")==1 and ck.get("health",0)>=1
sp=L(f"{R}/{t}-speed.txt") if False else json.loads(open(f"{R}/{t}-speed.txt").read())
out={"tag":t,"kv_tokens":kv,"speed":sp,"s2_ttft":[x.get("ttft_s") for x in s2],"s2_pt":[x.get("prompt_tokens") for x in s2],"gpu_peak_mib":ck.get("gpu_peak_mib")}
if base:
    b=json.loads(open(f"{R}/{base}-speed.txt").read())
    out["c1_ratio"]=round(sp["c1_mean"]/b["c1_mean"],4); out["c16_ratio"]=round(sp["c16_agg"]/b["c16_agg"],4)
    g["speed_c1"]=out["c1_ratio"]>=0.98; g["speed_c16"]=out["c16_ratio"]>=0.97; g["kv"]=kv>=4850000
out["gates"]=g; out["result"]="PASS" if all(g.values()) else "FAIL"
json.dump(out,open(f"{R}/{t}-verdict.json","w"),indent=1)
print(out["result"], json.dumps({k:("PASS" if v else "FAIL") for k,v in g.items()}), "c1_ratio",out.get("c1_ratio"),"c16_ratio",out.get("c16_ratio"),"kv",kv,"peak",out["gpu_peak_mib"])
PY
}

log "===== CARD KV2 START (A: stress live 8.5 GiB; B: R1b 9.5 GiB) ====="
o=$(ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh)' | grep -v cardKV2_runner.sh | grep -v grep | cut -c1-120 | tr '\n' '|')
[[ -n $o ]] && { log "ABORT: other runner live: $o"; exit 4; }
f=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '); [[ -n ${f// /} ]] && { log "ABORT: foreign container: $f"; exit 4; }
is_up "$LANE" && is_up "$RTXC" && bound || { log "ABORT: lane not serving"; exit 5; }
names | grep -qx "$K85" && { log "ABORT: $K85 exists"; exit 5; }
docker logs "$LANE" > "$R/A-boot.log" 2>&1   # never `docker logs | grep -q` under pipefail (SIGPIPE -> false negative)
KVA=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$R/A-boot.log" | tail -1 | tr -dc '0-9')
grep -q "reserved 8.5 GiB memory for KV Cache" "$R/A-boot.log" || { log "ABORT: live lane is not the 8.5 GiB config (KV=$KVA)"; exit 5; }
printf "OWNER: Milo cardKV2 since %s (A: stress on live lane; B: R1b 9.5 GiB boot). Authorized by James 2026-10-22 ~19:10.\nDo NOT start/stop :30006 containers while live. Rollback: %s (+-rtx) after B starts; v25 kv4.9-KEPT; v23 RETIRED-REF.\n" "$(date '+%F %T %Z')" "$K85" > $D/LANE-OWNER.txt

# ---------------- A: live 8.5 lane, no reboot ----------------
state A_SPEED; log "A: KV $KVA tokens, watchdog armed=$(pgrep -f sidecar_watchdog.py >/dev/null && echo yes || echo NO)"
speed A
state A_STRESS; stress A
if ! (is_up "$LANE" && bound); then log "A: lane DOWN after stress (trip or crash)"; recover_pair; fi
AV=$(judge A "" "$KVA"); log "A VERDICT: $AV"
if [[ $AV != PASS* ]]; then
  state A_FAIL; log "A failed: 8.5 GiB lane stays as is (it is production); B skipped. James decides."
  printf "Lane = v25 sidecar kv8.5 (unchanged). cardKV2 A stress FAIL: %s\n" "$AV" > $D/LANE-OWNER.txt
  { echo; echo "## $(date '+%F %H:%M %Z') — Card KV2 — A (stress, live 8.5 GiB) FAIL — B skipped"; echo "$AV"; echo "Receipts: $R"; } >> $LEDGER
  touch "$R/DONE"; exit 0; fi

# ---------------- B: R1b 9.5 GiB ----------------
state B_BOOT; disarm_wd
log "stop-and-keep $LANE -> $K85 (LANE DARK begins)"
docker stop "$LANE" "$RTXC" >/dev/null; docker rename "$LANE" "$K85"; docker rename "$RTXC" "$K85-rtx"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true
t0=$(date +%s)
restore85(){ log "RESTORE 8.5: $1"; docker logs "$LANE" > "$R/B-fail-server.log" 2>&1
  docker kill "$LANE" >/dev/null 2>&1; docker stop "$RTXC" >/dev/null 2>&1
  docker rename "$LANE" dsv41-vllm-1017-v25-kv9.5-FAIL 2>/dev/null; docker rename "$RTXC" dsv41-vllm-1017-v25-kv9.5-FAIL-rtx 2>/dev/null
  docker rename "$K85" "$LANE"; docker rename "$K85-rtx" "$RTXC"; docker start "$RTXC" >/dev/null; sleep 20; docker start "$LANE" >/dev/null
  wait_bind 20 && log "RESTORE bound: 8.5 GiB lane serving" || log "RESTORE FAILED — NEEDS HANDS (kv4.9-KEPT and v23 RETIRED-REF exist)"
  arm_wd; printf "Lane = v25 sidecar kv8.5 (restored). cardKV2 B (9.5 GiB) FAIL: %s\n" "$1" > $D/LANE-OWNER.txt
  { echo; echo "## $(date '+%F %H:%M %Z') — Card KV2 — A PASS ($AV) — B (9.5 GiB) FAIL: $1 — 8.5 restored"; echo "Receipts: $R"; } >> $LEDGER
  state B_FAIL; touch "$R/DONE"; exit 3; }
NAME=$LANE ROWMAP=$ROWMAP bash "$LAUNCH" sidecar || restore85 "sidecar launch failed"
PEER_SPINS=2000000000 REPLAY_OVERLAY=1 EXTRA="--kv-cache-memory $KV95" MEGA_PEER_CHECK=0 MEGA_COLD_TRT=0 \
  NAME=$LANE ROWMAP=$ROWMAP MAXLEN=$MAXLEN bash "$LAUNCH" server || restore85 "server launch failed"
wait_bind 45 || restore85 "no bind / died in boot"
log "LANE DARK ends after $(( $(date +%s)-t0 ))s"
docker logs "$LANE" > "$R/B-boot.log" 2>&1
KVB=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$R/B-boot.log" | tail -1 | tr -dc '0-9')
log "B FACTS: kv=$KVB $(grep -oE 'Maximum concurrency for 1,048,576 tokens per request: [0-9.]+x' "$R/B-boot.log" | tail -1) reserved=$(grep -c 'reserved 9.5 GiB memory for KV Cache' "$R/B-boot.log") peer2=$(grep -c 'peer mode 2, check 0' "$R/B-boot.log") rowmap40=$(grep -c "MEGA_PEER rowmap /w/$ROWMAP: 40 layers" "$R/B-boot.log") autotune_loaded=$(grep -oE 'Loaded [0-9]+ configs' "$R/B-boot.log" | head -1)"
grep -q 'reserved 9.5 GiB memory for KV Cache' "$R/B-boot.log" || restore85 "9.5 GiB reservation line missing"
grep -q 'peer mode 2, check 0' "$R/B-boot.log" || restore85 "peer mode 2 missing"
arm_wd
state B_SPEED; speed B
state B_STRESS; stress B
if ! (is_up "$LANE" && bound); then restore85 "lane down after stress (trip or crash)"; fi
BV=$(judge B A "$KVB"); log "B VERDICT: $BV"
[[ $BV == PASS* ]] || restore85 "gates: $BV"
printf "Lane = v25 sidecar with --kv-cache-memory 9.5 GiB (%s tokens). Rollback: %s (+-rtx, 8.5 GiB), then kv4.9-KEPT, then v23 RETIRED-REF. Fail-stop watchdog armed.\nRelaunch scripts default EXTRA=\"\": pass EXTRA=\"--kv-cache-memory %s\".\n" "$KVB" "$K85" "$KV95" > $D/LANE-OWNER.txt
{ echo; echo "## $(date '+%F %H:%M %Z') — Card KV2 — A (live 8.5 GiB stress) PASS — B (9.5 GiB) PASS, KEPT"; echo "A: $AV"; echo "B: $BV"; echo "Receipts: $R"; } >> $LEDGER
state B_PASS_KEPT; log "===== CARD KV2 DONE (9.5 kept) ====="; touch "$R/DONE"

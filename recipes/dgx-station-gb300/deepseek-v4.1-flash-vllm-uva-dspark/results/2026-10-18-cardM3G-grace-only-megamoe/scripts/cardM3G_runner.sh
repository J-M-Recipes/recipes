#!/usr/bin/env bash
# cardM3G_runner.sh — Card M3G: el8 "M3" MegaMoE hook, GRACE-ONLY, vs our v22 lane, same window, bracketed.
# 2026-10-17 ~23:10 CDT, Milo (hermes milo). James: "1) yes 2) downtime is fine; the station is yours 3) add [seqs32]".
# Arms (in order):  CTL1 = live v22 REF as-is (no reboot)  ->  M3G-mix (el8 hook, rowmap-mix-v1 285/99, af7f9488+#58132 overlay,
#   32 seqs, k 5/3/3 probabilistic, LL_GEMM)  ->  M3G-ours (same, rowmap-static-v1 295/89)  ->  V22S32 (v22 launch, seqs 32,
#   ksched [[1,8,5],[9,32,1]]; new autotune hash)  ->  CTL2 = v22 REF restarted and LEFT SERVING.
# Per arm: bind -> facts -> Ronan real-text harness (C1 x3 / C4 / C16 / C32, nonce, 0% prefix) -> run_window_T1T2 (knee r1/r2,
#   agent fixture, replay x2) -> gauntlet. Stop-and-keep only. Never :30003. Refuses foreign GPU containers.
# PRE-REGISTERED promotion bar (any M3G arm vs mean of CTL1/CTL2 on the same harness): C1 >= -3% AND C16 >= +5% AND
#   median TTFT(8K) <= 0.45 s AND gauntlet 12/12 AND replay tool-turns not below controls; GSM8K-200 run by hand before promotion.
#   V22S32 bar: C1/C4/C16 within -3% of controls AND C32 decode_agg > CTL C32 (which queues 8 of 32).
set -uo pipefail
D=$HOME/dsv41/m3; R=$D/results-cardM3G; mkdir -p "$R" "$D/logs"
B=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v22-ksched1to8-BOUND-REF
GDIR=$HOME/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 5; }
wait_bind(){ local n=$1 i
  for ((i=0;i<360;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $((i*10))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n"; return 1; }
    sleep 10
  done; log "TIMEOUT $n (60 min)"; return 1; }
rt_bench(){ # tag  — Ronan real-text harness, same slices as the 22:28 run; nonce per arm
  local tag=$1 N="m3g-$1-$(date +%s)" O="$R/rt-$1"; mkdir -p "$O"
  for r in 1 2 3; do python3 $B $U $M $P/eval-c1-r$r-try1.jsonl $O/c1-r$r.json 1 --warm $P/eval-c1-r$r-try1-warm.jsonl --nonce $N-c1-r$r; done
  python3 $B $U $M $P/eval-c16-try1.jsonl $O/c16.json 16 --warm $P/eval-c16-try1-warm.jsonl --nonce $N-c16
  python3 $B $U $M $P/eval-c16-try1.jsonl $O/c4.json 4 --head 8 --nonce $N-c4
  python3 $B $U $M $P/eval-c16-try1.jsonl $O/c32.json 32 --nonce $N-c32
  log "RT $tag: $(python3 - "$O" <<'PY'
import json,sys,glob,os
O=sys.argv[1]; r={}
for f in sorted(glob.glob(O+'/*.json')):
    try: d=json.load(open(f)); r[os.path.basename(f)[:-5]]=(d['decode_agg'],d.get('median_ttft_s'),d.get('acceptance_len'),d.get('prefix_hit_pct'))
    except Exception as e: r[os.path.basename(f)]=str(e)
c1=[r[k][0] for k in ('c1-r1','c1-r2','c1-r3') if isinstance(r.get(k),tuple)]
print('C1mean=%.1f'%(sum(c1)/len(c1)) if c1 else 'C1=?', {k:v for k,v in r.items()})
PY
)"
}
measure(){ # tag name
  local tag=$1 name=$2
  docker logs "$name" 2>&1 | grep -E "MEGA_PEER patched|hot experts in MegaMoE|cold experts in TRT-LLM layout|Decoder SWA bounded replay|PIN_HOT installed|PIN_HOT rehome done|Loaded [0-9]+ configs|GPU KV cache size|Maximum concurrency|MemAvailable|model loading took" > "$R/facts-$tag.txt"
  log "FACTS $tag: $(grep -cE 'hot experts in MegaMoE' "$R/facts-$tag.txt") mega-layer lines; replay=$(grep -c 'Decoder SWA bounded replay' "$R/facts-$tag.txt"); $(grep -m1 'GPU KV cache size' "$R/facts-$tag.txt" | sed 's/.*INFO//')"
  free -g | sed -n 2p | awk '{print "HOST used="$3" avail="$7}' | sed "s/^/$(date '+%F %T') $tag /"
  rt_bench "$tag"
  bash $HOME/dsv41/run_window_T1T2.sh "M3G-$tag" > "$R/nohup-$tag.log" 2>&1
  mv $HOME/dsv41/overnight-T1T2/*"M3G-$tag"* "$R"/ 2>/dev/null
  log "WINDOW $tag: $(grep -h 'conc=' "$R/window-M3G-$tag.log" 2>/dev/null | grep -E 'r2\]' | tr '\n' ';' | cut -c1-400)"
  rm -rf "$R/gauntlet-$tag"
  (cd "$GDIR" && python3 run_suite.py --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "m3g-$tag" --cache-salt "m3g-$tag" --sandbox-image "$GIMG" --workers 4) > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print({k:d[k] for k in d if 'count' in k or 'pass' in k or 'total' in k})" 2>/dev/null)"
  docker logs "$name" 2>&1 | grep -cE "Traceback|ERROR" > "$R/errcount-$tag.txt"
}
boot_m3(){ # tag rowmap
  local tag=$1 name=dsv41-vllm-M3G-$1-EXP t0
  stop_all; drop_caches
  log "strays: $(ps -eo pid,rss,args --sort=-rss | sed -n 2,4p | cut -c1-120 | tr '\n' '|')"
  docker rm -f "$name" >/dev/null 2>&1 || true
  t0=$(date +%s)
  NAME=$name ROWMAP=$2 bash $D/launch_m3g.sh || { log "launch failed $tag"; return 1; }
  if ! wait_bind "$name"; then docker logs "$name" > "$R/boot-$tag.log" 2>&1; docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-BOOT-FAIL"; return 1; fi
  log "boot->ready $tag $(( $(date +%s) - t0 ))s"
  docker logs "$name" > "$R/boot-$tag.log" 2>&1
  grep -q "hot experts in MegaMoE" "$R/boot-$tag.log" || { log "ARM $tag: hook did NOT patch MegaMoE — invalid arm"; return 1; }
  measure "$tag" "$name"
}

log "===== CARD M3G START ====="
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running for CTL1"; exit 5; }
for f in config/cache.py models/deepseek_v41/decoder_replay_layers.py models/deepseek_v41/nvidia/model.py models/deepseek_v41/nvidia/model_state.py models/deepseek_v41/nvidia/vl_model.py; do
  [[ -f $D/overlay-58132/tree/vllm/$f ]] || { log "ABORT: overlay missing $f"; exit 6; }; done
docker image inspect vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063 >/dev/null 2>&1 || { log "ABORT: af7f9488 image not present"; exit 7; }

# ---- CTL1: the live lane, as-is ----
measure CTL1 "$REF"

# ---- M3G arms ----
boot_m3 mix rowmap-mix-v1.json || log "M3G-mix FAILED"
boot_m3 ours rowmap-static-v1.json || log "M3G-ours FAILED"

# ---- V22S32 side-arm: v22 launch with 32 seqs ----
stop_all; drop_caches
S32=dsv41-vllm-1017-v22-seqs32-EXP; docker rm -f $S32 >/dev/null 2>&1 || true
t0=$(date +%s)
sed -e 's/--max-num-seqs 24/--max-num-seqs 32/' -e 's/\[\[1,8,5\],\[9,24,1\]\]/[[1,8,5],[9,32,1]]/' $HOME/dsv41/1017/launch_v22.sh > $D/launch_v22_seqs32.sh
grep -q 'max-num-seqs 32' $D/launch_v22_seqs32.sh && grep -q '9,32,1' $D/launch_v22_seqs32.sh || log "WARN: seqs32 sed did not apply"
bash $D/launch_v22_seqs32.sh $S32
if wait_bind $S32; then log "boot->ready V22S32 $(( $(date +%s) - t0 ))s (new autotune hash expected)"; docker logs $S32 > "$R/boot-V22S32.log" 2>&1; measure V22S32 $S32
else docker logs $S32 > "$R/boot-V22S32.log" 2>&1; docker stop $S32 >/dev/null 2>&1; docker rename $S32 $S32-BOOT-FAIL; log "V22S32 FAILED"; fi

# ---- CTL2: v22 REF back, left serving ----
stop_all; drop_caches
t0=$(date +%s); docker start "$REF" >/dev/null
if wait_bind "$REF"; then log "boot->ready CTL2 $(( $(date +%s) - t0 ))s"; measure CTL2 "$REF"; else log "CTL2 REF FAILED TO BIND — lane dark, needs hands"; fi
log "===== CARD M3G END (lane = $REF) ====="
grep -hE ' (RT|WINDOW|GAUNTLET|FACTS|boot->ready) ' "$R/runner.log" > "$R/SUMMARY.txt"

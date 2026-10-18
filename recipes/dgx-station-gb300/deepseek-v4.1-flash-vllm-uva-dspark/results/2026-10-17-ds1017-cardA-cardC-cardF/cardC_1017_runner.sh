#!/usr/bin/env bash
# Card C — k-schedule recheck on DeepSeek-V4.1-Flash-1017 (new drafter: shell/tool accept 91/88 -> 80/77, prose 35 -> 25).
# 2026-10-17, Milo (hermes milo) for James ("1) go" 08:45 CDT).
# CTL1 = window on the running dsv41-vllm-1017A-LIVE (ksched [[1,4,5],[5,24,1]])
# CAND1 = [[1,2,5],[3,24,1]]  (k=5 only at C1-2, k=1 from C3; v13-style)
# CAND2 = [[1,4,3],[5,24,1]]  (k=3 at C1-4, k=1 from C5)
# CTL2 = docker start LIVE + window (bracket). Stop-and-keep only. Never :30003. Ends with LIVE up.
# num_speculative_tokens stays 5 so the kernel-shape hash should hit; if a boot shows no "Loaded N configs" it is a live tune (~75 min) and is logged.
set -uo pipefail
D=/home/milo/dsv41/1017; R=$D/results-cardC; mkdir -p "$R"
SHA=1bae0e1088993b9b82de08f59de5be5a34e42170
NEW=/models/DeepSeek-V4.1-Flash-1017-$SHA
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
HOOK=/home/milo/pin-hot-experts/v19/hook
LIVE=dsv41-vllm-1017A-LIVE
LEDGER=/home/milo/dsv41/results/ledger.md
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
wait_bind(){ local n=$1 i
  for ((i=0;i<720;i++)); do
    curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva && { log "BOUND $n after $((i*10))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n"; return 1; }
    sleep 10
  done; log "TIMEOUT $n"; return 1; }
window(){ local tag=$1 name=$2
  docker logs "$name" > "$R/bootlog-$tag.txt" 2>&1
  grep -E "PIN_HOT rehome done|Loaded [0-9]+ configs|GPU KV cache size|num_speculative_tokens_per_batch_size" "$R/bootlog-$tag.txt" > "$R/facts-$tag.txt"
  grep -q "Loaded [0-9]* configs" "$R/facts-$tag.txt" && log "ARM $tag: autotune HASH HIT" || log "ARM $tag: NO cache-load line (live tune?)"
  bash /home/milo/dsv41/run_window_T1T2.sh "DS1017C-$tag" > "$R/nohup-$tag.log" 2>&1
  mv /home/milo/dsv41/overnight-T1T2/*"DS1017C-$tag"* "$R"/ 2>/dev/null
  log "ARM $tag knee: $(grep -h 'conc=' "$R/window-DS1017C-$tag.log" | grep -E 'r2\]' | grep -E 'conc= (1|2|4|8):|conc=16:' | sed -E 's/.*conc= ?([0-9]+):\s+([0-9.]+) agg.*/C\1=\2/' | tr '\n' ' ')"
  log "ARM $tag fixture: $(grep -hE "^\[DS1017C-$tag\] (prose|structured|code|shell_ops|tool_json)" "$R/window-DS1017C-$tag.log" | sed -E 's/\[[^]]+\]\s+(\w+)\s+([0-9.]+) tok\/s\s+accept=([0-9.]+)%.*/\1 \2(\3%)/' | tr '\n' ' ')"
  log "ARM $tag replay: $(grep -h 'REPLAY n=4' "$R/window-DS1017C-$tag.log" | sed -E 's/.*agg ([0-9.]+) tok\/s.*accept=([0-9.]+)%.*emitted=([0-9]+).*/agg \1 acc \2% toolturns \3/' | tr '\n' ';')"
}
cand(){ # tag ksched
  local tag=$1 ks=$2 name=dsv41-vllm-1017C-$1
  stop_all; drop_caches
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --gpus all --ipc host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
    -v "$NEW":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v /home/milo/dsv41/vllm-cache:/root/.cache/vllm \
    -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro \
    -v /home/milo/pin-hot-experts/v16:/home/milo/pin-hot-experts/v16 \
    -e VLLM_LOGGING_LEVEL=INFO -e PIN_MODE=split -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP=/w/rowmap-static-v1.json \
    "$IMAGE" \
    --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
    --offload-backend uva --cpu-offload-gb 60 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
    --engram-config '{"cpu_offload": true}' \
    --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
    --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
    --long-prefill-token-threshold 6144 \
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$ks}" \
    --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 \
    --host 0.0.0.0 --port 30006 >/dev/null
  local t0; t0=$(date +%s)
  log "launched $name ksched=$ks"
  if ! wait_bind "$name"; then docker logs "$name" > "$R/boot-$tag.log" 2>&1; docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-BOOT-FAIL"; return 1; fi
  log "boot->ready $tag $(( $(date +%s) - t0 ))s"
  docker logs "$name" > "$R/bootlog-$tag.txt" 2>&1
  grep -q "rehome done" "$R/bootlog-$tag.txt" || { log "ARM $tag: hook did not re-home — invalid"; return 1; }
  window "$tag" "$name"
}
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
log "===== CARD C RESUME (runner v2: file-based hook check; v1 false-flagged CAND1 via pipefail/SIGPIPE) ====="
# CAND2 container was launched by runner v1 and is booting; adopt it.
C2=dsv41-vllm-1017C-CAND2; C1=dsv41-vllm-1017C-CAND1
if wait_bind "$C2"; then
  docker logs "$C2" > "$R/bootlog-CAND2.txt" 2>&1
  grep -q "rehome done" "$R/bootlog-CAND2.txt" && window CAND2 "$C2" || log "ARM CAND2: hook did not re-home — invalid"
else log "CAND2 boot failed"; fi
stop_all; drop_caches
docker start "$C1" >/dev/null; if wait_bind "$C1"; then
  docker logs "$C1" > "$R/bootlog-CAND1.txt" 2>&1
  grep -q "rehome done" "$R/bootlog-CAND1.txt" && window CAND1 "$C1" || log "ARM CAND1: hook did not re-home — invalid"
else log "CAND1 restart failed"; fi
stop_all; drop_caches
docker start "$LIVE" >/dev/null; wait_bind "$LIVE" && window CTL2 "$LIVE" || log "CTL2: LIVE restart failed"
{ echo; echo "## $(date '+%F %H:%M CDT') — Card C: k-schedule recheck on 1017 (CTL=[[1,4,5],[5,24,1]] on LIVE; CAND1=[[1,2,5],[3,24,1]]; CAND2=[[1,4,3],[5,24,1]])"; echo "worker: Milo (hermes milo) · James: '1) go' 2026-10-17 08:45"; grep -E "ARM .*(knee|fixture|replay|HASH|cache-load)|boot->ready" "$R/runner.log"; echo "receipts $R · dsv41-vllm-1017C-* stopped-and-kept · LIVE restarted (schedule unchanged, no promotion)"; } >> "$LEDGER"
log "===== CARD C END; LIVE up ====="

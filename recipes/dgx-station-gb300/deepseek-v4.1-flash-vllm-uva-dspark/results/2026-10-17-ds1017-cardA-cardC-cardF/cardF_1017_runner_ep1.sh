#!/usr/bin/env bash
# Card F — vLLM v0.31.0 spike on DeepSeek-V4.1-Flash-1017: deep_gemm_mega_moe + FLASHMLA_MEGA_ATTN_DSV41, hook OFF, UVA off60.
# 2026-10-17, Milo (hermes milo) for James ("go" 09:4x CDT). Plan: work/dsv41-optimization/PLAN-1017-2026-10-17.md §Card F.
# F1: does mega_moe bind with host-pinned (UVA) experts?  F2: knee/fixture/replay hook-off vs 0909 hook-off (C1 97) and LIVE (192).
# Fresh autotune cache dir (new image). First boot 75-90 min expected. Stop-and-keep only. Never :30003.
# On boot failure: capture logs, keep container as -BOOT-FAIL, restart LIVE (lane must not stay dark on a dead spike).
# On success: leave F serving on :30006 for F3 (tf_logprob/BFCL, run by Milo), LIVE stays stopped-and-kept.
set -uo pipefail
D=/home/milo/dsv41/1017; R=$D/results-cardF; mkdir -p "$R"
SHA=1bae0e1088993b9b82de08f59de5be5a34e42170
NEW=/models/DeepSeek-V4.1-Flash-1017-$SHA
IMAGE=vllm/vllm-openai:v0.31.0
CACHE=/home/milo/dsv41/vllm-cache-v0310; mkdir -p "$CACHE"
LIVE=dsv41-vllm-1017A-LIVE
NAME=dsv41-vllm-1017F-v0310-megamoe-ep1-hookoff-EXP
LEDGER=/home/milo/dsv41/results/ledger.md
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
log "===== CARD F START (attempt 2: +--enable-expert-parallel; attempt 1 died: MegaMoE requires expert parallel) ====="
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: image $IMAGE missing"; exit 2; }
stop_all; drop_caches
log "MemAvailable $(awk '/MemAvailable/{printf "%.1f GiB", $2/1048576}' /proc/meminfo); archive copy running: $(pgrep -f archive_1017.sh >/dev/null && echo yes || echo no)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  -v "$NEW":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v "$CACHE":/root/.cache/vllm \
  -e VLLM_LOGGING_LEVEL=INFO -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --tokenizer-mode deepseek_v41 \
  --offload-backend uva --cpu-offload-gb 60 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' \
  --kernel-config '{"moe_backend":"deep_gemm_mega_moe"}' --enable-expert-parallel \
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41"}' \
  --kv-cache-dtype fp8_ds_mla \
  --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,24,1]]}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,32,40,48,64,96,128]}' \
  --host 0.0.0.0 --port 30006 >/dev/null
t0=$(date +%s); log "launched $NAME image=$IMAGE"
fail=""
for ((i=0;i<240;i++)); do   # 240 x 30 s = 120 min
  if curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $NAME after $(( $(date +%s)-t0 ))s"; break; fi
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { fail="DIED"; break; }
  if docker logs "$NAME" 2>&1 | grep -qE "ValueError:|Traceback|RuntimeError:|EngineCore failed|Engine core initialization failed"; then fail="FASTFAIL"; break; fi
  (( i % 10 == 0 )) && log "waiting $((i/2)) min: $(docker logs "$NAME" 2>&1 | grep -E "Loading safetensors|Autotun|autotun|Capturing|KV cache size|mega|Mega|offload" | tail -1 | cut -c1-160)"
  sleep 30
done
docker logs "$NAME" > "$R/bootlog.txt" 2>&1
grep -iE "mega_moe|MegaMoE|deep_gemm|MEGA_ATTN|FLASHMLA|attention backend|moe_backend|offload|Loaded [0-9]+ configs|KV cache size|Maximum concurrency|autotun" "$R/bootlog.txt" | head -60 > "$R/facts.txt"
if [[ -n $fail ]] || ! curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then
  [[ -z $fail ]] && fail="TIMEOUT"
  log "F1 RESULT: $fail — $(grep -E "ValueError:|RuntimeError:|Error|error" "$R/bootlog.txt" | grep -vE "^\s*$" | tail -3 | cut -c1-300 | tr '\n' '|')"
  docker stop "$NAME" >/dev/null 2>&1; docker rename "$NAME" "$NAME-BOOT-FAIL"
  drop_caches; docker start "$LIVE" >/dev/null; log "LIVE restarted (rollback after failed spike)"
  { echo; echo "## $(date '+%F %H:%M CDT') — Card F: v0.31.0 + deep_gemm_mega_moe + MEGA_ATTN on 1017, hook off, UVA off60 — F1 $fail"; echo "worker: Milo (hermes milo) · James: go"; grep -E "F1 RESULT|launched|waiting" "$R/runner.log" | tail -6; echo "receipts $R · $NAME-BOOT-FAIL kept · LIVE restarted"; } >> "$LEDGER"
  log "===== CARD F END (F1 $fail) ====="; exit 1
fi
log "F1 RESULT: BOUND with UVA off60 — facts: $(grep -iE "mega|deep_gemm|MEGA_ATTN|Loaded [0-9]+ configs|KV cache size" "$R/facts.txt" | cut -c1-140 | tr '\n' '|' | cut -c1-600)"
bash /home/milo/dsv41/run_window_T1T2.sh "DS1017F-hookoff" > "$R/nohup-window.log" 2>&1
mv /home/milo/dsv41/overnight-T1T2/*"DS1017F-hookoff"* "$R"/ 2>/dev/null
log "F2 knee: $(grep -h 'conc=' "$R/window-DS1017F-hookoff.log" | grep -E 'r2\]' | grep -E 'conc= (1|2|4|8):|conc=16:' | sed -E 's/.*conc= ?([0-9]+):\s+([0-9.]+) agg.*/C\1=\2/' | tr '\n' ' ')"
log "F2 fixture: $(grep -hE "^\[DS1017F-hookoff\] (prose|structured|code|shell_ops|tool_json)" "$R/window-DS1017F-hookoff.log" | sed -E 's/\[[^]]+\]\s+(\w+)\s+([0-9.]+) tok\/s\s+accept=([0-9.]+)%.*/\1 \2(\3%)/' | tr '\n' ' ')"
log "F2 replay: $(grep -h 'REPLAY n=4' "$R/window-DS1017F-hookoff.log" | sed -E 's/.*agg ([0-9.]+) tok\/s.*accept=([0-9.]+)%.*emitted=([0-9]+).*/agg \1 acc \2% toolturns \3/' | tr '\n' ';')"
docker logs "$NAME" 2>&1 | grep -cE "Traceback|ERROR" > "$R/errcount.txt"
{ echo; echo "## $(date '+%F %H:%M CDT') — Card F: v0.31.0 + deep_gemm_mega_moe + MEGA_ATTN on 1017, hook off, UVA off60"; echo "worker: Milo (hermes milo) · James: go"; grep -E "F1 RESULT|F2 |BOUND" "$R/runner.log"; echo "reference: 0909 hook-off C1 97 (v14 positional) · LIVE 0909 hook-on C1 192 · receipts $R · $NAME left serving on :30006 for F3; LIVE stopped-and-kept"; } >> "$LEDGER"
log "===== CARD F END (F up for F3) ====="

#!/usr/bin/env bash
# Card KV1 (2026-11-01, evening; James: "you have the station").
# GLM-5.3-Flash on the GB300: is the daily lane KV-starved, and which flag-only lever fixes it?
# Research bundle: ~/hermes/research/glm53-flash-kv-sidecar-20261101/ (M4). No sidecar arm: research + Fable 5.1 review closed it.
# All arms: daily image glmf-sglang:0922-582389ce-revert39688-tf5.16.1, nvidia NVFP4 09b04e5e, incoai DFlash2 7d74, block 7,
#           fp8 target KV, ctx 1M, maxbs 16, ReplaySSM. GB300 pinned by UUID (RTX is now a visible GPU). One axis per arm vs ctl.
#   ctl   : daily recipe as-is (draft backend fa4 -> draft KV bf16)                         [same-window control]
#   T     : --speculative-draft-attention-backend triton  (draft KV follows target -> fp8)  [pool lever]
#   F     : flashinfer draft backend — ONLY if T fails to boot or fails tools
#   L     : --mamba-radix-cache-strategy extra_buffer_lazy  (5 -> 4 slots/req, cap 9 -> 12) [concurrency lever]
#   M     : --mem-fraction-static 0.90                                                      [pool lever]
#   COMBO : best draft backend of T/F + L + M      (candidate config; measured, not promoted)
#   ctl2  : ctl again (drift bracket)
# Gates per arm: boot pool lines, vision, greedy vs W2 + vs C1 ref, TF vs v0.5.20 ctrl, tools, C1 low, DFlash acceptance, knee C8,
#                kvcap_probe (6x200K, 12x64K, 1x960K needle) with server-side running/queue/usage/retract scrape.
# Stop-and-keep everything. NOTHING restored at the end (James's standing rule) — ds41 + ds41-sidecar stay stopped-and-kept.
set -uo pipefail
cd ~/glmf
D=~/glmf/cardKV1-2026-11-01; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
CKPT=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DR=/models/GLM-5.3-Flash-DFlash2
KNOBS="--max-mamba-cache-size 48 --speculative-dflash-block-size 7 --enable-linear-replayssm-spec"; ENV_BASE="-e TORCHINDUCTOR_COMPILE_THREADS=1"
TF=~/window-20260913/scripts/tf_noninferiority.py; CTRL_TF=~/glmf/w5-ctrl/tf-w5-ctrl.json
W2=~/glmf/w3/greedy-w2-a.json; C1REF=~/glmf/cardK2-2026-09-29/C1/greedy-C1.json
export BASE_URL=http://127.0.0.1:30001/v1 MODEL_NAME=glm-5.3-flash API_KEY=none
IMG=glmf-sglang:0922-582389ce-revert39688-tf5.16.1
STOPF=$D/STOP
stop_glmf(){ for c in $(docker ps --format '{{.Names}}' | grep '^glmf-' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 5; }
wait_ready(){ local i; sleep 5; docker ps -a --format "{{.Names}}" | grep -q "^$1$" || { log "NO_CONTAINER $1"; return 5; }
  for i in $(seq 1 540); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "ready $1 after $((i*5+5))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "CONTAINER_EXITED $1"; docker logs "$1" 2>&1 | tr '\r' '\n' | grep -nE "Error|Traceback|raise" | tail -8 | cut -c1-250; return 5; }; sleep 5; done
  log "READY_TIMEOUT $1"; return 5; }
gates(){ local TAG=$1 OUT=$D/$1 C=glmf-KV1-$1; export MODEL=glm-5.3-flash
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,8), reps=0)' > "$OUT/warm.txt" 2>&1
  python3 tf5161/vision_gate.py > "$OUT/vision.txt" 2>&1
  python3 greedy_equiv.py "$OUT/greedy-$TAG.json" > "$OUT/greedy-gen.txt" 2>&1 || log "GREEDY_GEN_FAIL $TAG"
  test -f "$OUT/greedy-$TAG.json" && { python3 greedy_equiv.py --compare "$W2" "$OUT/greedy-$TAG.json" > "$OUT/greedy-vs-w2.txt" 2>&1
    python3 greedy_equiv.py --compare "$C1REF" "$OUT/greedy-$TAG.json" > "$OUT/greedy-vs-C1.txt" 2>&1; }
  LANE="$TAG" BASE_URL=http://127.0.0.1:30001 python3 "$TF" "$W2" "$OUT/tf-$TAG.json" > "$OUT/tf-gen.txt" 2>&1 || log "TF_GEN_FAIL $TAG"
  test -f "$OUT/tf-$TAG.json" && python3 "$TF" --compare "$CTRL_TF" "$OUT/tf-$TAG.json" > "$OUT/tf-compare.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$OUT/tool-1.txt" 2>&1
  TAG="KV1-$TAG-low" EFFORT=low python3 c1_methods.py > "$OUT/c1-low.txt" 2>&1
  CONT=$C TAG="KV1-$TAG" EFFORT=low python3 accept_probe.py > "$OUT/accept.txt" 2>&1
  python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=3)' > "$OUT/knee-c8.txt" 2>&1
  unset MODEL
  POOL=$(grep -oE '#tokens: [0-9]+' "$OUT/boot-excerpt.txt" | head -1 | grep -oE '[0-9]+') CONT=$C timeout 3600 python3 kvcap_probe.py "$OUT/kvcap.json" > "$OUT/kvcap.txt" 2>&1 || log "KVCAP_FAIL $TAG rc=$?"
  log "ARM $TAG: pools [$(grep -oE 'KV Cache is allocated. dtype: [a-z0-9_.]+, #tokens: [0-9]+' $OUT/boot-excerpt.txt | sed 's/KV Cache is allocated. //' | tr '\n' ';')] cap [$(grep -oE 'capped to [0-9]+' $OUT/boot-excerpt.txt | head -1)] | greedy-vs-W2 $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' $OUT/greedy-vs-w2.txt 2>/dev/null) vs-C1 $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' $OUT/greedy-vs-C1.txt 2>/dev/null) | TF $(grep -oE '"mean": [0-9.e-]+|"p99": [0-9.e-]+' $OUT/tf-compare.txt 2>/dev/null | tr '\n' ' ') | vision $(grep -o 'VISION_GATE [A-Z]*' $OUT/vision.txt 2>/dev/null) | tools $(grep -ho 'tool_ok=[0-9/]*' $OUT/tool-1.txt) | C1low $(grep -h recipe-method $OUT/c1-low.txt | grep -oE 'median [0-9.]+') code $(grep -h 'code 400' $OUT/c1-low.txt | grep -oE 'median [0-9.]+') | C8 $(tail -1 $OUT/knee-c8.txt | cut -c1-90) | kvcap $(grep -h '^\[kvcap\] S' $OUT/kvcap.txt | sed -E 's/ ttft.*//' | tr '\n' ' ' | cut -c1-600)"; }
# arm <tag> <draft_backend> <mem> <extra>
arm(){ local tag=$1 be=$2 mem=$3 extra=$4
  [[ -f $STOPF ]] && { log "STOP file present; skipping $tag"; return 0; }
  mkdir -p "$D/$tag"; stop_glmf; unset MODEL
  local t0=$(date +%s)
  TAG="KV1-$tag" MODEL=$CKPT DRAFT=$DR IMAGE="$IMG" SPEC=dflash DRAFT_BACKEND=$be KV=fp8_e4m3 MEM=$mem CTX=1048576 MAXBS=16 EXTRA="$KNOBS $extra" DOCKER_ENV="$ENV_BASE" bash launch-glmf-kv.sh > "$D/$tag/launch.txt" 2>&1
  if wait_ready "glmf-KV1-$tag"; then
    log "boot->ready $tag $(( $(date +%s) - t0 ))s"
    docker logs "glmf-KV1-$tag" 2>&1 | tr '\r' '\n' | grep -iE "KV Cache is allocated|Mamba Cache|capped to|overriding KV|draft runner|replayssm|max_total_num_tokens|mem_fraction" | cut -c1-300 | head -40 > "$D/$tag/boot-excerpt.txt"
    gates "$tag"; return 0
  else log "BOOT FAIL $tag"; docker logs "glmf-KV1-$tag" 2>&1 | tr '\r' '\n' | tail -60 > "$D/$tag/bootfail-tail.txt"; return 1; fi; }
tools_ok(){ grep -q 'tool_ok=10/10' "$D/$1/tool-1.txt" 2>/dev/null; }

log "===== CARD KV1 (GLM-Flash KV capacity: draft-KV dtype, lazy KDA buffer, mem-fraction) START ====="
for c in ds41 ds41-sidecar; do
  if docker ps --format '{{.Names}}' | grep -qx "$c"; then
    docker inspect "$c" --format '{{.Name}} image={{.Config.Image}} started={{.State.StartedAt}} restart={{.HostConfig.RestartPolicy.Name}}' >> "$D/stopped-lanes.txt"
    log "stop+keep $c (James: you have the station; restore = docker start ds41-sidecar, then ds41 — not done by this card)"; docker stop "$c" >/dev/null
  fi
done
sleep 10
other=$(docker ps --format '{{.Names}}' | grep -vE "^glmf-" || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
docker image inspect "$IMG" >/dev/null 2>&1 || { log "NO IMAGE $IMG"; exit 3; }
test -f "$DR/model.safetensors" -a -f "$CKPT/config.json" || { log "NO MODEL/DRAFT"; exit 3; }
for f in "$TF" "$CTRL_TF" "$W2" "$C1REF" tf5161/vision_gate.py greedy_equiv.py tool_harness.py c1_methods.py accept_probe.py flash_bench.py kvcap_probe.py launch-glmf-kv.sh; do test -f "$f" || { log "MISSING $f"; exit 3; }; done
sha256sum kvcap_probe.py launch-glmf-kv.sh "$0" > "$D/script-sha.txt"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || log "drop_caches skipped (no sudo -n)"
for t in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1); [[ $u -lt 2048 ]] && break; sleep 5; done; log "GB300 HBM used before first arm: ${u} MiB"

arm ctl fa4 0.85 ""
arm T triton 0.85 ""
BE=fa4
if tools_ok T; then BE=triton
else log "T failed boot/tools -> conditional arm F (flashinfer)"; arm F flashinfer 0.85 "" && tools_ok F && BE=flashinfer; fi
log "COMBO draft backend = $BE"
arm L fa4 0.85 "--mamba-radix-cache-strategy extra_buffer_lazy"
arm M fa4 0.90 ""
arm COMBO "$BE" 0.90 "--mamba-radix-cache-strategy extra_buffer_lazy"
arm ctl2 fa4 0.85 ""
stop_glmf
{ echo; echo "## $(date '+%F %H:%M %Z') — Card KV1: GLM-Flash KV capacity (draft-KV dtype / lazy KDA buffer / mem 0.90) on the daily image"
  echo "worker: Milo (hermes milo, claude-opus-5-5) · James: 'you have the station' 2026-11-01 ~20:55 CST"
  grep -h "ARM \|BOOT FAIL\|boot->ready\|stop+keep ds41\|COMBO draft\|KVCAP_FAIL" "$D/runner.log" | cut -c1-900
  echo "receipts $D · glmf-KV1-* stopped-and-kept · ds41 + ds41-sidecar stopped-and-kept (restore: docker start ds41-sidecar; then ds41) · nothing restored, nothing promoted"; } >> ~/dsv41/results/ledger.md
log "===== CARD KV1 END ====="

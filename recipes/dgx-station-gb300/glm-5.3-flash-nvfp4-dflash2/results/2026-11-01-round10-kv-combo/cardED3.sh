#!/usr/bin/env bash
# Card ED3 = ED2 + --moe-runner-backend flashinfer_cutlass (ED2 recorded 5,208 all-zero steps: flashinfer_trtllm fuses routing and never calls select_experts, the recorder hook). Card ED rerun with the recorder inference-mode overlay (ED crashed: inplace update to inference tensor). (2026-11-01, queued behind Card KV1; James: "yes plz" to the routing-skew measurement).
# Question: how cold are GLM-5.3-Flash's cold experts? i.e. would a DSV4.1-style RTX PRO 6000 cold-expert band
# carry little work and rarely sit on the critical path, or would every decode step pay a PCIe round trip per MoE layer?
# One boot, arm-T config (daily image, triton draft backend -> fp8 draft KV, mem 0.85) + SGLang expert-distribution recorder
# (stat mode, unbounded per-step buffer). Measurement only: no hook, no RTX use, nothing promoted, nothing restored.
#   A = synthetic fixtures (ranking set): accept_probe (7 classes), c1_methods, tool_harness, knee C8 x1
#   B = Hermes transcript fixture replay (holdout, real agent turns)
set -uo pipefail
cd ~/glmf
D=~/glmf/cardED3-2026-11-01; mkdir -p "$D/dump" "$D/dumpA" "$D/dumpB"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
IMG=glmf-sglang:0922-582389ce-revert39688-tf5.16.1
CKPT=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DR=/models/GLM-5.3-Flash-DFlash2
KNOBS="--max-mamba-cache-size 48 --speculative-dflash-block-size 7 --enable-linear-replayssm-spec"
REC="--expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size -1 --moe-runner-backend flashinfer_cutlass"
U=http://127.0.0.1:30001; export BASE_URL=$U/v1 MODEL_NAME=glm-5.3-flash API_KEY=none
C=glmf-ED3-rec
log "===== CARD ED queued; waiting for Card KV1 runner to exit ====="
sleep 20; while pgrep -f "bash ~/glmf/cardPC2.sh" >/dev/null; do sleep 60; done
grep -q "CARD PC2 END" ~/glmf/cardPC2-2026-11-01/runner.log && log "PC ended normally" || log "WARNING: PC runner exited without END marker"
[[ -f $D/STOP ]] && { log "STOP file present; exiting"; exit 0; }
other=$(docker ps --format '{{.Names}}' || true); [[ -n $other ]] && { log "REFUSE running container(s): $other"; exit 4; }
for f in ed_replay.py ed_analyze.py accept_probe.py c1_methods.py tool_harness.py flash_bench.py launch-glmf-kv.sh ~/dsv41/transcript_fixture.json; do test -f "$f" || { log "MISSING $f"; exit 3; }; done
sha256sum ed_replay.py ed_analyze.py "$0" > "$D/script-sha.txt"
log "===== CARD ED START ====="
t0=$(date +%s)
TAG=ED3-rec MODEL=$CKPT DRAFT=$DR IMAGE=$IMG SPEC=dflash DRAFT_BACKEND=triton KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 \
  EXTRA="$KNOBS $REC" DOCKER_ENV="-e TORCHINDUCTOR_COMPILE_THREADS=1 -e SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/out -v $D/dump:/out -v ~/glmf/ed_expert_distribution_overlay.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_distribution.py:ro" \
  bash launch-glmf-kv.sh > "$D/launch.txt" 2>&1
ok=0; for i in $(seq 1 540); do curl -sf --max-time 2 $U/v1/models >/dev/null 2>&1 && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -qx $C || break; sleep 5; done
if [[ $ok != 1 ]]; then log "BOOT FAIL ED"; docker logs $C 2>&1 | tr '\r' '\n' | tail -60 > "$D/bootfail-tail.txt"; docker stop $C >/dev/null 2>&1; exit 5; fi
log "ready $C after $(( $(date +%s) - t0 ))s"
docker logs $C 2>&1 | tr '\r' '\n' | grep -iE "KV Cache is allocated|capped to|expert.distribution|recorder" | cut -c1-300 | head -20 > "$D/boot-excerpt.txt"
export MODEL=glm-5.3-flash
python3 -c 'import flash_bench as fb; fb.knee(concs=(1,8), reps=0)' > "$D/warm.txt" 2>&1
# ---- A: synthetic fixtures (ranking set)
curl -s -X POST $U/start_expert_distribution_record > "$D/start-A.txt"; log "record A start: $(cat $D/start-A.txt | cut -c1-120)"
CONT=$C TAG=ED-A EFFORT=low python3 accept_probe.py > "$D/A-accept.txt" 2>&1
TAG=ED-A EFFORT=low python3 c1_methods.py > "$D/A-c1.txt" 2>&1
EFFORT=low python3 tool_harness.py > "$D/A-tools.txt" 2>&1
python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=1)' > "$D/A-knee8.txt" 2>&1
curl -s -X POST $U/dump_expert_distribution_record > "$D/dump-A.txt"; sleep 10
mv "$D"/dump/*.pt "$D/dumpA/" 2>/dev/null; log "dump A: $(ls -la $D/dumpA | tail -n +4 | awk '{print $5, $9}' | tr '\n' ' ')"
curl -s -X POST $U/stop_expert_distribution_record > /dev/null
# ---- B: Hermes transcript replay (holdout)
curl -s -X POST $U/start_expert_distribution_record > "$D/start-B.txt"
TAG=ED-B PASSES=1 OUT=$D/B-replay.json python3 ed_replay.py > "$D/B-replay.txt" 2>&1
curl -s -X POST $U/dump_expert_distribution_record > "$D/dump-B.txt"; sleep 10
mv "$D"/dump/*.pt "$D/dumpB/" 2>/dev/null; log "dump B: $(ls -la $D/dumpB | tail -n +4 | awk '{print $5, $9}' | tr '\n' ' ') | replay $(cat $D/B-replay.txt | tail -1)"
curl -s -X POST $U/stop_expert_distribution_record > /dev/null
unset MODEL
docker stop $C >/dev/null; log "stop+keep $C"
# ---- analysis (CPU, inside the serving image)
if ls "$D"/dumpA/*.pt >/dev/null 2>&1 && ls "$D"/dumpB/*.pt >/dev/null 2>&1; then
  docker run --rm --network none -v "$D":/w -v ~/glmf/ed_analyze.py:/a.py:ro --entrypoint python3 $IMG /a.py /w/dumpA /w/dumpB /w/ed-analysis.json > "$D/analysis.txt" 2>&1 || log "ANALYSIS_FAIL rc=$?"
  log "ANALYSIS: $(tr '\n' ' ' < $D/analysis.txt | cut -c1-1500)"
else log "ED_NO_DUMP (recorder produced no .pt files; see dump-A.txt / dump-B.txt / container log)"; fi
{ echo; echo "## $(date '+%F %H:%M %Z') — Card ED3: GLM-Flash expert-routing skew (recorder overlay, cutlass MoE for the hook) (RTX cold-band feasibility), measurement only"
  echo "worker: Milo (hermes milo, claude-opus-5-5) · James: 'yes plz' 2026-11-01 ~21:40 CST"
  grep -h "ready glmf\|dump A\|dump B\|ANALYSIS\|BOOT FAIL\|ED_NO_DUMP" "$D/runner.log" | cut -c1-1600
  echo "receipts $D · $C stopped-and-kept · nothing restored"; } >> ~/dsv41/results/ledger.md
log "===== CARD ED END ====="

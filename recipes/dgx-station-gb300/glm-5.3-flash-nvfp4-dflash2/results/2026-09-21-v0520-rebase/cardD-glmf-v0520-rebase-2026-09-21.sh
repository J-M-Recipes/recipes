#!/usr/bin/env bash
# CARD D (2026-09-21) — GLM-5.3-Flash NVFP4 + DFlash2: rebase pinned nightly-20260911-00143e9c -> tagged v0.5.20-cu130 (sglang #37818
# DFlash/KDA checkpoint fix). Lane :30001, containers glmf-*. Requires :30006 lane STOPPED (single GPU); dsv41 v18 is restarted at end.
# Four boots, same window: (1) pinned image AR (spec none) = truth; (2) pinned image DFlash2 b7 = bug-exposed; (3) v0.5.20 AR; (4) v0.5.20 DFlash2 b7.
# Instruments per boot: warm, c1_methods low, accept_probe (DFlash only), tool_harness x2, greedy_equiv (20x200), long_greedy (8x2500),
#   tf_noninferiority scored on w2-a reference text, knee C8 x3. Discriminator: long_greedy DFlash-vs-AR on each image.
# Draft: pinned 7d74cdd8 (the staged copy; byte-identical to HF 7d74 by size) — the newer bf582e4e draft is NOT in this card.
# Rules: never :30003 / dsfv-*, never glm53-*, launcher does docker rm -f on ITS OWN name only (pre-existing behaviour; tags here are unique),
#   stop-and-keep others, STOP file honoured, dsv41 v18 restored on :30006 at end. Never print API_KEY.
# Written by Milo (Hermes milo profile, claude-fable-5-1 via anthropic) for James Meadlock.
set -uo pipefail
cd /home/milo/glmf
export PYTHONUNBUFFERED=1
D=/home/milo/glmf/cardD-2026-09-21; mkdir -p "$D"
CAMP=/home/milo/dsv41/overnight-campaign-2026-09-20.log; LEDGER=/home/milo/dsv41/results/ledger.md; STOPF=/home/milo/dsv41/STOP-CAMPAIGN
OLD=lmsysorg/sglang:nightly-dev-cu13-20260911-00143e9c
NEW=lmsysorg/sglang:v0.5.20-cu130
V18=dsv41-vllm-v18-cgsizes-BOUND-REF
S=/home/milo/pin-hot-experts/scripts
TF=/home/milo/window-20260913/scripts/tf_noninferiority.py
export BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-flash API_KEY=""
ENV_BASE='-e TORCHINDUCTOR_COMPILE_THREADS=1'
exec >> "$CAMP" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-cardD-runner (nohup) · brief by Milo · James: go all (2026-09-20 22:15 CDT)\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; finish; exit 0; fi; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^(glmf-|dsv41-vllm-)'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
finish(){ stop_all; drop_caches; docker start "$V18" >/dev/null; for ((i=0;i<150;i++)); do curl -s -m 2 http://127.0.0.1:30006/v1/models | grep -q dsv41-flash-uva && { log "RESTORED $V18 after $((i*10))s"; return; }; sleep 10; done; log "WARN $V18 did not bind"; }
wait_ready(){ local n=$1; for i in $(seq 1 240); do
    curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "READY $n i=$i"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^$n$" || { log "EXITED $n"; docker logs --tail 40 "$n" 2>&1 | tr '\r' '\n' | tail -20; return 5; }
    sleep 5; done; log "READY_TIMEOUT $n"; return 5; }
boot(){ local TAG=$1 IMG=$2 SPEC=$3 EXTRA=$4 OUT=$D/$1; mkdir -p "$OUT"; export CONT=glmf-$TAG
  stop_check; stop_all; drop_caches
  log "BOOT $TAG image=$IMG spec=$SPEC extra=[$EXTRA]"
  TAG="$TAG" MODEL=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e DRAFT=/models/GLM-5.3-Flash-DFlash2 IMAGE="$IMG" \
    SPEC=$SPEC KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$EXTRA" DOCKER_ENV="$ENV_BASE" bash /home/milo/glmf/launch-glmf.sh > "$OUT/launch.txt" 2>&1 || { cat "$OUT/launch.txt"; return 1; }
  wait_ready "glmf-$TAG" || return 1
  docker logs "glmf-$TAG" 2>&1 | tr '\r' '\n' | grep -E "sglang version|Load weight|DFLASH|KDA|KV Cache is allocated|Mamba Cache|max_running_requests|checkpoint|track" | grep -v server_args | cut -c1-200 | head -30 > "$OUT/boot-excerpt.txt"
  docker exec "glmf-$TAG" python3 -c 'import sglang; print("sglang", sglang.__version__)' >> "$OUT/boot-excerpt.txt" 2>&1 || true
  nvidia-smi -i 1 --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader >> "$OUT/boot-excerpt.txt"
  return 0; }
instrument(){ local TAG=$1 SPEC=$2 OUT=$D/$1; export CONT=glmf-$TAG
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,2,4,8), reps=0)' > "$OUT/warm.txt" 2>&1
  TAG="$TAG-low" EFFORT=low python3 c1_methods.py > "$OUT/c1-low.txt" 2>&1
  [[ $SPEC == dflash ]] && TAG="$TAG-low" EFFORT=low python3 accept_probe.py > "$OUT/accept-low.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$OUT/tool-1.txt" 2>&1; EFFORT=low python3 tool_harness.py > "$OUT/tool-2.txt" 2>&1
  python3 greedy_equiv.py "$OUT/greedy-$TAG.json" > "$OUT/greedy-gen.txt" 2>&1
  python3 greedy_equiv.py --compare /home/milo/glmf/w3/greedy-w2-a.json "$OUT/greedy-$TAG.json" > "$OUT/greedy-vs-w2a.txt" 2>&1
  log "LONG_GREEDY $TAG"; python3 $S/long_greedy.py "$OUT/long-$TAG.json" > "$OUT/long-gen.txt" 2>&1; tail -1 "$OUT/long-gen.txt"
  LANE="$TAG" BASE_URL=http://127.0.0.1:30001 python3 "$TF" /home/milo/glmf/w3/greedy-w2-a.json "$OUT/tf-$TAG.json" > "$OUT/tf-log.txt" 2>&1; tail -1 "$OUT/tf-log.txt"
  python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=0)' > "$OUT/knee-c8-warm.txt" 2>&1
  python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=3)' > "$OUT/knee-c8.txt" 2>&1
  log "instrument $TAG done: $(grep -h 'GREEDY_EQUIV' $OUT/greedy-vs-w2a.txt) · c1: $(grep -h -m1 'tok/s' $OUT/c1-low.txt | cut -c1-80)"; }

log "===== CARD D START (GLM-5.3-Flash rebase 00143e9c -> v0.5.20, #37818) ====="
docker image inspect "$NEW" >/dev/null 2>&1 || { log "IMAGE NOT PRESENT: $NEW"; exit 3; }
docker image inspect "$OLD" >/dev/null 2>&1 || { log "IMAGE NOT PRESENT: $OLD"; exit 3; }
if ps -eo args | grep -q '^bash /home/milo/pin-hot-experts/scripts/cardC'; then log "REFUSE Card C running"; exit 4; fi
docker image inspect "$NEW" --format '{{.Id}} {{index .RepoDigests 0}}' > "$D/image-new.txt"; docker image inspect "$OLD" --format '{{.Id}} {{index .RepoDigests 0}}' > "$D/image-old.txt"
led "Card D start" "GLM-5.3-Flash: pinned $OLD vs $NEW, AR and DFlash2-b7 on each, same draft 7d74cdd8. Discriminator: 8x2500-token greedy DFlash-vs-AR per image (#37818 signature = late divergence on the pinned image only). :30006 dark for the card; v18 restored at end."

# order: old-AR, old-DFlash, new-AR, new-DFlash
for cfg in "oldar $OLD none" "olddf $OLD dflash" "newar $NEW none" "newdf $NEW dflash"; do
  set -- $cfg; tag=$1; img=$2; spec=$3
  if [[ $spec == dflash ]]; then extra="--max-mamba-cache-size 48 --speculative-dflash-block-size 7"; else extra="--max-mamba-cache-size 48"; fi
  if boot "$tag" "$img" "$spec" "$extra"; then instrument "$tag" "$spec"; else log "BOOT_FAIL $tag"; led "Card D boot fail $tag" "$(tail -5 $D/$tag/launch.txt 2>/dev/null | cut -c1-200 | tr '\n' ' ') $(docker logs glmf-$tag 2>&1 | tr '\r' '\n' | grep -iE 'error|Traceback' | tail -3 | cut -c1-200 | tr '\n' ' ')"; fi
done

# verdict
{
 echo "== long_greedy DFlash vs AR, pinned image (00143e9c):"; python3 $S/long_greedy.py --compare "$D/oldar/long-oldar.json" "$D/olddf/long-olddf.json" 2>&1
 echo "== long_greedy DFlash vs AR, v0.5.20:"; python3 $S/long_greedy.py --compare "$D/newar/long-newar.json" "$D/newdf/long-newdf.json" 2>&1
 echo "== long_greedy AR old vs AR new (image numerics):"; python3 $S/long_greedy.py --compare "$D/oldar/long-oldar.json" "$D/newar/long-newar.json" 2>&1
 echo "== greedy_equiv 20x200 vs w2-a:"; for t in oldar olddf newar newdf; do printf '%s: ' $t; grep -h GREEDY_EQUIV "$D/$t/greedy-vs-w2a.txt" 2>/dev/null || echo missing; done
 echo "== tf noninferiority vs pinned-AR self-score:"; for t in olddf newar newdf; do printf '%s: ' $t; python3 "$TF" --compare "$D/oldar/tf-oldar.json" "$D/$t/tf-$t.json" 2>/dev/null | tail -1 | cut -c1-200 || echo missing; done
 echo "== C1 low (first line):"; for t in oldar olddf newar newdf; do printf '%s: ' $t; grep -h -m1 'tok/s' "$D/$t/c1-low.txt" 2>/dev/null | cut -c1-100 || echo missing; done
 echo "== accept (DFlash):"; for t in olddf newdf; do printf '%s: ' $t; tail -2 "$D/$t/accept-low.txt" 2>/dev/null | tr '\n' ' ' | cut -c1-200; echo; done
 echo "== tools:"; for t in oldar olddf newar newdf; do printf '%s: ' $t; grep -h -E 'PASS|pass|[0-9]+/10' "$D/$t/tool-1.txt" "$D/$t/tool-2.txt" 2>/dev/null | tail -2 | tr '\n' ' '; echo; done
 echo "== C8:"; for t in oldar olddf newar newdf; do printf '%s: ' $t; grep -h 'conc=8\|C8\|8:' "$D/$t/knee-c8.txt" 2>/dev/null | tail -1 | cut -c1-100; done
} > "$D/VERDICT-cardD.txt" 2>&1
cat "$D/VERDICT-cardD.txt"
finish
led "Card D END" "$(head -12 $D/VERDICT-cardD.txt | tr '\n' ' ' | cut -c1-900) · all glmf-* stopped-and-kept · $V18 restored · receipts $D"
log "===== CARD D END ====="

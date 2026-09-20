#!/usr/bin/env bash
# T3b3 (2026-09-19 09:35) — discriminator boot; derived from T3b-off54 runner: nightly dee37d89 + v15 hook, --cpu-offload-gb 54 (9 UVA layers, not 10).
# Why: 07:48 attempt died at hook guard "MemAvailable 9.68 GiB < 10 GiB" at rehome layer 9. 0909 trace is identical
# layer-for-layer (layer 8: 11.36 vs 11.29 GiB) — the guard margin was ~1-2 GiB on every v15/v18 boot; not a nightly
# difference. off54 leaves ~6 GiB more host at the tail; hook re-homes to the same rowmap, so post-rehome residency is
# identical (206.6 HBM / 62.3 pinned). Offload GiB does not change kernel shapes (autotune hash unaffected).
# Same-window cand → v18ctl5 → cand(T3b2).
# Class check redone with -i (runner bug 04:24 skipped T3b wrongly): _invoke_kernel signature identical 0909 vs nightly.
# Rules: never :30003 / dsfv-*, never glm53-*, never docker rm, no pull, stop-and-keep, STOP file honoured, v18 restored at end.
# Written by Milo (Hermes milo profile) for James Meadlock, from Miloh's overnight-runner-2026-09-18.sh helpers.
set -uo pipefail
set -f
cd $BOX_HOME/dsv41
OUT=$BOX_HOME/dsv41/overnight-T1T2          # run_window_T1T2.sh writes here; moved to overnight-T3 after
T3D=$BOX_HOME/dsv41/overnight-T3
PAR=$BOX_HOME/pin-hot-experts/e2c/t3b-parity
mkdir -p "$OUT" "$T3D" "$PAR"
LEDGER=$BOX_HOME/dsv41/results/ledger.md
CAMP=$BOX_HOME/dsv41/overnight-campaign-2026-09-18.log
STOPF=$BOX_HOME/dsv41/STOP-CAMPAIGN
REF=dsv41-vllm-v18-cgsizes-BOUND-REF
S=$BOX_HOME/pin-hot-experts/scripts
T3B=dsv41-vllm-T3b3-nightly-dee37d89-hook-off54-FRESHCACHE-EXP
PROMPTS=$BOX_HOME/pin-hot-experts/scripts/e5_parity_prompts.json
exec >> "$CAMP" 2>&1

log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-t3b3-runner (nohup) · brief by Milo\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; restore_ref; exit 0; fi; }
safety(){ local up; up=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true)
  if [[ -n $up ]]; then log "REFUSE foreign container up: $up"; exit 4; fi
  if docker ps --format '{{.Names}}' | grep -qE 'dsfv-|glm53-'; then log "REFUSE prod lane up"; exit 4; fi; }
memcheck(){ local a; a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "MemAvailable=${a}GiB"; if (( a < 40 )); then log "MemAvailable<40GiB — abort"; restore_ref; exit 5; fi; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; return 1; fi
    if docker logs "$n" 2>&1 | grep -qE 'ValueError:|Traceback|RuntimeError:' ; then log "FASTFAIL $n"; docker logs "$n" 2>&1 | grep -E 'ValueError:|RuntimeError:' | tail -3; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
boot_facts(){ local n=$1; docker logs "$n" 2>&1 | grep -E 'PIN_HOT rowmap layer 0 |rehome order|HBM_expert=|GPU KV cache size|Graph capturing finished|autotune cache file|Loaded [0-9]+ configs|Autotuning process ends|Engram' | sed 's/^.*\] //' | cut -c1-160 | head -14; }
restore_ref(){ stop_keep; if ! docker ps --format '{{.Names}}' | grep -qx "$REF"; then log "restore $REF"; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || log "WARN: $REF did not bind"; fi; }
start_ref(){ stop_keep; drop_caches; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || { log "REF failed to bind"; exit 6; }; }
window(){ local t=$1; log "WINDOW $t"; bash $BOX_HOME/dsv41/run_window_T1T2.sh "$t" > "$OUT/nohup-$t.log" 2>&1; grep -h 'conc=' "$OUT/window-$t.log" | tr '\n' ';' | cut -c1-600; echo; mv "$OUT"/*"$t"* "$T3D"/ 2>/dev/null; }
parity(){ local t=$1; log "PARITY $t"; python3 $S/e2c_parity.py "$t" "$PROMPTS" "$PAR" > "$PAR/log-$t.txt" 2>&1 || log "parity $t nonzero exit"; }
seed(){ nohup bash $BOX_HOME/dsv41/seed_autotune.sh > $BOX_HOME/dsv41/seed-$1.log 2>&1 & }
smoke(){ bash $BOX_HOME/dsv41/smoke_vllm.sh 2>&1 | tail -8; }

log "===== T3b3 START ====="; safety; memcheck
led "T3b3 runner start" "Discriminator: T3b fresh boot (live autotune) gave C1 184 / C8 705; T3b2 docker-restart of the same container (189 configs loaded from cache) gave C1 172 / C4 347 / C12 672. T3b3 = fresh docker run, same launch, autotune cache PRESENT. If 172 -> the win does not survive the FlashInfer cache round trip; if 184 -> the restart path is the problem."

# ---------- T3b3: fresh docker run, cache present ----------
stop_check; stop_keep; drop_caches
if docker ps -a --format '{{.Names}}' | grep -qx "$T3B"; then log "REFUSE $T3B exists"; restore_ref; exit 5; fi
HOOK=1 OFFGB=54 NAME=$T3B bash $S/launch-t3-nightly.sh >/dev/null
if wait_bind $T3B 100; then
  boot_facts $T3B | tee $T3D/boot-T3b3.txt
  smoke | tee $T3D/smoke-T3b3.txt
  window T3b3
  parity T3b3
  led "T3b3 window complete" "$(grep -h 'conc= 1:' $T3D/window-T3b3.log | head -1 | cut -c1-120); receipts $T3D + $PAR"
else
  led "T3b3 boot FAIL" "$(docker logs $T3B 2>&1 | grep -E 'Error|ValueError|PIN_HOT' | tail -4 | cut -c1-300)"
  docker rename $T3B ${T3B}-BOOT-FAIL
fi
restore_ref
log "===== T3b3 END ====="; led "T3b3 runner END" "v18 restored on :30006; T3b stopped-and-kept."

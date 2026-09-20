#!/usr/bin/env bash
# T3b5 (2026-09-19 evening) — cache-pin discriminator. Nightly dee37d89 + v15 hook, --cpu-offload-gb 54,
# booted with the T3b4 "fast" autotune set (ed692e15….from-T3b4, the 196 tok/s live tune) IN PLACE as the cache dir.
# Question: does a LOADED good cache reproduce 196 (→ recipe = pin a good cache), or 172 (→ the load path is the tax)?
# Same-window: T3b5 → v18ctl6 → T3b5b (fresh docker run again, same cache). Two candidate windows, one control.
# Derived from t3b4-runner-2026-09-19.sh (Milo). Cache dirs are swapped by name and restored at the end; nothing deleted.
# Rules: never :30003 / dsfv-*, never glm53-*, never docker rm, no pull, stop-and-keep, STOP file honoured, v18 restored at end.
# Written by Milo (Hermes milo profile, claude-fable-5-1 via anthropic) for James Meadlock.
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
T3B=dsv41-vllm-T3b5-nightly-dee37d89-hook-off54-CACHEPIN-EXP
T3BB=dsv41-vllm-T3b5b-nightly-dee37d89-hook-off54-CACHEPIN-EXP
CACHEROOT=$BOX_HOME/dsv41/vllm-cache/flashinfer_autotune_cache/0.6.18.post1/103a
CACHEDIR=$CACHEROOT/ed692e1558f9ebb1fb60499884ccc66154022eda6e61e542e06c6e7e352177fb
FAST=$CACHEDIR.from-T3b4
PROMPTS=$BOX_HOME/pin-hot-experts/scripts/e5_parity_prompts.json
exec >> "$CAMP" 2>&1

log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-t3b5-runner (nohup) · brief by Milo\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
safety(){ local up; up=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true)
  if [[ -n $up ]]; then log "REFUSE foreign container up: $up"; exit 4; fi
  if docker ps --format '{{.Names}}' | grep -qE 'dsfv-|glm53-'; then log "REFUSE prod lane up"; exit 4; fi; }
memcheck(){ local a; a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "MemAvailable=${a}GiB"; if (( a < 40 )); then log "MemAvailable<40GiB — abort"; restore_caches; restore_ref; exit 5; fi; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; return 1; fi
    if docker logs "$n" 2>&1 | grep -qE 'ValueError:|Traceback|RuntimeError:' ; then log "FASTFAIL $n"; docker logs "$n" 2>&1 | grep -E 'ValueError:|RuntimeError:' | tail -3; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
boot_facts(){ local n=$1; docker logs "$n" 2>&1 | grep -E 'PIN_HOT rowmap layer 0 |rehome order|HBM_expert=|GPU KV cache size|Graph capturing finished|autotune cache file|Loaded [0-9]+ configs|Autotuning process ends|Saved [0-9]+|Engram' | sed 's/^.*\] //' | cut -c1-160 | head -16; }
power(){ nvidia-smi -i 1 --query-gpu=power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active --format=csv,noheader; }
# Cache-dir choreography: put the FAST set at the canonical hash path; keep the slow set aside; restore both at end.
place_fast(){ if [[ -d $CACHEDIR && ! -d $CACHEDIR.aside-T3b5 ]]; then sudo mv "$CACHEDIR" "$CACHEDIR.aside-T3b5"; log "slow set moved aside"; fi
  if [[ -d $FAST && ! -d $CACHEDIR ]]; then sudo cp -a "$FAST" "$CACHEDIR"; log "FAST set copied into place (original .from-T3b4 kept)"; fi
  sha256sum "$CACHEDIR/autotune_configs.json" "$FAST/autotune_configs.json" | tee "$T3D/cachepin-sha-T3b5.txt"; }
restore_caches(){ if [[ -d $CACHEDIR.aside-T3b5 ]]; then
    if [[ -d $CACHEDIR ]]; then sudo mv "$CACHEDIR" "$CACHEDIR.after-T3b5"; log "post-run cache kept as .after-T3b5"; fi
    sudo mv "$CACHEDIR.aside-T3b5" "$CACHEDIR"; log "slow set restored to canonical path"; fi; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; restore_caches; restore_ref; exit 0; fi; }
restore_ref(){ stop_keep; if ! docker ps --format '{{.Names}}' | grep -qx "$REF"; then log "restore $REF"; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || log "WARN: $REF did not bind"; fi; }
start_ref(){ stop_keep; drop_caches; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || { log "REF failed to bind"; restore_caches; exit 6; }; }
window(){ local t=$1; log "WINDOW $t"; log "power-pre $t: $(power)"; bash $BOX_HOME/dsv41/run_window_T1T2.sh "$t" > "$OUT/nohup-$t.log" 2>&1; log "power-post $t: $(power)"; grep -h 'conc=' "$OUT/window-$t.log" | tr '\n' ';' | cut -c1-600; echo; mv "$OUT"/*"$t"* "$T3D"/ 2>/dev/null; }
parity(){ local t=$1; log "PARITY $t"; python3 $S/e2c_parity.py "$t" "$PROMPTS" "$PAR" > "$PAR/log-$t.txt" 2>&1 || log "parity $t nonzero exit"; }
smoke(){ bash $BOX_HOME/dsv41/smoke_vllm.sh 2>&1 | tail -8; }
cand(){ local name=$1 tag=$2
  stop_check; stop_keep; drop_caches
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then log "REFUSE $name exists"; return 5; fi
  HOOK=1 OFFGB=54 NAME=$name bash $S/launch-t3-nightly.sh >/dev/null
  if wait_bind "$name" 100; then
    boot_facts "$name" | tee "$T3D/boot-$tag.txt"
    smoke | tee "$T3D/smoke-$tag.txt"
    window "$tag"; parity "$tag"
    led "$tag window complete" "$(grep -h 'conc= 1:' $T3D/window-$tag.log | head -1 | cut -c1-120); boot: $(grep -hE 'Loaded|Saved|Autotuning' $T3D/boot-$tag.txt | head -2 | tr '\n' ' '); receipts $T3D + $PAR"
    return 0
  else
    led "$tag boot FAIL" "$(docker logs $name 2>&1 | grep -E 'Error|ValueError|PIN_HOT' | tail -4 | cut -c1-300)"
    docker rename "$name" "${name}-BOOT-FAIL"; return 1
  fi; }

log "===== T3b5 START ====="; safety; memcheck
test -d "$FAST" || { log "FAST cache set missing: $FAST"; exit 3; }
led "T3b5 runner start" "Cache-pin discriminator: nightly+hook off54 booted with the T3b4 live-tune set (196 tok/s) placed at the canonical hash dir. Expect hash-hit 'Loaded 189'. 196 → pin-a-good-cache is the recipe; 172 → load path is the tax. Windows T3b5 → v18ctl6 → T3b5b."
place_fast
# ---------- T3b5: fresh docker run, FAST cache in place ----------
cand "$T3B" T3b5 || { restore_caches; restore_ref; log "===== T3b5 END (boot fail) ====="; exit 1; }
# ---------- control ----------
stop_check; start_ref; log "power-ref: $(power)"; window v18ctl6; parity v18ctl6
# ---------- T3b5b: second fresh docker run, same FAST cache ----------
cand "$T3BB" T3b5b || log "T3b5b boot fail (T3b5 window stands alone)"
restore_caches
restore_ref
log "power-end: $(power)"
log "===== T3b5 END ====="; led "T3b5 runner END" "v18 restored on :30006; T3b5/T3b5b stopped-and-kept; cache dirs: slow set back at canonical path, .from-T3b4 untouched, post-run copy kept as .after-T3b5."

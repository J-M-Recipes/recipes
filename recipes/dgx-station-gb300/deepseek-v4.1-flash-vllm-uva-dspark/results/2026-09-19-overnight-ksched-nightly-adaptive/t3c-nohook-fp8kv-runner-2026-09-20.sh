#!/usr/bin/env bash
# T3c (2026-09-20) — flag-only Station row for vllm-project/recipes. Nightly dee37d89, hook OFF, --cpu-offload-gb 60,
# --kv-cache-dtype fp8_ds_mla. Question: what does the pure-flag nightly launch serve at 1M ctx with fp8 KV, and does
# fp8 restore DSpark acceptance hook-off the way it did hook-on (T3b6)? Peer = T3 (hook-off nvfp4, 97.1 C1, 2026-09-19).
# Windows: T3c (fresh boot; KV dtype changes the hook-off hash bba7410c -> expect ~80-min live tune)
#       -> T3rs (docker start of the kept T3 container: hook-off nvfp4, cache loaded — the direct peer, same window)
#       -> v18ctl8 (reference anchor)
#       -> T3cb (second fresh boot loading T3c's saved set — the servable number; 4th live-vs-loaded pair).
# Rules: never :30003 / dsfv-*, never glm53-*, never docker rm, no pull, stop-and-keep, STOP file honoured, v18 restored at end.
# Written by Milo (Hermes milo profile, claude-fable-5-1 via anthropic) for James Meadlock, from t3b6-fp8kv-runner.
set -uo pipefail
set -f
cd /home/milo/dsv41
OUT=/home/milo/dsv41/overnight-T1T2          # run_window_T1T2.sh writes here; moved to overnight-T3 after
T3D=/home/milo/dsv41/overnight-T3
PAR=/home/milo/pin-hot-experts/e2c/t3b-parity
mkdir -p "$OUT" "$T3D" "$PAR"
LEDGER=/home/milo/dsv41/results/ledger.md
CAMP=/home/milo/dsv41/overnight-campaign-2026-09-18.log
STOPF=/home/milo/dsv41/STOP-CAMPAIGN
REF=dsv41-vllm-v18-cgsizes-BOUND-REF
S=/home/milo/pin-hot-experts/scripts
T3RS=dsv41-vllm-T3-nightly-dee37d89-nohook-EXP
T3C=dsv41-vllm-T3c-nightly-dee37d89-nohook-off60-fp8kv-EXP
T3CB=dsv41-vllm-T3cb-nightly-dee37d89-nohook-off60-fp8kv-EXP
CACHEROOT=/home/milo/dsv41/vllm-cache/flashinfer_autotune_cache/0.6.18.post1/103a
PROMPTS=/home/milo/pin-hot-experts/scripts/e5_parity_prompts.json
exec >> "$CAMP" 2>&1

log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-t3c-runner (nohup) · brief by Milo\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
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
boot_facts(){ local n=$1; docker logs "$n" 2>&1 | grep -E 'KV cache format|GPU KV cache size|Graph capturing finished|autotune cache file|Loaded [0-9]+ configs|Autotuning process ends|Saved [0-9]+|Engram|offload' | sed 's/^.*\] //' | cut -c1-160 | head -16; }
power(){ nvidia-smi -i 1 --query-gpu=power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active --format=csv,noheader; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; restore_ref; exit 0; fi; }
restore_ref(){ stop_keep; if ! docker ps --format '{{.Names}}' | grep -qx "$REF"; then log "restore $REF"; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || log "WARN: $REF did not bind"; fi; }
start_kept(){ local n=$1; stop_keep; drop_caches; docker start "$n" >/dev/null; wait_bind "$n" 25 || { log "$n failed to bind"; return 1; }; }
window(){ local t=$1; log "WINDOW $t"; log "power-pre $t: $(power)"; bash /home/milo/dsv41/run_window_T1T2.sh "$t" > "$OUT/nohup-$t.log" 2>&1; log "power-post $t: $(power)"; grep -h 'conc=' "$OUT/window-$t.log" | tr '\n' ';' | cut -c1-600; echo; mv "$OUT"/*"$t"* "$T3D"/ 2>/dev/null; }
parity(){ local t=$1; log "PARITY $t"; python3 $S/e2c_parity.py "$t" "$PROMPTS" "$PAR" > "$PAR/log-$t.txt" 2>&1 || log "parity $t nonzero exit"; }
smoke(){ bash /home/milo/dsv41/smoke_vllm.sh 2>&1 | tail -8; }
snap_cache(){ ls -1 "$CACHEROOT" > "$T3D/cachedirs-$1.txt"; }
cand(){ local name=$1 tag=$2
  stop_check; stop_keep; drop_caches
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then log "REFUSE $name exists"; return 5; fi
  snap_cache "pre-$tag"
  HOOK=0 OFFGB=60 KVDTYPE=fp8_ds_mla NAME=$name bash $S/launch-t3-nightly.sh >/dev/null
  if wait_bind "$name" 100; then
    boot_facts "$name" | tee "$T3D/boot-$tag.txt"
    snap_cache "post-$tag"
    smoke | tee "$T3D/smoke-$tag.txt"
    window "$tag"; parity "$tag"
    led "$tag window complete" "$(grep -h 'conc= 1:' $T3D/window-$tag.log | head -1 | cut -c1-120); boot: $(grep -hE 'Loaded|Saved|Autotuning' $T3D/boot-$tag.txt | head -2 | tr '\n' ' '); receipts $T3D + $PAR"
    return 0
  else
    led "$tag boot FAIL" "$(docker logs $name 2>&1 | grep -E 'Error|ValueError' | tail -4 | cut -c1-300)"
    docker rename "$name" "${name}-BOOT-FAIL"; return 1
  fi; }

log "===== T3c START ====="; safety; memcheck
led "T3c runner start" "Flag-only Station row: nightly dee37d89, hook OFF, off60, --kv-cache-dtype fp8_ds_mla. Peer = T3 (hook-off nvfp4). Windows T3c (live tune expected) -> T3rs (docker start of kept T3, nvfp4 loaded) -> v18ctl8 -> T3cb (fresh boot loading T3c's set). Verdict = fixture acceptance per class + knee vs T3rs; T3cb is the servable number for the recipes row."
# ---------- T3c: fresh docker run, hook off, fp8 KV ----------
cand "$T3C" T3c || { restore_ref; log "===== T3c END (boot fail) ====="; exit 1; }
# ---------- T3rs: kept T3 container restarted (hook-off nvfp4, cache loaded) ----------
stop_check
if start_kept "$T3RS"; then boot_facts "$T3RS" | tee "$T3D/boot-T3rs.txt"; smoke | tee "$T3D/smoke-T3rs.txt"; window T3rs; parity T3rs
  led "T3rs window complete" "$(grep -h 'conc= 1:' $T3D/window-T3rs.log | head -1 | cut -c1-120)"
else led "T3rs restart FAIL" "$(docker logs $T3RS 2>&1 | tail -3 | cut -c1-300)"; fi
# ---------- v18ctl8 ----------
stop_check; start_kept "$REF" || { log "REF failed to bind"; exit 6; }; log "power-ref: $(power)"; window v18ctl8; parity v18ctl8
# ---------- T3cb: second fresh docker run, T3c's saved set loaded ----------
cand "$T3CB" T3cb || log "T3cb boot fail (T3c window stands alone)"
restore_ref
log "power-end: $(power)"
log "===== T3c END ====="; led "T3c runner END" "v18 restored on :30006; T3c/T3cb/T3 stopped-and-kept; no cache dirs moved (new fp8 hook-off dir, if any, left in place — listed in cachedirs-post-T3c.txt)."

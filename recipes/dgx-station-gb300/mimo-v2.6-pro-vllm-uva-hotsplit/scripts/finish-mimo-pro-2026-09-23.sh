#!/usr/bin/env bash
# finish-mimo-pro-2026-09-23.sh — MiMo-V2.6-Pro hotsplit: experimental -> verified evidence run.
# Arms (same image sha256:f29125bc…, same weights, same flags; only hotsplit env differs):
#   ctrl = mimo26-pro-v20-ctrl-C1-30   (stock layer-order UVA placement)
#   v23  = mimo26-pro-v23-hot153-live  (hotsplit 152.8 GiB, the published headline boot)
#   v24  = mimo26-pro-v24-1m           (1M-context variant; phase C only: the ~1.04M needle row)
# Per arm A/B: warm (agent_fixture discard) -> greedy parity capture -> teacher-forced capture -> BFCL dev -> BFCL held-out.
# Then compare, then phase C (1M row on v24), then STOP-AND-KEEP everything (no restore-on-done; James rule).
# Bars are pinned in harness/protocol.yaml BEFORE this run. STOP file: ~/mimo26/finish/STOP.
# Written by Milo (Hermes milo profile) for James Meadlock, who asked "finish mimo pro" 2026-09-23 22:xx CDT.
set -uo pipefail
W=$HOME/mimo26
F=$W/finish; D=$F/run-2026-09-23; mkdir -p "$D"
BF=$HOME/pin-hot-experts/bfcl_data
STOPF=$F/STOP
CTRL=mimo26-pro-v20-ctrl-C1-30; V23=mimo26-pro-v23-hot153-live; V24=mimo26-pro-v24-1m
exec 9>"$F/.lock"; flock -n 9 || { echo "another finish runner holds the lock"; exit 5; }
exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
gpu_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^mimo26-pro-' || true; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^mimo26-pro-'); do log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; done; sleep 5; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30007/v1/models 2>/dev/null | grep -q mimo26-pro; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; docker logs --tail 20 "$n" 2>&1 | sed 's/^/    /'; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP file seen -> stop-and-keep, exit"; stop_keep; log "===== END (STOP) ====="; exit 0; fi; }
start_kept(){ local n=$1; stop_keep; local f; f=$(gpu_foreign); if [[ -n $f ]]; then log "REFUSE: foreign container up: $f"; exit 4; fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 | grep -q .; then log "REFUSE: GPU 1 has compute apps"; exit 4; fi
  drop_caches; log "docker start $n"; docker start "$n" >/dev/null; wait_bind "$n" 30; }
power_on(){ ( while :; do printf '%s,%s\n' "$(date +%s)" "$(nvidia-smi -i 1 --query-gpu=power.draw --format=csv,noheader,nounits)"; sleep 5; done ) > "$D/power-$1.csv" & echo $! > "$D/.pw"; }
power_off(){ kill "$(cat "$D/.pw")" 2>/dev/null; awk -F, '{s+=$2;n++} END{if(n) printf "power %s: mean %.1f W over %d samples\n", FILENAME, s/n, n}' "$D/power-$1.csv"; }
identity(){ local n=$1; docker inspect "$n" --format '{{.Name}} image={{.Image}}' ; docker exec "$n" sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py /w/hotsplit.py 2>/dev/null | sed 's/^/    /'
  docker logs "$n" 2>&1 | grep -E "hotsplit plan|hotsplit done|GPU KV cache size|Total CPU offloaded|MARLIN" | tail -6 | sed 's/^/    /'; }

arm(){ local tag=$1 n=$2
  stop_check; log "===== ARM $tag ($n) ====="
  start_kept "$n" || { log "ARM $tag bind FAIL"; return 1; }
  identity "$n" | tee "$D/identity-$tag.txt"
  ( cd "$W" && bash agent_fixture.sh "finish-$tag-warm" > "$D/warm-$tag.log" 2>&1 ); log "warm done: $(tail -2 "$D/warm-$tag.log" | tr '\n' ' ' | cut -c1-200)"
  stop_check; ( cd "$D" && python3 "$F/mimo_greedy.py" capture "$tag" ) > "$D/greedy-$tag.log" 2>&1; log "greedy: $(tail -1 "$D/greedy-$tag.log")"
  stop_check; ( cd "$D" && BASE_URL=http://127.0.0.1:30007/v1 MODEL=mimo26-pro MAXPOS=4096 CONC=4 python3 "$F/tf_logprob.py" capture "$tag" "$F/tf_corpus.jsonl" ) > "$D/tf-$tag.log" 2>&1; log "tf: $(tail -1 "$D/tf-$tag.log")"
  # smoke: 24 dev cases; abort the whole run if the tool path is broken (acc < 0.5 or >4 no-calls) rather than burn an hour
  ( cd "$D" && SUITE=dev LIMIT=24 BASE_URL=http://127.0.0.1:30007/v1 MODEL=mimo26-pro CONC=8 MAXTOK=2048 CTK='{"enable_thinking": false}' \
      python3 "$F/bfcl_gate.py" "$tag-smoke" "$BF" "$D" ) > "$D/bfcl-$tag-smoke.log" 2>&1
  log "bfcl smoke: $(tail -1 "$D/bfcl-$tag-smoke.log")"
  if ! python3 -c "import json,sys;s=json.load(open('$D/bfcl-$tag-smoke-summary.json'))['all'];sys.exit(0 if s['acc'] and s['acc']>=0.5 and s['no_call']<=4 else 1)"; then
    log "ABORT: BFCL smoke failed on $tag (tool path broken?) -> stop-and-keep"; stop_keep; exit 7; fi
  for suite in dev heldout; do stop_check
    power_on "$tag-$suite"
    ( cd "$D" && SUITE=$suite BASE_URL=http://127.0.0.1:30007/v1 MODEL=mimo26-pro CONC=8 MAXTOK=2048 CTK='{"enable_thinking": false}' \
        python3 "$F/bfcl_gate.py" "$tag-$suite" "$BF" "$D" ) > "$D/bfcl-$tag-$suite.log" 2>&1
    power_off "$tag-$suite"; log "bfcl $suite: $(tail -1 "$D/bfcl-$tag-$suite.log")"
  done
  log "ARM $tag done"; }

log "===== FINISH-MIMO-PRO START ====="
( cd "$BF" && sha256sum -c "$F/BFCL-SHA256SUMS" ) > "$D/bfcl-data-check.txt" 2>&1 || { log "BFCL data sha mismatch"; cat "$D/bfcl-data-check.txt"; exit 3; }
sha256sum "$F"/* 2>/dev/null > "$D/runner-SHA256SUMS"
f=$(gpu_foreign); if [[ -n $f ]]; then log "REFUSE at start: foreign container up: $f"; exit 4; fi

arm ctrl "$CTRL"
arm v23 "$V23"
stop_check
( cd "$D" && python3 "$F/tf_logprob.py" compare tf-ctrl.jsonl tf-v23.jsonl ) > "$D/tf-compare.log" 2>&1; log "$(tail -1 "$D/tf-compare.log")"
( cd "$D" && python3 "$F/mimo_greedy.py" compare greedy-ctrl.json greedy-v23.json ) > "$D/greedy-compare.log" 2>&1; log "$(head -1 "$D/greedy-compare.log")"

# ---- phase C: the 1M row (v24), one needle size, 3 depths + decode timing, ~70-80 min ----
stop_check
if start_kept "$V24"; then
  identity "$V24" | tee "$D/identity-v24.txt"
  ( cd "$W" && bash agent_fixture.sh "finish-v24-warm" > "$D/warm-v24.log" 2>&1 )
  stop_check; ( cd "$D" && python3 "$W/longctx_bench.py" finish-v24-1m 1040000 ) > "$D/longctx-v24-1m.log" 2>&1
  log "longctx 1M: $(tail -1 "$D/longctx-v24-1m.log")"
fi

stop_keep
python3 "$F/verdict.py" "$D" > "$D/VERDICT.md" 2>&1; cat "$D/VERDICT.md"
log "===== FINISH-MIMO-PRO END (all mimo26 containers stopped-and-kept; GPU idle) ====="

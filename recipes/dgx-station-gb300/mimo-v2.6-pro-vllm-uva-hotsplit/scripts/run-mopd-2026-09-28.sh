#!/usr/bin/env bash
# run-mopd-2026-09-28.sh — MiMo-V2.6-Pro-MOPD weight swap on the hotsplit recipe. Bars: harness/protocol-v2.yaml (pinned BEFORE this run).
# Arms (same image sha256:f29125bc…, same three bind-mounted patches, same flags; only weights / HOTSPLIT_* differ):
#   rl-hot = kept container mimo26-pro-v23-hot153-live (RL weights, hotsplit)         : flood fixture + BFCL dev (fresh RL reference + RL live counts)
#   m-ctrl = NEW mimo26-pro-mctrl-<run> (MOPD weights, stock layer-order placement)    : greedy, TF, BFCL smoke/dev/held-out, flood
#   m-hot  = NEW mimo26-pro-mhot153-live-<run> (MOPD weights, hotsplit 152.8, LIVE=1)  : greedy, TF, BFCL smoke/dev/held-out, flood, live counts
# Then compare (TF m-ctrl vs m-hot; TF RL-ctrl-2026-09-24 vs m-ctrl; greedy; ranking portability; flood A/B) and STOP-AND-KEEP everything.
# No restore-on-done (James rule): :30006 stays down until James says otherwise. STOP file: ~/mimo26/mopd/STOP.
# Written by Milo (Hermes milo profile) 2026-09-28 for James Meadlock. v2.1 05:40 CDT: MAX_TURNS=6 MAX_CALLS=120 (protocol-v2.yaml revision_note).
set -uo pipefail
W=$HOME/mimo26; F=$W/finish; M=$W/mopd; RUN=${RUN:-2026-09-28-mopd}; D=$M/run-$RUN; mkdir -p "$D"
BF=$HOME/pin-hot-experts/bfcl_data; STOPF=$M/STOP
MODEL_MOPD=/models/milo/MiMo-V2.6-Pro-MOPD
IMG=vllm/vllm-openai:nightly-d05da62e9ccdf8e342b15bf6785d83224cc165af
RLHOT=mimo26-pro-v23-hot153-live; MCTRL=mimo26-pro-mctrl-$RUN; MHOT=mimo26-pro-mhot153-live-$RUN
exec 9>"$M/.lock"; flock -n 9 || { echo "another mopd runner holds the lock"; exit 5; }
exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
gpu_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^mimo26-pro-' || true; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^mimo26-pro-'); do log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; done; sleep 5; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30007/v1/models 2>/dev/null | grep -q mimo26-pro; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; docker logs --tail 30 "$n" 2>&1 | sed 's/^/    /'; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP file seen -> stop-and-keep, exit"; stop_keep; log "===== END (STOP) ====="; exit 0; fi; }
refuse_if_busy(){ local f; f=$(gpu_foreign); if [[ -n $f ]]; then log "REFUSE: foreign container up: $f"; exit 4; fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 | grep -q .; then log "REFUSE: GPU 1 has compute apps"; exit 4; fi; }
start_kept(){ local n=$1; stop_keep; refuse_if_busy; drop_caches; log "docker start $n"; docker start "$n" >/dev/null; wait_bind "$n" 30; }
# docker run line = the inspected Args of v20-ctrl / v23 verbatim, only the model mount and HOTSPLIT_* env differ
boot_new(){ local n=$1 mode=$2; stop_keep; refuse_if_busy; drop_caches
  local hs=(); if [[ $mode == hot ]]; then hs=(-e HOTSPLIT_COUNTS=/w/dumps/expert_hist_mix.json -e HOTSPLIT_LIVE_COUNTS=/live/counts.json -e HOTSPLIT_LIVE_SECS=600 -e HOTSPLIT_WEIGHTS=decode=1,prefill=0 -e HOTSPLIT_HOT_GIB=152.8); fi
  log "docker run $n ($mode) model=$MODEL_MOPD"
  docker run -d --name "$n" --label gpu-lane=1 --gpus all --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
    -v "$MODEL_MOPD":/model:ro -v "$W"/vllm-cache:/root/.cache/vllm -v "$W":/w:ro -v "$W"/live:/live \
    -v "$W"/patch/mimo_v2_omni.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py:ro \
    -v "$W"/patch/mimo_v2_hotsplit.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py:ro \
    -e VLLM_LOGGING_LEVEL=INFO -e VLLM_USE_DEEP_GEMM=0 -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 "${hs[@]}" \
    "$IMG" --model /model --served-model-name mimo26-pro --trust-remote-code --tensor-parallel-size 1 \
    --offload-backend uva --cpu-offload-gb 320 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
    --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.96 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --tool-call-parser mimo --reasoning-parser mimo --enable-auto-tool-choice \
    --generation-config auto --moe-backend marlin --host 0.0.0.0 --port 30007 >/dev/null || { log "docker run failed"; return 1; }
  nohup docker logs -f "$n" > "$M/$n.log" 2>&1 &
  wait_bind "$n" 40; }
power_on(){ ( while :; do printf '%s,%s\n' "$(date +%s)" "$(nvidia-smi -i 1 --query-gpu=power.draw --format=csv,noheader,nounits)"; sleep 5; done ) > "$D/power-$1.csv" & echo $! > "$D/.pw"; }
power_off(){ kill "$(cat "$D/.pw")" 2>/dev/null; awk -F, '{s+=$2;n++} END{if(n) printf "power %s: mean %.1f W over %d samples\n", FILENAME, s/n, n}' "$D/power-$1.csv"; }
identity(){ local n=$1; docker inspect "$n" --format '{{.Name}} image={{.Image}} model={{range .HostConfig.Binds}}{{if eq (index (split . ":") 1) "/model"}}{{index (split . ":") 0}}{{end}}{{end}}'
  docker exec "$n" sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py /w/hotsplit.py 2>/dev/null | sed 's/^/    /'
  docker exec "$n" sha256sum /model/config.json /model/chat_template.jinja 2>/dev/null | sed 's/^/    /'
  docker logs "$n" 2>&1 | grep -E "hotsplit plan|hotsplit done|GPU KV cache size|Total CPU offloaded|MARLIN" | tail -6 | sed 's/^/    /'; }
snapshot_live(){ local tag=$1; log "waiting up to 630s for the live counter flush, then snapshot live counts as $tag"
  local age; age=$(( $(date +%s) - $(stat -c %Y "$W/live/counts.json") )); [[ $age -lt 630 ]] && sleep $((630 - age)) || true
  sudo cp "$W/live/counts.json" "$D/live-counts-$tag.json" && sudo chown milo:milo "$D/live-counts-$tag.json"
  python3 -c "import json;d=json.load(open('$D/live-counts-$tag.json'));l=d['live']['decode'];print('live tokens/layer', min(sum(v) for v in l.values())/8, d.get('meta'))" | tee -a "$D/live-$tag.txt"; }
BASE=http://127.0.0.1:30007/v1
flood(){ local tag=$1; stop_check; power_on "$tag-flood"
  ( cd "$D" && BASE_URL=$BASE MODEL=mimo26-pro CONC=4 MAX_TURNS=6 MAX_CALLS=120 MAXTOK=4096 THINKING=1 HISTORY=both python3 "$W/flood_fixture.py" run "$tag" "$D" ) > "$D/flood-$tag.log" 2>&1
  power_off "$tag-flood"; log "flood: $(grep '^FLOOD' "$D/flood-$tag.log" | tail -1)"; }
bfcl(){ local tag=$1 suite=$2; stop_check; power_on "$tag-$suite"
  ( cd "$D" && SUITE=$suite BASE_URL=$BASE MODEL=mimo26-pro CONC=8 MAXTOK=2048 CTK='{"enable_thinking": false}' python3 "$F/bfcl_gate.py" "$tag-$suite" "$BF" "$D" ) > "$D/bfcl-$tag-$suite.log" 2>&1
  power_off "$tag-$suite"; log "bfcl $suite: $(tail -1 "$D/bfcl-$tag-$suite.log")"; }
fidelity(){ local tag=$1
  stop_check; ( cd "$D" && python3 "$F/mimo_greedy.py" capture "$tag" ) > "$D/greedy-$tag.log" 2>&1; log "greedy: $(tail -1 "$D/greedy-$tag.log")"
  stop_check; ( cd "$D" && BASE_URL=$BASE MODEL=mimo26-pro MAXPOS=4096 CONC=4 python3 "$F/tf_logprob.py" capture "$tag" "$F/tf_corpus.jsonl" ) > "$D/tf-$tag.log" 2>&1; log "tf: $(tail -1 "$D/tf-$tag.log")"
  ( cd "$D" && SUITE=dev LIMIT=24 BASE_URL=$BASE MODEL=mimo26-pro CONC=8 MAXTOK=2048 CTK='{"enable_thinking": false}' python3 "$F/bfcl_gate.py" "$tag-smoke" "$BF" "$D" ) > "$D/bfcl-$tag-smoke.log" 2>&1
  log "bfcl smoke: $(tail -1 "$D/bfcl-$tag-smoke.log")"
  if ! python3 -c "import json,sys;s=json.load(open('$D/bfcl-$tag-smoke-summary.json'))['all'];sys.exit(0 if s['acc'] and s['acc']>=0.5 and s['no_call']<=4 else 1)"; then
    log "ABORT: BFCL smoke failed on $tag (tool path broken on MOPD?) -> stop-and-keep"; stop_keep; exit 7; fi; }

log "===== MOPD RUN START ($RUN) ====="
( cd "$BF" && sha256sum -c "$F/BFCL-SHA256SUMS" ) > "$D/bfcl-data-check.txt" 2>&1 || { log "BFCL data sha mismatch"; exit 3; }
test -f "$MODEL_MOPD/config.json" && test -f "$MODEL_MOPD/dflash/config.json" || { log "MOPD model incomplete"; exit 3; }
test -f "$M/MANIFEST-OK" || { log "MOPD per-file size manifest not verified (no $M/MANIFEST-OK)"; exit 3; }
sha256sum "$W/flood_fixture.py" "$W/ranking_portability.py" "$F"/*.py "$M"/*.sh "$M"/protocol-v2.yaml 2>/dev/null > "$D/runner-SHA256SUMS"
python3 "$W/flood_fixture.py" sha > "$D/flood-fixture-sha256.txt"
refuse_if_busy

# ---- arm 1: rl-hot (kept v23, RL weights): flood + BFCL dev + live counts
stop_check; log "===== ARM rl-hot ($RLHOT) ====="
start_kept "$RLHOT" || { log "rl-hot bind FAIL"; exit 6; }
identity "$RLHOT" | tee "$D/identity-rlhot.txt"
( cd "$W" && bash agent_fixture.sh "mopd-rlhot-warm" > "$D/warm-rlhot.log" 2>&1 ); log "warm done"
flood rlhot
bfcl rlhot dev
snapshot_live rlhot

# ---- arm 2: m-ctrl (MOPD, stock placement)
stop_check; log "===== ARM m-ctrl ($MCTRL) ====="
boot_new "$MCTRL" ctrl || { log "m-ctrl boot FAIL"; stop_keep; exit 6; }
identity "$MCTRL" | tee "$D/identity-mctrl.txt"
( cd "$W" && bash agent_fixture.sh "mopd-mctrl-warm" > "$D/warm-mctrl.log" 2>&1 ); log "warm done"
fidelity mctrl
bfcl mctrl dev
bfcl mctrl heldout
flood mctrl

# ---- arm 3: m-hot (MOPD, hotsplit 152.8 from the RL ranking, live counter)
stop_check; log "===== ARM m-hot ($MHOT) ====="
boot_new "$MHOT" hot || { log "m-hot boot FAIL"; stop_keep; exit 6; }
identity "$MHOT" | tee "$D/identity-mhot.txt"
( cd "$W" && bash agent_fixture.sh "mopd-mhot-warm" > "$D/warm-mhot.log" 2>&1 ); log "warm done"
( cd "$W" && python3 ttft_bench.py mopd-mhot ) > "$D/ttft-mhot.log" 2>&1; log "ttft/decode: $(tail -3 "$D/ttft-mhot.log" | tr '\n' ' ' | cut -c1-300)"
( cd "$W" && bash agent_fixture.sh "mopd-mhot-r2" 2>&1 | grep -E '^\[' ) > "$D/fixture-mhot.log" 2>&1; log "fixture: $(tr '\n' ' ' < "$D/fixture-mhot.log" | cut -c1-300)"
fidelity mhot
bfcl mhot dev
bfcl mhot heldout
flood mhot
snapshot_live mhot

# ---- compare
stop_keep
( cd "$D" && python3 "$F/tf_logprob.py" compare tf-mctrl.jsonl tf-mhot.jsonl ) > "$D/tf-compare-residency.log" 2>&1; log "TF residency: $(tail -1 "$D/tf-compare-residency.log")"
( cd "$D" && cp "$F/run-2026-09-23/tf-ctrl.jsonl" tf-rlctrl0924.jsonl && python3 "$F/tf_logprob.py" compare tf-rlctrl0924.jsonl tf-mctrl.jsonl ) > "$D/tf-compare-weights.log" 2>&1; log "TF weights RL->MOPD: $(tail -1 "$D/tf-compare-weights.log")"
( cd "$D" && python3 "$F/mimo_greedy.py" compare greedy-mctrl.json greedy-mhot.json ) > "$D/greedy-compare-residency.log" 2>&1; log "$(head -1 "$D/greedy-compare-residency.log")"
( cd "$D" && cp "$F/run-2026-09-23/greedy-ctrl.json" greedy-rlctrl0924.json && python3 "$F/mimo_greedy.py" compare greedy-rlctrl0924.json greedy-mctrl.json ) > "$D/greedy-compare-weights.log" 2>&1; log "$(head -1 "$D/greedy-compare-weights.log")"
( cd "$D" && python3 "$W/ranking_portability.py" "$W/dumps/expert_hist_mix.json" live-counts-mhot.json live-counts-rlhot.json 8692 --json portability.json ) > "$D/portability.log" 2>&1; log "portability: $(grep -E 'verdict|hbm_served_share_rl_ranking' "$D/portability.log" | tr '\n' ' ')"
( cd "$D" && python3 "$W/flood_fixture.py" compare flood-rlhot-summary.json flood-mhot-summary.json ) > "$D/flood-compare-weights.log" 2>&1; log "flood RL->MOPD: $(head -1 "$D/flood-compare-weights.log")"
( cd "$D" && python3 "$W/flood_fixture.py" compare flood-mctrl-summary.json flood-mhot-summary.json ) > "$D/flood-compare-residency.log" 2>&1; log "flood residency: $(head -1 "$D/flood-compare-residency.log")"
python3 "$M/verdict2.py" "$D" > "$D/VERDICT.md" 2>&1; cat "$D/VERDICT.md"
log "===== MOPD RUN END (all mimo26 containers stopped-and-kept; GPU idle; :30006 NOT restored) ====="

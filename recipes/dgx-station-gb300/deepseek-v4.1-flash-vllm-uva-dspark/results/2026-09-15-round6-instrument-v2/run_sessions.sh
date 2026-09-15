#!/usr/bin/env bash
set -uo pipefail; cd /home/milo/dsv41; L=results/campaign-round6b-$(date +%F).log; exec >> "$L" 2>&1
log(){ printf "%s %s\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "$*"; }
V14=dsv41-vllm-v14-1M-ksched-agent-BOUND-REF
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; log "starting $V14"; docker start $V14 >/dev/null
t0=$SECONDS; while ! curl -sf -m 3 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; do sleep 15; [[ $((SECONDS-t0)) -gt 2400 ]] && { log FATAL bind; docker stop $V14; exit 1; }; done; log "BOUND after $((SECONDS-t0))s"
TAG=R6-sess-v14-a N=4 python3 replay_sessions.py; TAG=R6-sess-v14-b N=4 python3 replay_sessions.py
echo "$(date +%F) ROUND6b | v14 session-ordered | $(python3 -c "import json;[print(f\"{t}: agg={d[\"agg_tok_s\"]} ci={d[\"agg_ci95_boot\"]} pchit={d[\"prefix_cache_hit_rate\"]:.2f} ptok={d[\"prompt_tokens\"]} ttft={d[\"ttft_all_med\"]}/{d[\"ttft_all_p90\"]} acc={d[\"accept\"]:.2f} aps={d[\"acc_per_step\"]} err={d[\"errors\"]}\") for t in (\"a\",\"b\") for d in [json.load(open(f\"replays-R6-sess-v14-{t}-n4.json\"))]]" | tr "\n" " ") |" >> results/ledger.md
docker stop $V14 >/dev/null; log "lane dark; ROUND6b done"

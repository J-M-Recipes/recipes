#!/usr/bin/env bash
# campaign_round6.sh — instrument v2 (300 real turns, CI, prefix-cache, TTFT) on v14 and v13; then OFFGB=55 probe.
set -euo pipefail
cd /home/milo/dsv41
V14=dsv41-vllm-v14-1M-ksched-agent-BOUND-REF; V13=dsv41-vllm-v13-1M-ksched-BOUND-REF
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
LEDGER=results/ledger.md; CAMPLOG=results/campaign-round6-$(date +%F).log; STOPF=STOP-CAMPAIGN
mkdir -p results; exec >> "$CAMPLOG" 2>&1
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }
die() { log "FATAL: $*"; finish || true; exit 1; }
trap 'log "ERR line=$LINENO status=$?"' ERR
check_stop() { [[ -f "$STOPF" ]] && { log "STOP-CAMPAIGN"; finish; exit 0; }; return 0; }
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; log "dropped page cache; $(free -g | awk 'NR==2{print "avail="$7"GiB"}')"; }
wait_bound() { local name=$1 maxs=$2 t0=$SECONDS; log "waiting bind for $name"
  while true; do check_stop; docker ps --format '{{.Names}}' | grep -qx "$name" || return 1
    curl -sf -m 3 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva && { log "BOUND $name after $((SECONDS-t0))s"; return 0; }
    [[ $((SECONDS-t0)) -gt $maxs ]] && return 1; sleep 15; done; }
stop_lane() { docker ps --format '{{.Names}}' | grep '^dsv41-vllm-' | xargs -r docker stop >/dev/null || true; }
up() { docker ps --format '{{.Names}}' | grep -qx "$1" && return 0; stop_lane; drop_caches; log "starting $1"; docker start "$1" >/dev/null; wait_bound "$1" 2400 || die "$1 bind failed"; smoke || die "smoke failed on $1"; }
smoke() { python3 - <<'PY'
import json,urllib.request,sys
B="http://127.0.0.1:30006/v1/chat/completions"
def chat(m,mx=64):
    r=json.load(urllib.request.urlopen(urllib.request.Request(B,data=json.dumps({"model":"dsv41-flash-uva","messages":m,"max_tokens":mx,"temperature":0,"chat_template_kwargs":{"thinking":False}}).encode(),headers={"Content-Type":"application/json"}),timeout=300))
    return r["choices"][0]["message"].get("content") or ""
f=[]
"323" in chat([{"role":"user","content":"What is 17*19? Return only the integer."}]).replace(",","") or f.append("arith")
n=[int(x) for x in chat([{"role":"user","content":"Count from 1 to 30 separated by spaces. Only the numbers."}],80).split() if x.isdigit()]; n[:30]==list(range(1,31)) or f.append("count")
if f: print("SMOKE_FAIL",f); sys.exit(2)
print("SMOKE_PASS")
PY
}
ledger() { echo "$(date +%F) ROUND6 $*" >> "$LEDGER"; log "LEDGER $*"; }
r2() { python3 -c "import json;d=json.load(open('/home/milo/dsv41/replay2-$1-n4.json'));print(f\"agg={d['agg_tok_s']} ci={d['agg_ci95_boot']} acc={d['accept']:.2f} aps={d['acc_per_step']} pchit={d['prefix_cache_hit_rate']:.2f} ptok={d['prompt_tokens']} ttft_tool={d['tool']['ttft_med']}/{d['tool']['ttft_p90']} ttft_text={d['text']['ttft_med']}/{d['text']['ttft_p90']} err={d['errors']}\")" 2>/dev/null || echo NA; }
k6() { python3 -c "import json;r=json.load(open('/home/milo/dsv41/knee-$1.json'))['rows'];print(' '.join(f'C{c}={a:.1f}' for c,a,_ in r))" 2>/dev/null || echo NA; }
fx() { python3 - "$1" <<'PY' 2>/dev/null || echo NA
import json,sys; d=json.load(open(f"/home/milo/dsv41/agentfix-{sys.argv[1]}.json")); r=d["res"]
print(" ".join(f"{c}={r[c]['tok_s']}({r[c]['accept']:.2f})" for c in ("tool_json","code","shell_ops","structured","prose") if c in r)+f" wacc={d['weighted_accept']:.3f}")
PY
}
finish() { log "FINISH"; docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-R6-' | xargs -r docker stop >/dev/null || true; stop_lane; log "lane dark"; }

log "ROUND6 start"
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true); [[ -n "$other" ]] && die "other container running: $other"
for f in replay2.py knee6.sh agent_fixture_o.sh transcript_fixture_v2.json launch-dsv41-vllm.sh; do [[ -s $f ]] || die "missing $f"; done

# --- instrument v2 on v14 (reference) and v13, two runs each ---
up "$V14"; TAG=R6-v14-a N=4 ROUNDS=1 python3 replay2.py; TAG=R6-v14-b N=4 ROUNDS=1 python3 replay2.py
ledger "| v14 replay2 | a $(r2 R6-v14-a) | b $(r2 R6-v14-b) |"; check_stop
up "$V13"; TAG=R6-v13-a N=4 ROUNDS=1 python3 replay2.py; TAG=R6-v13-b N=4 ROUNDS=1 python3 replay2.py
ledger "| v13 replay2 | a $(r2 R6-v13-a) | b $(r2 R6-v13-b) |"; check_stop

# --- OFFGB=55 probe: v14 flags, 5 GiB fewer experts offloaded; hash should hold (offload GiB is not in the key) ---
stop_lane; drop_caches; tag=R6-off55; t0=$(date '+%H:%M')
TAG="$tag" OFFGB=55 UTIL=0.97 SEQS=16 CTX=1048576 SPEC="" MODEL="$MODEL" \
  EXTRA='--long-prefill-token-threshold 6144 --speculative-config {"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,16,1]]}' \
  bash -f launch-dsv41-vllm.sh
if wait_bound "dsv41-vllm-$tag" 7200 && smoke; then
  grep -aq "Loaded [0-9]* configs" "dsv41-vllm-$tag.log" && log "autotune HIT" || log "autotune retuned"
  { grep -aoE "autotune cache file: [^ ]*|Loaded [0-9]+ configs|Saved [0-9]+ configs|Available KV cache memory: [0-9.]+ GiB|Actual usage is [0-9.]+ GiB" "dsv41-vllm-$tag.log" | head -5; } > "results/facts-$tag.txt"
  TAG=R6-off55-a N=4 ROUNDS=1 python3 replay2.py; TAG=R6-off55-b N=4 ROUNDS=1 python3 replay2.py; bash knee6.sh "$tag"; bash agent_fixture_o.sh "$tag" || true
  ledger "| OFFGB=55 | $t0 → $(date '+%H:%M') | replay2 a $(r2 R6-off55-a) | b $(r2 R6-off55-b) | knee $(k6 $tag) | $(fx $tag) | $(tr '\n' ';' < results/facts-$tag.txt) |"
  docker stop "dsv41-vllm-$tag" >/dev/null; docker rename "dsv41-vllm-$tag" "dsv41-vllm-$tag-EXP" 2>/dev/null || true
else
  log "OFFGB=55 did not bind (expected if HBM headroom < 5 GiB); tail of log:"; grep -aiE "out of memory|OOM|Error|ValueError" "dsv41-vllm-$tag.log" | tail -3
  ledger "| OFFGB=55 | did not bind |"; docker stop "dsv41-vllm-$tag" >/dev/null 2>&1 || true; docker rename "dsv41-vllm-$tag" "dsv41-vllm-$tag-FAILED" 2>/dev/null || true
fi
# leave lane on v14 reference? No — receipts rule: dark at end.
finish; log "ROUND6 done"

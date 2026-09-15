#!/usr/bin/env bash
# campaign_round5.sh — one boot: [[1,4,5],[5,16,1]] judged on the 4-worker real-transcript replay.
# Sequence: v13 up → replay n=4 ×2 + knee6 (control) → v12 up → replay n=4 ×2 (second pass on the caveat) →
#           boot R5-S3-b4 (new hash, ~80 min) → replay n=4 ×2 + knee6 + fixture → dark.
set -euo pipefail
cd /home/milo/dsv41
V13=dsv41-vllm-v13-1M-ksched-BOUND-REF; V12=dsv41-vllm-v12-1M-k5-off60-util97-RETIRED-REF
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
LEDGER=results/ledger.md; CAMPLOG=results/campaign-round5-$(date +%F).log; STOPF=STOP-CAMPAIGN
mkdir -p results; exec >> "$CAMPLOG" 2>&1
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }
die() { log "FATAL: $*"; finish || true; exit 1; }
trap 'log "ERR line=$LINENO status=$?"' ERR
check_stop() { [[ -f "$STOPF" ]] && { log "STOP-CAMPAIGN"; finish; exit 0; }; return 0; }
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; log "dropped page cache; $(free -g | awk 'NR==2{print "avail="$7"GiB"}')"; }
wait_bound() { local name=$1 maxs=$2 t0=$SECONDS; log "waiting bind for $name"
  while true; do check_stop; docker ps --format '{{.Names}}' | grep -qx "$name" || die "$name not running"
    curl -sf -m 3 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva && { log "BOUND $name after $((SECONDS-t0))s"; return 0; }
    [[ $((SECONDS-t0)) -gt $maxs ]] && die "$name bind timeout"; sleep 15; done; }
stop_lane() { docker ps --format '{{.Names}}' | grep '^dsv41-vllm-' | xargs -r docker stop >/dev/null || true; }
up() { docker ps --format '{{.Names}}' | grep -qx "$1" && return 0; stop_lane; drop_caches; log "starting $1"; docker start "$1" >/dev/null; wait_bound "$1" 2400; smoke || die "smoke failed on $1"; }
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
ledger() { echo "$(date +%F) ROUND5 $*" >> "$LEDGER"; log "LEDGER $*"; }
rp() { python3 -c "import json;d=json.load(open('/home/milo/dsv41/replayc-$1.json'));print(f\"agg={d['agg_tok_s']} acc={d['accept']:.2f} aps={d['acc_per_step']} tool={d['tool']['per_req_tok_s']} text={d['text']['per_req_tok_s']} emitted={d['tool']['emitted']}/{d['tool']['n']}\")" 2>/dev/null || echo NA; }
k6() { python3 -c "import json;r=json.load(open('/home/milo/dsv41/knee-$1.json'))['rows'];print(' '.join(f'C{c}={a:.1f}' for c,a,_ in r))" 2>/dev/null || echo NA; }
fx() { python3 - "$1" <<'PY' 2>/dev/null || echo NA
import json,sys; d=json.load(open(f"/home/milo/dsv41/agentfix-{sys.argv[1]}.json")); r=d["res"]
print(" ".join(f"{c}={r[c]['tok_s']}({r[c]['accept']:.2f})" for c in ("tool_json","code","shell_ops","structured","prose") if c in r)+f" wacc={d['weighted_accept']:.3f}")
PY
}
replay4x2() { local L=$1; TAG="R5-$L-a" N=4 ROUNDS=2 python3 replay_c.py; TAG="R5-$L-b" N=4 ROUNDS=2 python3 replay_c.py; }
finish() { log "FINISH"; docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-R5-' | xargs -r docker stop >/dev/null || true; stop_lane; log "lane dark"; }

log "ROUND5 start"
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true); [[ -n "$other" ]] && die "other container running: $other"
for f in replay_c.py knee6.sh agent_fixture_o.sh transcript_fixture.json launch-dsv41-vllm.sh; do [[ -s $f ]] || die "missing $f"; done

up "$V13"; log "control v13 replay n=4 x2"; replay4x2 v13; bash knee6.sh R5-C-v13
ledger "| v13 ctrl | replay a $(rp R5-v13-a-n4) | b $(rp R5-v13-b-n4) | knee $(k6 R5-C-v13) |"; check_stop
up "$V12"; log "control v12 replay n=4 x2"; replay4x2 v12
ledger "| v12 ctrl | replay a $(rp R5-v12-a-n4) | b $(rp R5-v12-b-n4) |"; check_stop

stop_lane; drop_caches; tag=R5-S3-b4; sched='[[1,4,5],[5,16,1]]'; t0=$(date '+%H:%M')
TAG="$tag" OFFGB=60 UTIL=0.97 SEQS=16 CTX=1048576 SPEC="" MODEL="$MODEL" \
  EXTRA="--long-prefill-token-threshold 6144 --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$sched}" \
  bash -f launch-dsv41-vllm.sh
docker inspect "dsv41-vllm-$tag" --format '{{join .Args " "}}' | grep -qF "$sched" || die "$tag: schedule did not reach container args"
wait_bound "dsv41-vllm-$tag" 7200; smoke || die "smoke failed $tag"
grep -aq "Loaded [0-9]* configs" "dsv41-vllm-$tag.log" && log "autotune HIT" || log "autotune retuned (expected)"
{ grep -aoE "autotune cache file: [^ ]*|Loaded [0-9]+ configs|Saved [0-9]+ configs|Available KV cache memory: [0-9.]+ GiB" "dsv41-vllm-$tag.log" | head -4; } > "results/facts-$tag.txt"
log "$tag replay n=4 x2"; replay4x2 S3; bash knee6.sh "$tag"; bash agent_fixture_o.sh "$tag" || true
ledger "| S3 $sched | $t0 → $(date '+%H:%M') | replay a $(rp R5-S3-a-n4) | b $(rp R5-S3-b-n4) | knee $(k6 $tag) | $(fx $tag) |"
docker stop "dsv41-vllm-$tag" >/dev/null; docker rename "dsv41-vllm-$tag" "dsv41-vllm-$tag-EXP" 2>/dev/null || true
finish; log "ROUND5 done"

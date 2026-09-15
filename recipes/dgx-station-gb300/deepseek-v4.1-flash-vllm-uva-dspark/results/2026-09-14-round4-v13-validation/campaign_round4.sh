#!/usr/bin/env bash
# campaign_round4.sh — Round 4 on :30006. v13 (k-schedule) validation + schedule-shape sweep.
#   A  tool_json root cause : agent_fixture with CAT_ORDER=tool_json first, then tool_json alone, on v13 and on v12 control (0 boots)
#   B  real-agent replay    : replay_c.py N=1 and N=4 on v13 and on v12 control (0 boots)
#   C  knee incl. C3/C6     : knee6.sh on v13 and v12 (0 boots)
#   D  schedule shapes      : S1=[[1,3,5],[4,16,1]]  S2=[[1,2,5],[3,4,3],[5,16,1]]  (2 boots, hash hits ~150 s)
# Usage: bash campaign_round4.sh [ABC|D|ALL]. STOP-CAMPAIGN halts between steps. Ends dark unless RESTORE_REF=1.
set -euo pipefail
cd /home/milo/dsv41
V13=dsv41-vllm-v13-1M-ksched-BOUND-REF
V12=dsv41-vllm-v12-1M-k5-off60-util97-RETIRED-REF
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
LEDGER=/home/milo/dsv41/results/ledger.md
CAMPLOG=/home/milo/dsv41/results/campaign-round4-$(date +%F).log
STOPF=/home/milo/dsv41/STOP-CAMPAIGN
RESTORE_REF="${RESTORE_REF:-0}"; WHICH="${1:-ALL}"
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
up() {  # $1 container: start it if not the one running
  docker ps --format '{{.Names}}' | grep -qx "$1" && return 0
  stop_lane; drop_caches; log "starting $1"; docker start "$1" >/dev/null; wait_bound "$1" 2400; smoke || die "smoke failed on $1"; }
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
ledger() { echo "$(date +%F) ROUND4 $*" >> "$LEDGER"; log "LEDGER $*"; }
fx() { python3 - "$1" <<'PY' 2>/dev/null || echo NA
import json,sys; d=json.load(open(f"/home/milo/dsv41/agentfix-{sys.argv[1]}.json")); r=d["res"]
print(" ".join(f"{c}={r[c]['tok_s']}({r[c]['accept']:.2f})" for c in ("tool_json","code","shell_ops","structured","prose") if c in r)+f" wacc={d['weighted_accept']:.3f}")
PY
}
k6() { python3 -c "import json;r=json.load(open('/home/milo/dsv41/knee-$1.json'))['rows'];print(' '.join(f'C{c}={a:.1f}' for c,a,_ in r))" 2>/dev/null || echo NA; }
rp() { python3 -c "import json;d=json.load(open('/home/milo/dsv41/replayc-$1.json'));print(f\"agg={d['agg_tok_s']} acc={d['accept']:.2f} tool_req={d['tool']['per_req_tok_s']} text_req={d['text']['per_req_tok_s']} emitted={d['tool']['emitted']}/{d['tool']['n']}\")" 2>/dev/null || echo NA; }
finish() { log "FINISH"; docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-R4-' | xargs -r docker stop >/dev/null || true
  if [[ "$RESTORE_REF" == "1" ]]; then up "$V13"; else stop_lane; log "lane dark"; fi; }

validate_on() {  # $1 container  $2 label (v13|v12)
  local c=$1 L=$2; up "$c"
  log "A1 $L fixture tool_json-first";  CAT_ORDER="tool_json code shell_ops structured prose" bash agent_fixture_o.sh "R4-A1-$L-tjfirst"
  log "A2 $L fixture tool_json alone";  CAT_ORDER="tool_json" bash agent_fixture_o.sh "R4-A2-$L-tjalone"
  log "A3 $L fixture default order";    bash agent_fixture_o.sh "R4-A3-$L-default"
  ledger "| A $L | tjfirst $(fx R4-A1-$L-tjfirst) | tjalone $(fx R4-A2-$L-tjalone) | default $(fx R4-A3-$L-default) |"
  check_stop
  log "B $L replay n=1"; TAG="R4-B-$L" N=1 ROUNDS=1 python3 replay_c.py
  log "B $L replay n=4"; TAG="R4-B-$L" N=4 ROUNDS=2 python3 replay_c.py
  ledger "| B $L | n1 $(rp R4-B-$L-n1) | n4 $(rp R4-B-$L-n4) |"
  check_stop
  log "C $L knee6"; bash knee6.sh "R4-C-$L"; ledger "| C $L | $(k6 R4-C-$L) |"
  check_stop
}
shape() {  # $1 tag  $2 schedule json
  local tag=$1 sched=$2
  up "$V13"; log "D control knee6 for $tag"; bash knee6.sh "R4-D-ctrl-$tag"
  stop_lane; drop_caches; local t0=$(date '+%H:%M')
  TAG="$tag" OFFGB=60 UTIL=0.97 SEQS=16 CTX=1048576 SPEC="" MODEL="$MODEL" \
    EXTRA="--long-prefill-token-threshold 6144 --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$sched}" \
    bash -f launch-dsv41-vllm.sh
  docker inspect "dsv41-vllm-$tag" --format '{{join .Args " "}}' | grep -qF "$sched" || die "$tag: schedule did not reach container args"
  wait_bound "dsv41-vllm-$tag" 7200; smoke || die "smoke failed $tag"
  grep -aq "Loaded [0-9]* configs" "dsv41-vllm-$tag.log" || log "WARN $tag: autotune did not hit"
  bash knee6.sh "$tag"; bash agent_fixture_o.sh "$tag" || true
  ledger "| D $tag $sched | $t0 → $(date '+%H:%M') | ctrl $(k6 R4-D-ctrl-$tag) | cand $(k6 $tag) | $(fx $tag) |"
  docker stop "dsv41-vllm-$tag" >/dev/null; docker rename "dsv41-vllm-$tag" "dsv41-vllm-$tag-EXP" 2>/dev/null || true
  check_stop
}

log "ROUND4 start: which=$WHICH"
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true); [[ -n "$other" ]] && die "other container running: $other"
for f in agent_fixture_o.sh knee6.sh replay_c.py transcript_fixture.json launch-dsv41-vllm.sh; do [[ -s $f ]] || die "missing $f"; done
docker ps -a --format '{{.Names}}' | grep -qx "$V13" || die "v13 REF missing"; docker ps -a --format '{{.Names}}' | grep -qx "$V12" || die "v12 REF missing"
if [[ "$WHICH" == "ALL" || "$WHICH" == "ABC" ]]; then validate_on "$V13" v13; validate_on "$V12" v12; fi
if [[ "$WHICH" == "ALL" || "$WHICH" == "D" ]]; then shape R4-S1-b3 '[[1,3,5],[4,16,1]]'; shape R4-S2-mid3 '[[1,2,5],[3,4,3],[5,16,1]]'; fi
finish; log "ROUND4 done"

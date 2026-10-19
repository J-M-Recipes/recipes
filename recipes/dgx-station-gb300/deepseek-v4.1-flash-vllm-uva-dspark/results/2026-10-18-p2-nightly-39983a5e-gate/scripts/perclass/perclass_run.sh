#!/usr/bin/env bash
# perclass_run.sh — per-class decode wall-clock window on the LIVE :30006 lane. Client-side only: never stops/starts/pulls
#   containers, never touches the GPU directly. Authored 2026-10-18 by Milo (hermes milo) subagent claude-opus-5-5.
# Stages (fixed order, identical on every arm):
#   0 guard  : /v1/models serves dsv41-flash-uva; wait (<=10 min) for num_requests_running==0; record serving container (read-only)
#   1 nat    : perclass_decode.py --mode c1 --natural --fl 384  (natural lengths + prefix-cache warm pass for stages 2-3)
#   2 r1     : C1 forced, then C8 forced (FL_MAP, ignore_eos)
#   3 r2     : C1 forced, then C8 forced
#   4 x3     : agent_fixture_x3.py once (8/class acceptance canary, unedited)
#   5 table  : perclass_table.py (and the gate, if CTL_DIR/CTL are given)
# Env: ARM (required; e.g. v23, candC), OUTDIR (default ~/dsv41/1017/results-perclass-<ARM>), FL_MAP (default below),
#      PROMPTS (default ~/dsv41/bench/perclass_prompts.jsonl), CTL_DIR + CTL (optional: gate vs a control arm).
set -uo pipefail
ARM=${ARM:?set ARM}; B=$HOME/dsv41/bench; U=http://127.0.0.1:30006
OUTDIR=${OUTDIR:-$HOME/dsv41/1017/results-perclass-$ARM}; mkdir -p "$OUTDIR"
PROMPTS=${PROMPTS:-$B/perclass_prompts.jsonl}
FL_MAP=${FL_MAP:-'{"prose":256,"structured":256,"code":256,"shell_ops":256,"tool_json":256}'}
exec >> "$OUTDIR/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
log "START perclass_run ARM=$ARM OUTDIR=$OUTDIR PROMPTS=$PROMPTS sha1=$(sha1sum "$PROMPTS" | cut -c1-12) FL_MAP=$FL_MAP"
log "serving: $(docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | grep -E '^dsv41-' | tr '\n' '|')"
curl -fsS --max-time 10 $U/v1/models | grep -q dsv41-flash-uva || { log "ABORT lane not serving dsv41-flash-uva"; exit 2; }
running(){ curl -fsS --max-time 10 $U/metrics | awk '/^vllm:num_requests_running/{s+=$NF} END{print s+0}'; }
for i in $(seq 1 40); do r=$(running); [[ ${r%.*} -eq 0 ]] && break; log "WAIT lane busy running=$r"; sleep 15; done
log "lane idle check: running=$(running)"
cd "$B" || exit 2
run(){ log "RUN $*"; python3 perclass_decode.py --prompts "$PROMPTS" --outdir "$OUTDIR" "$@" || log "FAIL rc=$? $*"; }
run --tag "$ARM-nat" --mode c1 --natural --fl 384
for r in r1 r2; do
  run --tag "$ARM-$r" --mode c1 --fl-map "$FL_MAP"
  run --tag "$ARM-$r" --mode c8 --conc 8 --fl-map "$FL_MAP"
done
log "RUN agent_fixture_x3.py"
TAG="$ARM-x3" OUT="$OUTDIR/agentfix-$ARM-x3.json" python3 $HOME/dsv41/1017/agent_fixture_x3.py || log "FAIL x3 rc=$?"
if [[ -n ${CTL_DIR:-} && -n ${CTL:-} ]]; then
  python3 perclass_table.py --outdir "$OUTDIR" --arm "$ARM" --control-dir "$CTL_DIR" --control "$CTL"
fi
python3 perclass_table.py --outdir "$OUTDIR" --arm "$ARM"
log "DONE perclass_run ARM=$ARM"

#!/usr/bin/env bash
# GPQA-64K rerun (2026-09-23) — GLM-5.3-Flash NVFP4 + DFlash2 on the recipe pin, restarting the stopped Card E
# container glmf-e0 UNCHANGED (docker start: identical args — v0.5.20-cu130, draft 7d74cdd8, b7, 48 mamba slots,
# mem 0.85, 1M ctx, fp8 KV). James 2026-09-23: "go gpqa".
# Scope: rerun ONLY the Card E questions that hit the 32K cap (finish=length), same prompt/shuffle/T=0/default
# (max) effort/C8, at MAXTOK=65536; merge 1:1 by question index into the 198-q result.
# Lane :30001. Stop-and-keep at end; NO other lane restored (by instruction). STOP file honoured.
# Written by Milo (Hermes milo profile) for James Meadlock.
set -uo pipefail
export PYTHONUNBUFFERED=1
R=$HOME/glmf/gpqa64k-2026-09-23; cd "$R"
E0=$HOME/glmf/cardE-2026-09-23/e0
GPQA=$HOME/pin-hot-experts/gpqa_diamond.csv
C=glmf-e0; LOG=$R/run.log; STOPF=$R/STOP
exec >> "$LOG" 2>&1
exec 9>"$R/.lock"; flock -n 9 || { echo "LOCKED"; exit 9; }
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
finish(){ docker ps --format '{{.Names}}' | grep -qx "$C" && { docker stop "$C" >/dev/null; log "stop-and-keep $C"; }
          log "finish: no other lane restored (by instruction)"; log "===== GPQA64K END ====="; }
trap finish EXIT
log "===== GPQA64K BEGIN ====="
busy=$(docker ps --format '{{.Names}}' || true)
[[ -n $busy ]] && { log "REFUSE: containers running: $busy"; trap - EXIT; exit 4; }
[[ -f $STOPF ]] && { log "STOP seen"; exit 0; }

python3 - "$E0/gpqa-e0.jsonl" > idx.txt <<'PY'
import json,sys
print(",".join(str(r["i"]) for r in map(json.loads,open(sys.argv[1])) if r.get("finish")=="length"))
PY
N=$(tr ',' '\n' < idx.txt | grep -c .); log "rerun set: $N questions (finish=length in Card E e0 @32K)"
[[ $N -ge 1 ]] || { log "EMPTY_SET"; exit 3; }
sha256sum gpqa_diamond.py merge_gpqa.py run-gpqa64k.sh idx.txt "$E0/gpqa-e0.jsonl" "$GPQA" > SHA256SUMS
docker inspect "$C" --format '{{.Config.Image}} {{json .Args}}' > inspect.txt

sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
t0=$(date +%s); docker start "$C" >/dev/null; log "START $C (page cache dropped)"
nohup docker logs -f --since 1s "$C" > "$R/glmf-e0-restart.log" 2>&1 &
for i in $(seq 1 360); do
  curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && break
  docker ps --format '{{.Names}}' | grep -qx "$C" || { log "EXITED $C"; docker logs --tail 40 "$C" 2>&1 | tail -30; exit 5; }
  [[ -f $STOPF ]] && { log "STOP seen during boot"; exit 0; }
  sleep 5; done
curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null || { log "BOOT_TIMEOUT"; exit 6; }
log "READY after $(( $(date +%s) - t0 ))s"
grep -m1 -o "max_total_num_tokens=[0-9]*.*max_running_requests=[0-9]*" "$R/glmf-e0-restart.log" | head -1 | sed 's/^/POOL /' | while read l; do log "$l"; done

# warm + smoke: one short T=0 request must return parsed content
python3 - > warm.txt 2>&1 <<'PY'
import json,urllib.request
p={"model":"glm-5.3-flash","temperature":0,"max_tokens":256,"messages":[{"role":"user","content":"What is 17*23? Answer: X"}]}
r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30001/v1/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"}),timeout=600))
c=r["choices"][0]["message"].get("content") or ""; print("WARM_OK" if "391" in c else "WARM_BAD", repr(c[-120:]))
PY
log "$(head -1 warm.txt)"
grep -q WARM_OK warm.txt || { log "WARM gate failed"; exit 7; }
[[ -f $STOPF ]] && { log "STOP seen"; exit 0; }

log "GPQA rerun: n=$N C8 T=0 default(max) effort MAXTOK=65536"
TAG=e0-gpqa64k OUT="$R/gpqa-e0-64k.jsonl" BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-flash API_KEY=none \
  CONC=8 MAXTOK=65536 THINKING=1 IDX="@$R/idx.txt" python3 "$R/gpqa_diamond.py" "$GPQA" > "$R/gpqa64k.log" 2>&1
log "$(grep 'GPQA-Diamond' "$R/gpqa64k.log" || echo 'NO_SUMMARY_LINE')"
python3 "$R/merge_gpqa.py" "$E0/gpqa-e0.jsonl" "$R/gpqa-e0-64k.jsonl" "$R/merged-summary.json" > merge.txt 2>&1
log "$(cat merge.txt)"
[[ -s $R/merged-summary.json ]] && log "ROWS_OK $(wc -l < "$R/gpqa-e0-64k.jsonl") rerun rows" || log "ROWS_MISSING"
echo GPQA64K_DETACHED_END

#!/usr/bin/env bash
# CARD H2 (2026-09-24): which transformers gives the right model function? Score the vLLM FP8-oracle text
# (independent engine) on the stopped glmf-stock (tf 5.12.1) and glmf-tf5 (tf 5.16.1) containers, one at a time.
# Also re-score the w2-a text on stock to confirm stock still reproduces Card D exactly (instrument check).
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardH-2026-09-24/h2; mkdir -p "$D"
exec > "$D/runner.log" 2>&1
TF=$HOME/window-20260913/scripts/tf_noninferiority.py
export MODEL=glm-5.3-flash API_KEY=""
log(){ printf '%s %s\n' "$(date -u '+%FT%TZ')" "$*"; }
wait_ready(){ for i in $(seq 1 240); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "READY $1 after $((i*5))s"; return 0; }
  docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "EXITED $1"; return 5; }; sleep 5; done; return 5; }
log "H2 START"
for c in glmf-stock glmf-tf5; do
  for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do docker stop "$n" >/dev/null; done; sleep 3
  log "start $c"; docker start "$c" >/dev/null; wait_ready "$c" || continue
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$D/warm-$c.txt" 2>&1
  BASE=http://127.0.0.1:30001/v1 python3 kl_gate2.py score kl-ref-FP8-oracle.json "$D/kl-$c-on-FP8-text.json" > "$D/kl-score-$c.txt" 2>&1
  python3 kl_gate2.py cmp kl-ref-FP8-oracle.json "$D/kl-$c-on-FP8-text.json" > "$D/kl-cmp-$c.txt" 2>&1; log "$c vs FP8: $(tail -1 $D/kl-cmp-$c.txt)"
  LANE=$c BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$D/tf-$c.json" > "$D/tf-log-$c.txt" 2>&1
  python3 "$TF" --compare cardD-2026-09-21/newdf/tf-newdf.json "$D/tf-$c.json" > "$D/tf-vs-newdf-$c.txt" 2>&1; log "$c TF vs CardD newdf: $(grep -A4 '"bar"' $D/tf-vs-newdf-$c.txt | tr -d '\n ' )"
  docker stop "$c" >/dev/null
done
python3 kl_gate2.py cmp kl-ref-FP8-oracle.json kl-cand-nvidia-on-FP8-text.json > "$D/kl-cmp-2026-09-11-nvidia.txt" 2>&1; log "09-11 nvidia (00143e9c, tf 5.12.1) vs FP8: $(tail -1 $D/kl-cmp-2026-09-11-nvidia.txt)"
log "H2 END (all glmf-* stopped)"

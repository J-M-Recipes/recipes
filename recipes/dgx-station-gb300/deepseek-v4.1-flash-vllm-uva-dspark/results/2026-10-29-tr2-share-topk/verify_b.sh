#!/usr/bin/env bash
# verify_b.sh — if the bracket's arm-B verify window comes back thin, prove the identity claim directly:
# boot the SAME arm-B config with L2_VERIFY=1 from env, drive decode traffic at temperature 1.0, and read the
# per-hit verify counters (each hit recomputes the original op and torch.equal()s the table).
set -uo pipefail
R=$HOME/dsv41/m3/tr2/results-cardTR2
T=$HOME/dsv41/m3/tr2
U=http://127.0.0.1:30006
NAME=dsv41-vllm-1017-TR2-BV
RUN=$T/bv/ar
mkdir -p "$RUN" "$R/BV"
exec >> "$R/BV/verify.log" 2>&1
log(){ echo "[$(date '+%F %T')] $*"; }
log "verify-b: starting $NAME (SHARE=2 VERIFY=1)"
sudo -n iptables -C INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null \
  || sudo -n iptables -I INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset
log "ISOLATE on"
for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do docker stop -t 60 "$n" >/dev/null; log "stopped $n"; done
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
NAME=$NAME RUN=$RUN ROWMAP=rowmap-mixed-ar-v1.json \
  SPEC_EXTRA=',"draft_sample_method":"probabilistic"' \
  L2_SHARE=2 L2_VERIFY=1 L2_STATS_EVERY=1 L2_VERIFY_FILE= \
  bash "$T/launch_tr2.sh" || { log "launch failed"; exit 5; }
for i in $(seq 1 180); do curl -sf -m 3 "$U/v1/models" >/dev/null 2>&1 && { log "bound after $((i*5))s"; break; }; sleep 5; done
touch "$RUN/GO"; printf '%s' '{"freeze": false, "baseline": true}' > "$RUN/ctrl.json"
sleep 90
for i in 1 2 3 4 5 6 7 8; do
  curl -s -m 240 "$U/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"Write a careful technical paragraph about why consensus in distributed systems needs a quorum."}],"max_tokens":256,"temperature":1.0,"chat_template_kwargs":{"thinking":false}}' \
    -o /dev/null -w "" &
done; wait
sleep 3
docker logs "$NAME" 2>&1 | tr '\r' '\n' | grep "L2 stats" | tail -2 > "$R/BV/verify-stats.txt"
docker logs "$NAME" 2>&1 | tr '\r' '\n' | grep "VERIFY MISMATCH" | head -5 >> "$R/BV/verify-stats.txt"
log "verify stats:"; cat "$R/BV/verify-stats.txt" | cut -c1-300
docker stop -t 60 "$NAME" >/dev/null; log "stopped $NAME (kept)"
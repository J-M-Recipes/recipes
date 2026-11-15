#!/usr/bin/env bash
# Milo takeover 2026-11-14: speculation-schedule arms on the stock control tree (W1 FP8 OFF). One boot per arm.
# arms2.tsv (re-read each loop, first `todo` wins): status<TAB>name<TAB>schedule   schedule = stock | JSON band list
# Cells mirror run_serving_screen.py (gt_mix_drive pass/conc), same file names, so the same analysis applies.
# No promotion, no restore: every arm ends DARK. Touch $T/ABORT to stop.
set -u
R=$HOME/dsv41/allround-20261112
T=$R/milo-takeover/spec
GR=$HOME/dsv41/garner
MODEL=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
GT=$HOME/dsv41/decode/bench/gt_mix_drive.py
MF=$HOME/dsv41/m3/logs/MAINTENANCE
STOCK='[[1,4,5],[5,8,3],[9,32,3]]'
exec 3>>"$T/runner.log"
log(){ printf '%s %s\n' "$(date '+%F %T')" "$*" >&3; }
running(){ docker ps --format '{{.Names}}'; }
lane(){ env WORK="$W" DS41_HOME="$W/home" MODEL_DIR="$MODEL" IMAGE="$IMAGE" DS41_IMAGE="$IMAGE" \
          timeout "$2" bash "$GR/launcher/launch-pg61.sh" "$1"; }
down(){ lane down 240 >>"$1" 2>&1; local left; left=$(running | grep -xE 'ds41|ds41-sidecar' || true)
        [ -z "$left" ] || { docker stop -t 10 ds41 ds41-sidecar >>"$1" 2>&1; }
        [ -z "$(running | grep -xE 'ds41|ds41-sidecar' || true)" ]; }
setstat(){ python3 - "$T/arms2.tsv" "$1" "$2" <<'PY'
import sys
p, name, st = sys.argv[1:]
out = []
for r in open(p).read().splitlines():
    f = r.split("\t")
    if len(f) >= 2 and f[1] == name and f[0] == "todo":
        f[0] = st
    out.append("\t".join(f))
open(p, "w").write("\n".join(out) + "\n")
PY
}
finish(){ log "RUNNER END ($1)"; exit 0; }
metrics(){ curl -sf -m 10 http://127.0.0.1:30006/metrics; }

log "RUNNER START pid $$"
grep -qi allround "$MF" || { log "ABORT no allround MAINTENANCE flag"; exit 1; }
[ "$(sha256sum < "$GT" | cut -c1-64)" = 80838e09af3fc389759f7d8aee7157f35ae2023e3c42ec8dbab4e84c0a7d1d61 ] || { log "ABORT driver drift"; exit 1; }
while :; do
  [ -e "$T/ABORT" ] && finish "ABORT file, lane dark"
  line=$(awk -F'\t' '$1=="todo"{print; exit}' "$T/arms2.tsv")
  [ -n "$line" ] || finish "no todo arms, lane dark"
  IFS=$'\t' read -r _ name sched tree <<<"$line"
  W=$R/${tree:-serve-spec}
  [ -f "$W/tree/serve/serve.sh.stock" ] || { log "ARM $name ABORT no serve.sh.stock in $W"; setstat "$name" abort; finish "no stock serve.sh"; }
  OUT=$T/arm-$name
  if [ -e "$OUT" ]; then log "ARM $name SKIP (dir exists)"; setstat "$name" dup; continue; fi
  mkdir -p "$OUT"
  [ "$sched" = stock ] && sched=$STOCK
  log "ARM $name START schedule=$sched tree=$W"
  if [ -n "$(running)" ]; then log "ARM $name ABORT running: $(running | tr '\n' ' ')"; setstat "$name" abort; finish "running containers"; fi
  free_g=$(df -Pk / | awk 'NR==2{print int($4/1048576)}')
  [ "$free_g" -ge 200 ] || { log "ARM $name ABORT disk ${free_g}G"; setstat "$name" abort; finish "disk"; }
  q=0; for i in 1 2 3 4 5 6; do
    if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$1>=2000||$2>=10{b=1} END{exit b}'; then q=1; break; fi; sleep 10; done
  [ "$q" = 1 ] || { log "ARM $name ABORT GPUs not quiet (needs hands)"; setstat "$name" abort; finish "gpu busy"; }
  # serve.sh = stock copy with only the band list replaced
  python3 - "$W/tree/serve/serve.sh.stock" "$W/tree/serve/serve.sh" "$STOCK" "$sched" <<'PY'
import sys
src, dst, old, new = sys.argv[1:]
s = open(src).read()
n = s.count(old)
assert n >= 1, "stock band list not found"
open(dst, "w").write(s.replace(old, new))
print("replaced", n)
PY
  diff "$W/tree/serve/serve.sh.stock" "$W/tree/serve/serve.sh" >"$OUT/serve.diff"; hunks=$(grep -c '^[<>]' "$OUT/serve.diff")
  log "ARM $name serve.sh changed lines=$hunks"
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; sleep 30
  log "ARM $name boot"
  if ! lane up 2400 >"$OUT/up.log" 2>&1; then
    log "ARM $name BOOTFAIL (see up.log)"; docker logs ds41 >"$OUT/server.log" 2>&1; docker logs ds41-sidecar >"$OUT/sidecar.log" 2>&1
    down "$OUT/down.log" || log "ARM $name down incomplete"
    setstat "$name" bootfail; finish "boot failure, lane dark"
  fi
  docker logs ds41 >"$OUT/server.log" 2>&1
  docker inspect ds41 --format '{{json .Args}}' >"$OUT/args.json"
  kv=$(grep -o 'GPU KV cache size: [0-9,]* tokens' "$OUT/server.log" | tail -1 | tr -dc 0-9)
  live=$(grep -o 'num_speculative_tokens_per_batch_size[^]]*\]\]' "$OUT/args.json" | head -1)
  log "ARM $name BOUND kv=$kv live=$live"
  ok=1; [ "$kv" = 11696582 ] || log "ARM $name KV DRIFT $kv (warn: measured anyway, capacity gate not met)"
  grep 'FP8_MASK label\|compressed_fp8=' "$OUT/server.log" | tail -2 | cut -c1-200 >&3
  jobs="warm-A:pass --set A|warm-B:pass --set B|warm-c8:conc --n 8 --tag takeover-spec-warm-c8|warm-c32:conc --n 32 --tag takeover-spec-warm-c32"
  for r in 1 2 3; do jobs="$jobs|c1-A-r$r:pass --set A|c1-B-r$r:pass --set B|c8-r$r:conc --n 8 --tag takeover-spec-c8-r$r|c32-r$r:conc --n 32 --tag takeover-spec-c32-r$r"; done
  IFS='|' read -r -a cells <<<"$jobs"
  for c in "${cells[@]}"; do
    [ "$ok" = 1 ] || break
    cn=${c%%:*}; ca=${c#*:}
    metrics >"$OUT/$cn-metrics-before.txt" || { log "ARM $name metrics fail before $cn"; ok=0; break; }
    # shellcheck disable=SC2086
    if ! W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva timeout 600 python3 "$GT" $ca --out "$OUT/$cn.json" >"$OUT/$cn.log" 2>&1; then
      log "ARM $name CELL $cn FAILED"; ok=0; fi
    metrics >"$OUT/$cn-metrics-after.txt" || { log "ARM $name SERVER DOWN after $cn"; ok=0; }
    grep -q '"trip": null' "$W/wdp3.status.json" || { log "ARM $name WATCHDOG TRIP at $cn"; ok=0; }
    [ "$ok" = 1 ] && log "ARM $name cell $cn $(python3 -c "
import json,sys
d=json.load(open(sys.argv[1])); a=d.get('agg',{})
print({k:a[k] for k in ('decode_tok_s','step_ms','tokens_per_step','agg_tok_s','stream_decode_tok_s_p50','errors') if k in a})" "$OUT/$cn.json" 2>&1 | cut -c1-200)"
  done
  docker logs ds41 >"$OUT/server-final.log" 2>&1
  st=done; [ "$ok" = 1 ] || st=fail
  log "ARM $name CELLS $st"
  if down "$OUT/down.log"; then log "ARM $name DOWN (dark)"; else log "ARM $name DOWN INCOMPLETE"; setstat "$name" "$st"; finish "down incomplete (needs hands)"; fi
  setstat "$name" "$st"
done

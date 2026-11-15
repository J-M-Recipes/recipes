#!/usr/bin/env bash
# Milo takeover 2026-11-14: W13 FP8-attention bisect. One arm per boot; arms come from arms.tsv (re-read every loop,
# first `todo` line wins): status<TAB>name<TAB>layers<TAB>widths[<TAB>flag]   layers/widths: all | none | 2-7,9
# flag: fp8off = boot with DS41_W1_COMPRESSED_FP8=0 (pure control through the same tree).
# No promotion, no restore: every arm ends DARK. Touch $T/ABORT to stop after (or during) the current arm.
set -u
R=$HOME/dsv41/allround-20261112
T=$R/milo-takeover/bisect
W=$R/serve-bisect
GR=$HOME/dsv41/garner
REF=$R/serve-quality-a2
MODEL=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
DRV=$HOME/dsv41/decode/w0/w0_drive.py
MF=$HOME/dsv41/m3/logs/MAINTENANCE
exec 3>>"$T/runner.log"
log(){ printf '%s %s\n' "$(date '+%F %T')" "$*" >&3; }
running(){ docker ps --format '{{.Names}}'; }
lane(){ env WORK="$W" DS41_HOME="$W/home" MODEL_DIR="$MODEL" IMAGE="$IMAGE" DS41_IMAGE="$IMAGE" \
          timeout "$2" bash "$GR/launcher/launch-pg61.sh" "$1"; }
down(){ lane down 240 >>"$1" 2>&1; local left; left=$(running | grep -xE 'ds41|ds41-sidecar' || true)
        [ -z "$left" ] || { docker stop -t 10 ds41 ds41-sidecar >>"$1" 2>&1; }
        [ -z "$(running | grep -xE 'ds41|ds41-sidecar' || true)" ]; }
setstat(){ python3 - "$T/arms.tsv" "$1" "$2" <<'PY'
import sys
p, name, st = sys.argv[1:]
rows = open(p).read().splitlines()
out = []
for r in rows:
    f = r.split("\t")
    if len(f) >= 2 and f[1] == name and f[0] == "todo":
        f[0] = st
    out.append("\t".join(f))
open(p, "w").write("\n".join(out) + "\n")
PY
}
finish(){ log "RUNNER END ($1)"; exit 0; }

log "RUNNER START pid $$"
grep -qi allround "$MF" || { log "ABORT no allround MAINTENANCE flag"; exit 1; }
[ "$(sha256sum < "$DRV" | cut -c1-64)" = 6c1fb1be1080d6362ef29ebbf6e8a4734f47a948df447f1c0ccd9c5a7adbe561 ] || { log "ABORT driver drift"; exit 1; }
while :; do
  [ -e "$T/ABORT" ] && finish "ABORT file, lane dark"
  line=$(awk -F'\t' '$1=="todo"{print; exit}' "$T/arms.tsv")
  [ -n "$line" ] || finish "no todo arms, lane dark"
  IFS=$'\t' read -r _ name layers widths flag <<<"$line"
  OUT=$T/arm-$name
  if [ -e "$OUT" ]; then log "ARM $name SKIP (dir exists)"; setstat "$name" dup; continue; fi
  mkdir -p "$OUT"
  log "ARM $name START layers=$layers widths=$widths flag=${flag:-}"
  # preflight: nothing running, GPUs quiet, disk
  if [ -n "$(running)" ]; then log "ARM $name ABORT foreign/leftover running: $(running | tr '\n' ' ')"; setstat "$name" abort; finish "running containers"; fi
  free_g=$(df -Pk / | awk 'NR==2{print int($4/1048576)}')
  [ "$free_g" -ge 200 ] || { log "ARM $name ABORT disk ${free_g}G"; setstat "$name" abort; finish "disk"; }
  q=0; for i in 1 2 3 4 5 6; do
    if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$1>=2000||$2>=10{b=1} END{exit b}'; then q=1; break; fi; sleep 10; done
  [ "$q" = 1 ] || { log "ARM $name ABORT GPUs not quiet (needs hands)"; setstat "$name" abort; finish "gpu busy"; }
  # mask + flag
  python3 - "$W/tree/fastpath/w1attn/fp8_mask.json" "$name" "$layers" "$widths" "${flag:-}" <<'PY'
import json, sys
path, name, layers, widths, flag = sys.argv[1:]
def spec(s):
    if s == "all": return None
    if s == "none": return []
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))
open(path, "w").write(json.dumps({"label": name, "layers": spec(layers), "widths": spec(widths),
                             "pdl": flag != "nopdl",
                             "fp8_plan": "bitwise_seq" if flag == "seq" else None,
                             "fp8_extra": flag if flag in ("stock_after", "stock_before") else None}) + "\n")
PY
  fp8=1; [ "${flag:-}" = fp8off ] && fp8=0
  sed -i "s/^DS41_W1_COMPRESSED_FP8=.*/DS41_W1_COMPRESSED_FP8=$fp8/" "$W/policy.env"
  cp "$W/tree/fastpath/w1attn/fp8_mask.json" "$W/policy.env" "$OUT/"
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; sleep 30
  log "ARM $name boot"
  if ! lane up 1800 >"$OUT/up.log" 2>&1; then
    log "ARM $name BOOTFAIL (see up.log)"; docker logs ds41 >"$OUT/server.log" 2>&1; docker logs ds41-sidecar >"$OUT/sidecar.log" 2>&1
    down "$OUT/down.log" || log "ARM $name down incomplete"
    setstat "$name" bootfail; finish "boot failure, lane dark"
  fi
  docker logs ds41 >"$OUT/server.log" 2>&1
  kv=$(grep -o 'GPU KV cache size: [0-9,]* tokens' "$OUT/server.log" | tail -1 | tr -dc 0-9)
  grep 'FP8_MASK\|captured decode attention calls\|compressed_fp8=' "$OUT/server.log" | cut -c1-900 >"$OUT/engage.txt"
  log "ARM $name BOUND kv=$kv $(grep -c . "$OUT/engage.txt") engage lines; $(grep 'FP8_MASK label' "$OUT/engage.txt" | tail -1 | sed 's/.*FP8_MASK/FP8_MASK/' | cut -c1-300)"
  ok=1
  [ "$kv" = 11696582 ] || { log "ARM $name KV DRIFT $kv"; ok=0; }
  for rep in 0 1; do
    if ! W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva timeout 300 python3 "$DRV" parity "$OUT/r$rep.jsonl" >"$OUT/r$rep.log" 2>&1; then
      log "ARM $name quality rep $rep FAILED"; ok=0; break; fi
  done
  grep -q '"trip": null' "$W/wdp3.status.json" || { log "ARM $name WATCHDOG TRIP"; ok=0; }
  if [ "$ok" = 1 ]; then
    v=$(python3 "$T/compare.py" "$REF" "$OUT" 2 2>&1 | tail -1); log "ARM $name $v"; st=done
  else st=fail; fi
  docker logs ds41 >"$OUT/server-final.log" 2>&1
  if down "$OUT/down.log"; then log "ARM $name DOWN (dark)"; else log "ARM $name DOWN INCOMPLETE"; setstat "$name" "$st"; finish "down incomplete (needs hands)"; fi
  setstat "$name" "$st"
done

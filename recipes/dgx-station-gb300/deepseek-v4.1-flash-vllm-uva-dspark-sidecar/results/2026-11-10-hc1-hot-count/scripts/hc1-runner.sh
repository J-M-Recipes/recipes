#!/bin/bash
# Card HC1 (Milo, 2026-11-10, James: "try the hot count test"): Pg61 with HC_HOT hot experts per layer instead of 255.
# One variable: the rowmap (Garner's h255 hot set + HC_HOT-255 promoted experts per layer; the RTX holds 384-HC_HOT).
#   control A  = cells on the live Pg61 lane (no window)
#   arm        = Pg61 down -> copy of the Pg61 tree with the new rowmap, launched by the unmodified launch-pg61.sh
#                (WORK=$H/work, own DS41_HOME, p3 watchdog ENFORCE) -> same cells -> down
#   restore    = Pg61 (restore-pg61-after-hc1.sh, a clone of the gt1 restore) -> control A2 = same cells again
# Refuses unless Pg61 is serving and nothing else runs. Every exit path after Pg61 is down restores Pg61, unless
# containers or the GB300 do not come free (then the lane stays DARK and LANE-OWNER says needs hands).
# The frozen production build ($GR/pg61-work) is never written. Usage: HC_STAMP=<stamp> [HC_HOT=275] bash hc1-runner.sh
set -u
: "${HC_STAMP:?}"
HC_HOT=${HC_HOT:-275}
GR=$HOME/dsv41/garner; DEC=$HOME/dsv41/decode; H=$HOME/dsv41/hc1
W=$H/work; HM=$H/home; R=$H/receipts/$HC_STAMP
PG=$GR/pg61-work; PGH=$GR/home
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
MAP=$DEC/docs/maps/hot-garner255-plus-$HC_HOT.json
RM=rowmap-mix-v1-h$HC_HOT.json
LO1=$HOME/dsv41/m3/LANE-OWNER.txt; LO2=$HOME/dsv41/m3/logs/LANE-OWNER.txt
MF=$HOME/dsv41/m3/logs/MAINTENANCE
mkdir -p "$R" || exit 9
exec 3>>"$R/runner.log"
say() { echo "$(date '+%F %T %Z') [hc1] $*" >&3; }
note() { echo "- $(date '+%F %T %Z') $*" | tee -a "$LO1" >> "$LO2"; }
names() { docker ps --format '{{.Names}}' | sort | tr '\n' ' '; }
serving() { curl -sf -m 5 localhost:30006/v1/models | grep -q dsv41-flash-uva; }
refuse() { say "REFUSED: $*"; exit 2; }

exec 9>"$H/.hc1.lock"; flock -n 9 || refuse "another hc1 run holds the lock"
[ "$(names)" = "ds41 ds41-sidecar " ] || refuse "containers are not exactly Pg61's: $(names)"
serving || refuse "Pg61 not serving"
[ ! -e "$MF" ] || refuse "a MAINTENANCE flag is already up: $(cat "$MF")"
pgrep -af "[c]ard-gt1|[r]estore-pg61|[r]ollback-pg61|[a]5-driver|[a]5-v28" >/dev/null && refuse "another runner is active"
[ "$(df -Pk / | awk 'NR==2{print int($4/1048576)}')" -ge 200 ] || refuse "< 200 GiB free on /"
for f in "$MAP" "$H/hc1_make_rowmap.py" "$H/hc1-cells.sh" "$PG/policy.env" "$PG/sidecar_watchdog_p3.py" \
         "$PG/tree/fastpath/hook/rowmap-mix-v1-h255.json" "$PG/tree/config/recipe.env" "$GR/launcher/launch-pg61.sh" \
         "$GR/promote/restore-pg61-after-gt1.sh" "$DEC/bench/gt_mix_drive.py"; do [ -f "$f" ] || refuse "missing $f"; done
grep -q '"trip": null' "$PG/wdp3.status.json" || refuse "Pg61 watchdog status is not trip: null"
bash -n "$H/hc1-cells.sh" || refuse "syntax: hc1-cells.sh"

# ---- build the arm (CPU only, Pg61 still serving)
build() {
  mkdir -p "$W" "$HM/state" || return 1
  [ -d "$W/tree" ] && chmod -R u+w "$W/tree"
  rsync -a --delete --exclude __pycache__ "$PG/tree/" "$W/tree/" || return 1
  chmod -R u+w "$W/tree" || return 1   # the production tree is frozen read-only; the copy must take one new file
  cp -f "$PG/policy.env" "$PG/sidecar_watchdog_p3.py" "$W/" || return 1   # -f: the sources are read-only, so a rerun must replace
  python3 "$H/hc1_make_rowmap.py" "$W/tree/fastpath/hook/rowmap-mix-v1-h255.json" "$MAP" "$W/tree/fastpath/hook/$RM" >&3 || return 1
  python3 - "$W/tree/config/recipe.env" "$RM" <<'PY' || return 1
import sys
p, rm = sys.argv[1:3]
s = open(p).read(); old = "DS41_ROWMAP='rowmap-mix-v1-h255.json'"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, f"DS41_ROWMAP='{rm}'"))
PY
  local d; d=$(diff -r -x __pycache__ "$PG/tree" "$W/tree" | grep -c '^[<>]')
  local o; o=$(diff -rq -x __pycache__ "$PG/tree" "$W/tree" | wc -l)
  [ "$d" = 2 ] && [ "$o" = 2 ] || { say "FAIL: arm tree differs from Pg61's by $d lines / $o files (want 2 / 2)"; return 1; }
  cmp -s "$PG/policy.env" "$W/policy.env" || return 1
  cp -n "$PGH/config.env" "$HM/config.env" 2>/dev/null || true
  for c in jit flashinfer b12x vllm; do   # copies of Pg61's kernel caches; the arm never writes Pg61's own
    [ -d "$PGH/state/$c" ] && [ ! -d "$HM/state/$c" ] && { rsync -a --exclude ds41-fi-tune "$PGH/state/$c/" "$HM/state/$c/" || return 1; }   # ds41-fi-tune is rewritten from the pinned tune file at every boot (root-only)
  done
  rm -f "$W"/wdp3.* "$W/soak-start.json"
  sha256sum "$W/tree/fastpath/hook/$RM" "$W/tree/config/recipe.env" "$W/tree/fastpath/libw6mk.so" > "$R/arm-sha256.txt"
  say "built arm: tree = Pg61 tree + $RM + recipe.env rowmap line (2 changed lines, 2 files); $(head -1 "$R/arm-sha256.txt" | cut -c1-16)"
}
build || refuse "arm build failed (nothing touched)"

# restore script: clone of the gt1 restore with the window name changed
( cd "$GR/promote" && python3 - <<'PY' && bash -n restore-pg61-after-hc1.sh
s = open("restore-pg61-after-gt1.sh").read()
a = "restore-after-gt1-"; b = "after the gracetier card GT1 window"
assert s.count(a) == 1 and s.count(b) == 2, (s.count(a), s.count(b))
open("restore-pg61-after-hc1.sh", "w").write(s.replace(a, "restore-after-hc1-").replace(b, "after the hot-count card HC1 window"))
PY
) || refuse "restore clone failed"
say "restore clone: $(diff "$GR/promote/restore-pg61-after-gt1.sh" "$GR/promote/restore-pg61-after-hc1.sh" | grep -c '^[<>]') changed lines"
say "preflight OK: HC_HOT=$HC_HOT; $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';') $(free -g | awk 'NR==2{print "MemAvailable " $7 " GiB"}')"

[ "${HC_PREFLIGHT_ONLY:-0}" = 1 ] && { say "PREFLIGHT_ONLY: stopping before control A"; exit 0; }

# ---- control A on the live lane
say "== control A (live Pg61, no window)"
HC_OUT="$R/ctrlA" HC_ARM=ctrlA bash "$H/hc1-cells.sh" >&3 2>&1
cp "$PGH/state/sidecar-logs/peer3_stats.json" "$R/ctrlA-peer3_stats.json" 2>/dev/null
docker logs ds41 2>&1 | grep -E "Available KV cache memory|GPU KV cache size|MEGA_PEER rowmap|Actual usage" > "$R/ctrlA-bootfacts.txt"
serving && [ "$(names)" = "ds41 ds41-sidecar " ] || { say "ABORT: Pg61 not healthy after control A; nothing taken down"; exit 1; }

# ---- window
DOWN=0; DONE=0
finish() {
  [ "$DONE" = 1 ] && return; DONE=1
  [ "$DOWN" = 1 ] || { say "END (lane never taken down)"; return; }
  say "== arm down"
  docker logs ds41 > "$R/arm-server.log" 2>&1 || true
  docker logs ds41-sidecar > "$R/arm-sidecar.log" 2>&1 || true
  cp "$W"/wdp3.log "$W"/wdp3.status.json "$R/" 2>/dev/null
  ( cd "$GR" && WORK=$W DS41_HOME=$HM bash launcher/launch-pg61.sh down ) > "$R/arm-down.log" 2>&1
  sleep 10
  local used=99999 i
  for i in 1 2 3 4 5 6; do used=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '/GB300/{print $2}'); [ "${used:-99999}" -lt 2000 ] && break; sleep 10; done
  if [ -n "$(names)" ] || [ "${used:-99999}" -ge 2000 ]; then
    say "FAIL: not restoring: containers [$(names)] GB300 used ${used} MiB"
    note "CARD HC1 ($HC_STAMP): restore NOT run (containers [$(names)], GB300 ${used} MiB); lane DARK, needs hands"
    return
  fi
  say "== restore Pg61"
  bash "$GR/promote/restore-pg61-after-hc1.sh" 9>&- 3>&-; local rrc=$?
  say "restore rc=$rrc; $(docker ps --format '{{.Names}} {{.Status}}' | tr '\n' ';')"
  if [ "$rrc" = 0 ] && serving; then
    mv "$MF" "$MF.stale-hc1-$HC_STAMP" 2>/dev/null
    say "== control A2 (restored Pg61)"
    HC_OUT="$R/ctrlA2" HC_ARM=ctrlA2 bash "$H/hc1-cells.sh" >&3 2>&1
    docker logs ds41 2>&1 | grep -E "Available KV cache memory|GPU KV cache size|MEGA_PEER rowmap|Actual usage" > "$R/ctrlA2-bootfacts.txt"
    note "CARD HC1 END (Milo, $HC_STAMP, lane = Pg61): restore rc=0; arm verdict ${ARM_STATE:-unknown}; receipts $R"
    say "CARD HC1 END (lane = Pg61); arm ${ARM_STATE:-unknown}; serving=$(serving && echo yes || echo NO)"
  else
    note "CARD HC1 ($HC_STAMP): RESTORE FAILED rc=$rrc; lane DARK; MAINTENANCE left up; receipts $R"
    say "CARD HC1 END: RESTORE FAILED rc=$rrc; lane DARK"
  fi
}
trap finish EXIT
trap 'say "signal"; exit 1' INT TERM HUP

echo "card HC1 $HC_STAMP (Milo; James: try the hot count test) -- Pg61 with $HC_HOT hot experts on :30006; Pg61 comes back at this runner's END line" > "$MF"
note "CARD HC1 START (Milo, $HC_STAMP, James go): Pg61 down -> Pg61 tree with $RM ($HC_HOT hot / $((384-HC_HOT)) cold per layer) -> cells -> Pg61 restore. log $R/runner.log"
say "== Pg61 down"
( cd "$GR" && WORK=$PG DS41_HOME=$PGH bash launcher/launch-pg61.sh down ) > "$R/pg61-down.log" 2>&1
say "Pg61 down rc=$? ($(tail -1 "$R/pg61-down.log" | cut -c1-140))"
DOWN=1; ARM_STATE=boot-failed; sleep 5
[ -z "$(names)" ] || { say "FAIL: containers still up after Pg61 down"; exit 1; }
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
say "$(free -g | awk 'NR==2{print "MemAvailable " $7 " GiB"}'); $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')"
say "== arm up (launch-pg61.sh up, WORK=$W)"
( cd "$GR" && WORK=$W MODEL_DIR=$MODEL_DIR DS41_HOME=$HM bash launcher/launch-pg61.sh up 9>&- 3>&- ) > "$R/arm-up.log" 2>&1
urc=$?
sed -i -E 's/(API key[^:]*: *)[0-9a-fA-F]{16,}/\1REDACTED/' "$R/arm-up.log"
say "arm up rc=$urc ($(tail -2 "$R/arm-up.log" | tr '\n' ' ' | cut -c1-260))"
[ "$urc" = 0 ] && serving || { say "FAIL: arm did not come up"; docker logs ds41 2>&1 | tail -5 | cut -c1-400 >&3; exit 1; }
docker logs ds41 2>&1 | grep -E "Available KV cache memory|GPU KV cache size|MEGA_PEER rowmap|Actual usage" > "$R/arm-bootfacts.txt"
docker inspect -f '{{json .Args}}' ds41 > "$R/arm-args.json"; docker inspect -f '{{json .Config.Env}}' ds41 > "$R/arm-env.json"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader > "$R/arm-nvidia-smi.txt"
say "arm boot facts: $(cut -c60-220 "$R/arm-bootfacts.txt" | tr '\n' '|' | cut -c1-600); $(tr '\n' ';' < "$R/arm-nvidia-smi.txt")"
ARM_STATE=cells-running
HC_OUT="$R/arm" HC_ARM=arm bash "$H/hc1-cells.sh" >&3 2>&1
cp "$HM/state/sidecar-logs/peer3_stats.json" "$R/arm-peer3_stats.json" 2>/dev/null
cp "$HM/state/sidecar-logs/peer_stats.json" "$R/arm-peer_stats.json" 2>/dev/null
if serving && grep -q '"trip": null' "$W/wdp3.status.json"; then ARM_STATE=cells-done-alive-notrip; else ARM_STATE=cells-done-DEAD-or-TRIP; fi
say "arm after cells: $ARM_STATE"
exit 0

#!/usr/bin/env bash
# wd3_fault.sh <outdir> — CARD R1 phase P2: fault-inject the DEPLOYED p3 watchdog (the process launch-pg61.sh armed,
# file ba8373e9). Method = G3b: 4 streaming clients, `docker pause ds41-sidecar`, then ctrl=1 at kOffCtrl=384 so the
# persistent RTX kernel exits and cannot relaunch (the pause alone freezes the host loop; ctrl=1 makes it fatal).
# 2026-10-31 (Milo) changes vs W3's draft: no second watchdog is armed (the draft armed a duplicate enforce instance
# writing the production status file); TRIP/fail-closed lines are read from the production log, only lines after the
# fault; the client wait is a real poll (a subshell `wait` cannot wait on the parent's child).
# Exits 0 only if B1-B7 of R1-PREREG.md hold. Does NOT restore: the runner runs `launch-pg61.sh down` next.
set -uo pipefail
OUT=${1:?usage: wd3_fault.sh <outdir>}
GR=${GR:-$HOME/dsv41/garner}; LW=$GR/pg61-work; WDP3=$LW/sidecar_watchdog_p3.py; WLOG=$LW/wdp3.log; WST=$LW/wdp3.status.json
SIDE=ds41-sidecar; SRV=ds41; U=http://127.0.0.1:30006; M=dsv41-flash-uva
PAUSE_BAR=6.0; NOSTREAM_BAR=0.5
mkdir -p "$OUT" || exit 5
exec 3>> "$OUT/fault.log"; exec >> "$OUT/fault.log" 2>&1
log(){ printf '%s [P2] %s\n' "$(date '+%F %T %Z')" "$*" >&3; }
now(){ date +%s.%N; }
wdp(){ ps -eo pid=,ppid=,comm=,args= | awk '$3=="python3" && index($0,"sidecar_watchdog_p3.py") && index($0,"--mode enforce"){print $1, $2}'; }

sha256sum "$WDP3" | tee "$OUT/wd3.sha256" >&3
python3 "$GR/test_watchdog_p3.py" > "$OUT/unit.txt" 2>&1; log "unit tests rc=$? $(tail -1 "$OUT/unit.txt")"
wdp > "$OUT/wd-before.txt"; NW=$(wc -l < "$OUT/wd-before.txt")
[[ $NW == 1 ]] || { log "REFUSE: expected exactly 1 enforce p3 watchdog, found $NW"; exit 2; }
read -r WPID WPPID < "$OUT/wd-before.txt"
grep -q '"trip": null' "$WST" || { log "REFUSE: production watchdog already tripped: $(cut -c1-200 "$WST")"; exit 2; }
L0=$(wc -l < "$WLOG"); log "deployed watchdog pid=$WPID ppid=$WPPID; log baseline $L0 lines"

python3 "$GR/fault_client.py" "$U" "$M" 4 "$OUT/fault-client.json" > "$OUT/fault-client.log" 2>&1 &
CP=$!
sleep 20
docker pause "$SIDE" >/dev/null 2>&1; TP=$(now)
sudo -n python3 - > "$OUT/ctrl.out" 2>&1 <<'PY'
import mmap, os, struct, time
fd = os.open("/dev/shm/vllm_peer_tier3", os.O_RDWR)
m = mmap.mmap(fd, 4096)
struct.pack_into("<Q", m, 384, 1)
m.flush()
print(f"{time.time():.3f}")
PY
TINJ=$(tail -1 "$OUT/ctrl.out"); echo "$TP $TINJ" > "$OUT/fault-tinj.txt"; log "paused at $TP, ctrl=1 at $TINJ"

for i in $(seq 1 240); do tail -n +$((L0+1)) "$WLOG" | grep -q ' TRIP ' && break; sleep 0.25; done
tail -n +$((L0+1)) "$WLOG" > "$OUT/wdp3.new.log"
grep -E ' TRIP |fail-closed' "$OUT/wdp3.new.log" > "$OUT/wdp3.trip.txt"
python3 - "$OUT/fault-tinj.txt" "$OUT/wdp3.new.log" > "$OUT/lat.txt" <<'PY'
import re, sys, time
tp = float(open(sys.argv[1]).read().split()[0])
line = next((l for l in open(sys.argv[2]) if " TRIP " in l), None)
if not line: print("NO_TRIP"); raise SystemExit
ts = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line).group(1)
print(f"{time.mktime(time.strptime(ts, '%Y-%m-%d %H:%M:%S')) - tp:.2f}")
PY
log "TRIP latency $(cat "$OUT/lat.txt") s"
# amendment A4 (2026-10-31, after R1 run 1): judge only after the watchdog has finished its fail-closed sequence and
# exited (it returns 2 after kill server -> unpause -> stop -t 5 / kill sidecar). Run 1 judged 5 s after the trip
# while the watchdog was inside `docker stop -t 5`, and `down` then killed it mid-sequence.
for i in $(seq 1 240); do [[ -z "$(wdp)" ]] && break; sleep 0.25; done
log "watchdog exited: $([[ -z "$(wdp)" ]] && echo yes || echo NO-after-60s)"
tail -n +$((L0+1)) "$WLOG" > "$OUT/wdp3.new.log"; grep -E ' TRIP |fail-closed' "$OUT/wdp3.new.log" > "$OUT/wdp3.trip.txt"
log "watchdog log: $(tr '\n' '|' < "$OUT/wdp3.trip.txt")"
docker ps -a --format '{{.Names}}={{.State}}' | grep -E '^ds41' > "$OUT/docker-ps.txt" 2>&1
for i in $(seq 1 180); do kill -0 $CP 2>/dev/null || break; sleep 1; done
kill $CP 2>/dev/null; wait $CP 2>/dev/null
cat "$OUT/fault-client.log" >&3
docker unpause "$SIDE" >/dev/null 2>&1 || true
for i in $(seq 1 60); do
  MIB=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '/GB300/{print $2}')
  [[ ${MIB:-99999} -lt 4096 ]] && break; sleep 2
done
echo "${MIB:-unknown}" > "$OUT/gb300-mem.txt"
wdp > "$OUT/wd-after.txt"
python3 - "$OUT" "$PAUSE_BAR" "$NOSTREAM_BAR" > "$OUT/verdict.json" <<'PY'
import json, os, sys
out, pb, nb = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rd = lambda f: open(os.path.join(out, f)).read() if os.path.exists(os.path.join(out, f)) else ""
lat = rd("lat.txt").strip(); trip = rd("wdp3.trip.txt"); ps = rd("docker-ps.txt").replace(" ", "")
cl = json.loads(rd("fault-client.json") or "[]"); pause = float(rd("fault-tinj.txt").split()[0])
mib = rd("gb300-mem.txt").strip()
ik, isd = trip.find("fail-closed: docker kill"), trip.find("fail-closed: sidecar")
late = max((r.get("last") or 0) for r in cl) - pause if cl else 99
g = {
 "B1_trip_latency": ["PASS" if lat not in ("", "NO_TRIP") and float(lat) <= pb else "FAIL", f"{lat}s <= {pb}s"],
 "B2_server_killed": ["PASS" if "ds41=exited" in ps else "FAIL", ps.strip().replace("\n", " ")],
 "B3_log_order": ["PASS" if 0 <= ik < isd and "rc=" in trip else "FAIL", "server kill logged before sidecar stop"],
 # B4 (amended pre-run 2026-10-31): the protected property is "no stream claims a normal completion"; a client-side
 # exception on the killed connection is a louder failure, not a pass-breaker. Errors are recorded, not gated.
 # B4 (consult-3): only streams IN FLIGHT at the pause count (>= 1 chunk, last chunk >= t_pause - 2 s); >= 3 of 4 must
 # qualify or the fault was not exercised (FAIL). None of the in-flight streams may carry a finish_reason.
 "B4_no_finish_reason": ["PASS" if len([r for r in cl if r.get("chunks", 0) >= 1 and (r.get("last") or 0) >= pause - 2]) >= 3
                          and all(r.get("finish") is None for r in cl if r.get("chunks", 0) >= 1 and (r.get("last") or 0) >= pause - 2) else "FAIL",
                          json.dumps([{k: r.get(k) for k in ("chunks", "finish", "error")} for r in cl])],
 "B5_no_tokens_after_stall": ["PASS" if late <= nb else "FAIL", f"max_last - t_pause = {late:.3f}s <= {nb}s"],
 "B6_gb300_freed_wd_exited": ["PASS" if mib.isdigit() and int(mib) <= 4096 and rd("wd-after.txt").strip() == "" else "FAIL",
                               f"GB300 {mib} MiB; watchdog after: {rd('wd-after.txt').strip() or 'none'}"],
 "B7_sidecar_stopped": ["PASS" if "ds41-sidecar=exited" in ps else "FAIL", "stopped, not merely paused"],
}
print(json.dumps({"gates": g, "pass": all(v[0] == "PASS" for v in g.values())}, indent=1))
PY
log "verdict: $(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["pass"], {k:v[0] for k,v in d["gates"].items()})' "$OUT/verdict.json")"
grep -q '"pass": true' "$OUT/verdict.json" && { log "FAULT PASS"; exit 0; }
log "FAULT FAIL"; exit 2

#!/usr/bin/env bash
# soak_check.sh — sample the sidecar lane the way the soak does, from any machine with ssh.
# Usage: soak_check.sh <ssh-host> <server-container>
# Exit 0 = healthy, 1 = a trip condition was observed (prints why).
# Checks: watchdog status (trip / timeouts / server up), sidecar container running, /v1/models answers.
set -uo pipefail
HOST=${1:?usage: soak_check.sh <ssh-host> <server-container>}
SRV=${2:?usage: soak_check.sh <ssh-host> <server-container>}
ssh "$HOST" bash -s -- "$SRV" <<'EOS'
set -uo pipefail
SRV=${1:?server container name}
st=$(cat "$HOME/dsv41/m3/logs/sidecar_watchdog.status.json" 2>/dev/null) || { echo "FAIL no watchdog status file (watchdog not running?)"; exit 1; }
python3 - "$st" "$SRV" <<'PY'
import json, subprocess, sys, urllib.request
st = json.loads(sys.argv[1]); srv = sys.argv[2]
bad = []
if st.get("trip"):
    bad.append(f"watchdog trip: {st['trip']}")
w = st.get("words") or []
if len(w) == 7 and w[2] != 0:
    bad.append(f"sidecar timeouts words[2]={w[2]} (cold-expert add dropped => wrong tokens)")
if not st.get("server_up"):
    bad.append("server container not up")
run = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", srv + "-rtx"],
                     capture_output=True, text=True).stdout.strip()
if run != "true":
    bad.append("sidecar container not running")
try:
    urllib.request.urlopen("http://127.0.0.1:30006/v1/models", timeout=5)
except Exception as e:
    bad.append(f"/v1/models not answering: {e}")
print(f"published={w[0] if w else None} completed={w[1] if w else None} timeouts={w[2] if w else None} "
      f"wait_ms={st.get('wait_ms_per_wait')} mode={st.get('mode')}")
print("FAIL " + " | ".join(bad) if bad else "OK")
sys.exit(1 if bad else 0)
PY
EOS

#!/usr/bin/env python3
"""sidecar_watchdog.py — fail-closed guard for the GB300 + RTX PRO 6000 sidecar lane.

Why this exists: the GB300 hook waits for the sidecar by spinning on a shared counter with a spin limit
(VLLM_EXP_PEER_SPINS). If that limit runs out it increments words[2] and SKIPS the cold-expert add — tokens
keep streaming and are silently wrong. Production therefore runs VLLM_EXP_PEER_SPINS=2000000000 (a late or
dead sidecar HANGS the lane instead of corrupting it) and this watchdog turns that hang into a stop within
seconds.

It reads /dev/shm/vllm_peer_tier2 (int64 words: [0] published seq, [1] completed seq, [2] timeouts,
[3] rows sent, [4] tokens seen, [5] GB300 wait ns, [6] waits) every --interval seconds and trips when:
  * words[2] increases (a cold contribution was dropped -> wrong tokens; fatal),
  * the sidecar container is not running while the server is,
  * completed seq is frozen while published is ahead for longer than --stall seconds (sidecar wedged/paused),
  * the shm buffer is unreadable while the server runs.

On trip in --mode enforce: docker kill the server (in-flight streams are cut), docker stop -t 5 the sidecar,
and — only if --rollback is given — wait for the GB300 to free memory, docker start the rollback container and
poll until it answers /v1/models. Without --rollback it FAIL-STOPS: the lane is killed and nothing auto-starts.
That is the production policy on this lane (owner's call): a silent lane is better than a wrong-token lane, and
a human (or an external check) decides when to bring the rollback up.

Fault-tested 2026-10-22: pause the sidecar -> trip in ~3 s, request cut, v23 rollback bound in 260 s.
stdlib only.

Usage:
  python3 sidecar_watchdog.py --server dsv41-vllm-sidecar --once
  python3 sidecar_watchdog.py --server dsv41-vllm-sidecar --mode enforce \
      --status ~/dsv41/m3/logs/sidecar_watchdog.status.json --log ~/dsv41/m3/logs/sidecar_watchdog.log
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request

SHM = "/dev/shm/vllm_peer_tier2"


def words(path):
    with open(path, "rb") as f:
        return struct.unpack("<7q", f.read(56))


def sh(args, timeout=10):
    """Run a command; never hang the loop. Returns (rc, stdout) with rc=None on timeout."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return None, ""


def running(name):
    """True / False, or None if docker did not answer in time."""
    rc, out = sh(["docker", "inspect", "-f", "{{.State.Running}}", name])
    if rc is None:
        return None
    return rc == 0 and out == "true"


def log(msg, fh):
    line = f"{time.strftime('%F %T %Z')} {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def gpu_used_mib(uuid):
    if not uuid:
        return None
    rc, out = sh(["nvidia-smi", f"--id={uuid}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=15)
    try:
        return int(out.split()[0]) if rc == 0 else None
    except (ValueError, IndexError):
        return None


def bound(url, model):
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=5) as r:
            return model in r.read().decode()
    except Exception:
        return False


def fail_closed(a, side, fh):
    rc, _ = sh(["docker", "kill", a.server], timeout=30)
    side_rc, _ = sh(["docker", "stop", "-t", "5", side], timeout=30)
    if side_rc != 0:
        side_rc, _ = sh(["docker", "kill", side], timeout=30)
    log(f"stopped {a.server} {side} (fail closed) kill_rc={rc} sidecar_rc={side_rc}", fh)
    if not a.rollback:
        return
    t0 = time.time()
    while time.time() - t0 < a.gpu_wait:
        used = gpu_used_mib(a.gpu)
        if used is not None and used < 8192:
            break
        time.sleep(2)
    log(f"GB300 memory.used {gpu_used_mib(a.gpu)} MiB after {time.time() - t0:.0f}s", fh)
    rc, _ = sh(["docker", "start", a.rollback], timeout=60)
    log(f"rollback docker start {a.rollback} rc={rc}", fh)
    t0 = time.time()
    while time.time() - t0 < a.bind_wait:
        if bound(a.url, a.model):
            log(f"rollback BOUND: {a.rollback} answers {a.url} after {time.time() - t0:.0f}s", fh)
            return
        time.sleep(a.bind_poll)
    log(f"rollback NOT bound after {a.bind_wait:.0f}s — NEEDS HANDS", fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--sidecar", default=None, help="default: <server>-rtx")
    ap.add_argument("--rollback", default=None, help="container to docker start after a trip (omit = fail-stop)")
    ap.add_argument("--mode", choices=["enforce", "report"], default="enforce")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--stall", type=float, default=2.0)
    ap.add_argument("--shm", default=SHM)
    ap.add_argument("--gpu", default=os.environ.get("GB300_UUID", ""), help="GB300 UUID; used to wait for HBM to free before a rollback")
    ap.add_argument("--gpu-wait", type=float, default=120.0)
    ap.add_argument("--url", default="http://127.0.0.1:30006")
    ap.add_argument("--model", default="dsv41-flash-uva")
    ap.add_argument("--bind-wait", type=float, default=900.0)
    ap.add_argument("--bind-poll", type=float, default=10.0)
    ap.add_argument("--status", default=os.path.expanduser("~/dsv41/m3/logs/sidecar_watchdog.status.json"))
    ap.add_argument("--log", default=os.path.expanduser("~/dsv41/m3/logs/sidecar_watchdog.log"))
    ap.add_argument("--once", action="store_true", help="print one reading and exit (smoke test)")
    a = ap.parse_args()
    side = a.sidecar or a.server + "-rtx"
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    fh = open(a.log, "a")
    base = None
    lag_since = None
    last_done = None
    log(f"start mode={a.mode} server={a.server} sidecar={side} rollback={a.rollback} stall={a.stall}s interval={a.interval}s", fh)
    while True:
        reason = None
        srv_up = running(a.server)
        try:
            w = words(a.shm)
        except Exception as e:
            w = None
            if srv_up:
                reason = f"shm unreadable while server runs: {e}"
        if w:
            if base is None or w[2] < base:  # first read, or the buffer was recreated by a sidecar restart
                base = w[2]
            if w[2] > base:
                reason = f"sidecar TIMEOUTS {base}->{w[2]} (cold contribution dropped => wrong tokens)"
            # stall = work outstanding AND completed not advancing for > --stall s
            if w[0] != w[1] and last_done is not None and w[1] == last_done:
                lag_since = lag_since or time.time()
                if time.time() - lag_since > a.stall:
                    reason = reason or f"sidecar stalled: published {w[0]} completed {w[1]} (not advancing) for >{a.stall}s"
            else:
                lag_since = None
            last_done = w[1]
        if srv_up and running(side) is False:
            reason = reason or f"sidecar container {side} not running while {a.server} serves"
        if srv_up is None:
            log("WARN docker inspect timed out (dockerd slow?)", fh)
        st = {"t": time.time(), "server_up": bool(srv_up), "words": w, "timeouts_base": base,
              "wait_ms_per_wait": (w[5] / w[6] / 1e6) if w and w[6] else None, "mode": a.mode,
              "trip": reason, "stall_s": a.stall, "v": 2}
        try:
            tmp = a.status + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, a.status)
        except Exception:
            pass
        if a.once:
            print(json.dumps(st))
            return 0
        if reason and srv_up:
            log(f"TRIP {reason}", fh)
            if a.mode == "enforce":
                fail_closed(a, side, fh)
                return 2
            base = w[2] if w else base  # report mode: re-arm on the new count
            lag_since = None
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

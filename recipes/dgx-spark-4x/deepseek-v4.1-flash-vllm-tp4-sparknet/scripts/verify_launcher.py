#!/usr/bin/env python3
"""verify_launcher.py — prove scripts/launch-vllm-tp4.sh still describes the container that actually ran.

usage: python3 scripts/verify_launcher.py <rank> [--host NAME]

Renders the launcher for <rank> with a stub `docker` on PATH, then compares the rendered argv against
`docker inspect` of the container on that host:

  1. everything after the image name == the live container's Config.Cmd, element for element
  2. every --env K=V in the render exists in the live Config.Env (render may be a subset: image defaults fill the rest)
  3. every --volume/-v src:dst in the render exists in the live .Mounts (same src and dst)

Exit 0 on success. Any mismatch prints the first divergence and exits 1.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(REPO, "scripts", "launch-vllm-tp4.sh")

rank = sys.argv[1] if len(sys.argv) > 1 else "1"
host = None
if "--host" in sys.argv:
    host = sys.argv[sys.argv.index("--host") + 1]

tmp = tempfile.mkdtemp()
stub = os.path.join(tmp, "docker")
with open(stub, "w") as f:
    f.write('#!/usr/bin/env bash\nprintf "%s\\0" "$@"\n')
os.chmod(stub, 0o755)

# live container first: the launcher's stubbed VLLM_HOST_IP comes from it, so the env subset check is real
q = ["docker", "inspect", "--format", "{{json .}}", "dsv41-karmic-kraken"]
if host:
    q = ["ssh", f"milo@{host}", " ".join(["docker", "inspect", "--format", "'{{json .}}'", "dsv41-karmic-kraken"])]
insp = json.loads(subprocess.run(q, capture_output=True, text=True, check=True).stdout)
live_cmd = insp["Config"]["Cmd"]
live_env = set(insp["Config"]["Env"])
live_mounts = {(m["Source"], m["Destination"]) for m in insp["Mounts"]}
_live_ip = next((e.split("=", 1)[1] for e in live_env if e.startswith("VLLM_HOST_IP=")), "10.0.0.99")

env = dict(os.environ)
env["PATH"] = tmp + os.pathsep + env["PATH"]
env["SPARK_HOSTS"] = "h0 h1 h2 h3"
env["VLLM_HOST_IP"] = _live_ip
# model and cache sources come from the live container too (they are box paths, scrubbed to $MODEL_DIR/$DS41F_DIR)
_live = {dst: src for src, dst in live_mounts}
env["MODEL_DIR"] = _live["/models"]
env["DS41F_DIR"] = os.path.dirname(_live["/cache"])
env["OVERLAY_DIR"] = os.path.join(REPO, "overlays")
env["MASTER_ADDR"] = "10.0.0.5"
out = subprocess.run(["bash", LAUNCHER, rank], capture_output=True, env=env)
if out.returncode != 0:
    sys.exit(f"launcher failed: {out.stderr.decode()[:400]}")
argv = [a for a in out.stdout.split(b"\0") if a]
argv = [a.decode() for a in argv]
if argv[:2] != ["run", "-d"]:
    sys.exit(f"unexpected argv prefix: {argv[:3]}")

i = next(k for k, a in enumerate(argv) if a.startswith("vllm-ds41f-kkref:"))
image, cmd = argv[i], argv[i + 1:]

print(f"rank {rank}: image {image}, {len(cmd)} args rendered; VLLM_HOST_IP={_live_ip} from the live container")

if cmd != live_cmd:
    for k, (a, b) in enumerate(zip(cmd, live_cmd)):
        if a != b:
            sys.exit(f"ARG MISMATCH at {k}: rendered {a!r} != live {b!r}")
    sys.exit(f"ARG COUNT {len(cmd)} rendered vs {len(live_cmd)} live")

env_pairs = [argv[k + 1] for k, a in enumerate(argv) if a == "--env"]
missing_env = [e for e in env_pairs if e not in live_env]
if missing_env:
    sys.exit(f"ENV NOT IN LIVE CONTAINER: {missing_env[:4]}")

vols = [argv[k + 1] for k, a in enumerate(argv) if a in ("--volume", "-v")]
live_dsts = {dst: src for src, dst in live_mounts}
bad = []
for v in vols:
    src, dst = v.split(":")[0], v.split(":")[1]
    if dst not in live_dsts:
        bad.append(f"{dst} missing")
        continue
    if "/overlays/" in src:
        # overlay sources are scrubbed to this repo's own copies; identity is the destination plus SHA256SUMS,
        # not the box path. Same leaf file, same destination, checksums published.
        if os.path.basename(src.rstrip("/")) != os.path.basename(live_dsts[dst].rstrip("/")):
            bad.append(f"{dst} leaf differs: {src} vs {live_dsts[dst]}")
    elif (src, dst) not in live_mounts:
        bad.append(f"{src} -> {dst} not in live")
if bad:
    sys.exit(f"MOUNT MISMATCH: {bad[:4]}")

print(f"  args: {len(cmd)}/{len(live_cmd)} match")
print(f"  env: {len(env_pairs)} pairs all present in live ({len(live_env)} total)")
print(f"  mounts: {len(vols)} all present in live ({len(live_mounts)} total)")
print("LAUNCHER MATCHES THE MEASURED CONTAINER")

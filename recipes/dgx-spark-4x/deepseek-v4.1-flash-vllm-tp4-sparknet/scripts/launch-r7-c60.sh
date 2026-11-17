#!/usr/bin/env bash
# launch-r7-c60.sh — r7 lane driver for 4x DGX Spark (spark-ds41f candidate-60, TP4 over sparknet RoCE).
#
# This is the campaign runner that booted every r7 arm on 2026-11-16 (`lanes/boot-arm.sh`), generalised from
# "pool A/B" to an explicit host list. Same render, same post-process, same hygiene, same rank order — the
# body of the docker command is not rewritten here, it comes out of Owen's own renderer.
#
#   SPARK_HOSTS="spark5 spark6 spark7 spark8" bash scripts/launch-r7-c60.sh up      # rank order, rank 0 = head
#   SPARK_HOSTS=... bash scripts/launch-r7-c60.sh down
#   SPARK_HOSTS=... bash scripts/launch-r7-c60.sh status
#
# Env:
#   ARM           arm to render (default A3-our-shape-yield-short — the promoted arm)
#   CFG           config path inside the checkout (default experiments/2026-11-16-c60-prefill/arms/$ARM.json)
#   DS41F_DIR     spark-ds41f checkout on RENDER_HOST (default ~/spark-ds41f-r7-c60, commit fcc97650)
#   RENDER_HOST   host that holds the checkout (default: the head node)
#   OVERLAY_DIR   directory on every node holding the two overlay files (default ~/r7-overlays)
#   STOCK=1       run stock candidate-60: no overlay mounts at all (this is arm r7-a3b's shape)
#   NAME          container name (default dsv41-r7-lane; the arms ran as dsv41-r7-<arm>)
#   PORT 8002, MIN_AVAIL_GIB 100, BOOT_TIMEOUT 3600, MODEL_DIR, WORK (scratch dir on this workstation)
#
# Prerequisites, all four nodes: image spark-ds41f:candidate-60 (built locally from fcc97650), the checkpoint at
# $MODEL_DIR, an empty/validated $DS41F_DIR/cache on shared-fast local disk, passwordless sudo for drop_caches.
# Prerequisites, RENDER_HOST: the checkout at fcc97650 plus the two files this recipe adds, copied into it —
#   scripts/c60cfg/nodes-switched.laneC.local.json -> $DS41F_DIR/config/
#   scripts/c60cfg/arms/*.json                     -> $DS41F_DIR/experiments/2026-11-16-c60-prefill/arms/
set -u
D=$(cd "$(dirname "$0")" && pwd)
read -r -a H <<<"${SPARK_HOSTS:?set SPARK_HOSTS to 4 ssh hosts, rank order (rank 0 = head)}"
[ "${#H[@]}" -eq 4 ] || { echo "SPARK_HOSTS needs 4 hosts" >&2; exit 2; }

HEAD=${H[0]}
PORT=${PORT:-8002}
NAME=${NAME:-dsv41-r7-lane}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-100}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-3600}
DS41F_DIR=${DS41F_DIR:-~/spark-ds41f-r7-c60}
OVERLAY_DIR=${OVERLAY_DIR:-~/r7-overlays}
MODEL_DIR=${MODEL_DIR:-~/models/DeepSeek-V4.1-Flash-1017}
ARM=${ARM:-A3-our-shape-yield-short}
CFG=${CFG:-experiments/2026-11-16-c60-prefill/arms/$ARM.json}
RENDER_HOST=${RENDER_HOST:-$HEAD}
WORK=${WORK:-$PWD/r7-launch-$NAME}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
S() { $SSH "$1" "$2"; }
mkdir -p "$WORK"

case "${1:-}" in
down)
  for n in "${H[@]}"; do S "$n" "docker stop -t 60 $NAME >/dev/null 2>&1; echo $n stopped"; done
  for n in "${H[@]}"; do S "$n" "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null"; done
  echo "ARM down"; exit 0 ;;
status)
  for n in "${H[@]}"; do
    echo "$n $(S "$n" "docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null || echo absent; awk '/MemAvailable/{printf \"avail %.1f GiB\", \$2/1048576}' /proc/meminfo" | tr '\n' ' ')"
  done
  exit 0 ;;
up) ;;
*) echo "usage: SPARK_HOSTS=\"h0 h1 h2 h3\" $0 up|down|status" >&2; exit 2 ;;
esac

# ---- 1. render on RENDER_HOST, post-process here -------------------------------------------------------------
# Owen's bin/spark renders his own docker run for one node. Two edits, both ours, both visible below:
# the container name (plus --restart=no) and the overlay bind mounts; and one path rewrite because our nodes keep
# the JIT/AOT caches at ~/spark-ds41f/cache rather than inside the checkout.
for i in 0 1 2 3; do
  n=${H[$i]}
  S "$RENDER_HOST" "cd $DS41F_DIR && bin/spark --cluster-config $CFG render $n" > "$WORK/render-$n.raw" 2> "$WORK/render-$n.err" \
    || { echo "ABORT render $n: $(tail -3 "$WORK/render-$n.err")" >&2; exit 3; }
done
python3 - "$WORK" "$NAME" "$OVERLAY_DIR" "${STOCK:-0}" "${H[@]}" <<'PY'
import sys, shlex, re, pathlib
work, name, overlay_dir, stock, *hosts = sys.argv[1:]
mounts = []
if stock != "1":
    mounts = [
        f"{overlay_dir}/so-bitmask/vllm/v1/structured_output/__init__.py:"
        "/opt/spark-ds41f/candidate/vllm/vllm/v1/structured_output/__init__.py:ro",
        f"{overlay_dir}/dsml-c60/vllm/parser/deepseek_v41.py:"
        "/opt/spark-ds41f/candidate/vllm/vllm/parser/deepseek_v41.py:ro",
    ]
for h in hosts:
    src = pathlib.Path(work, f"render-{h}.raw")
    s = src.read_text().strip()
    assert s.startswith("docker run"), s[:80]
    toks = shlex.split(s)
    assert toks.count("--name") == 1, "--name appears more than once"
    toks[toks.index("--name") + 1] = name
    toks.insert(toks.index("--name"), "--restart=no")
    toks = [t.replace("/home/milo/projects/spark-ds41f/cache", "/home/milo/spark-ds41f/cache") for t in toks]
    img = [i for i, t in enumerate(toks) if re.match(r"^spark-ds41f:[\w.-]+$", t)]
    assert len(img) == 1, img
    toks[img[0]:img[0]] = sum((["--volume", m] for m in mounts), [])
    if "--default-chat-template-kwargs" not in toks:
        toks += ["--default-chat-template-kwargs", '{"thinking":false}']
    out = pathlib.Path(work, f"rendered-{h}.sh")
    out.write_text(" ".join(shlex.quote(t) for t in toks) + "\n")
    print(f"{h}: image {toks[img[0] + 2 * len(mounts)]} mounts {len(mounts)}")
PY
[ $? -eq 0 ] || { echo "ABORT post-process" >&2; exit 4; }

IMG=$(grep -o 'spark-ds41f:[A-Za-z0-9._-]*' "$WORK/rendered-${H[0]}.sh" | tail -1)
echo "UP arm=$ARM cfg=$CFG img=$IMG stock=${STOCK:-0}"

# ---- 2. preflight --------------------------------------------------------------------------------------------
for n in "${H[@]}"; do
  S "$n" "docker image inspect $IMG --format '{{.Id}}'" > "$WORK/imgid-$n" 2>&1 \
    || { echo "ABORT $n: image $IMG missing" >&2; exit 5; }
  if S "$n" "docker ps --format '{{.Names}}' | grep -q ."; then
    echo "ABORT $n: containers already running: $(S "$n" "docker ps --format '{{.Names}}' | tr '\n' ' '")" >&2; exit 6
  fi
  S "$n" "docker rm $NAME >/dev/null 2>&1; mkdir -p /home/milo/spark-ds41f/cache; test -d $MODEL_DIR" \
    || { echo "ABORT $n: checkpoint or cache dir missing" >&2; exit 7; }
  if [ "${STOCK:-0}" != "1" ]; then
    for f in so-bitmask/vllm/v1/structured_output/__init__.py dsml-c60/vllm/parser/deepseek_v41.py; do
      S "$n" "test -f $OVERLAY_DIR/$f" || { echo "ABORT $n: missing overlay $OVERLAY_DIR/$f" >&2; exit 8; }
    done
  fi
  scp -q "$WORK/rendered-$n.sh" "$n:/tmp/r7-run-$NAME.sh" || { echo "ABORT scp $n" >&2; exit 9; }
done
ids=$(cat "$WORK"/imgid-* | sort -u | wc -l | tr -d ' ')
[ "$ids" = 1 ] || { echo "ABORT image ids differ: $(cat "$WORK"/imgid-* | tr '\n' ' ')" >&2; exit 10; }
echo "image id on all four nodes: $(cat "$WORK"/imgid-${H[0]})"

# ---- 3. hygiene gate, then launch ranks 3..0 -----------------------------------------------------------------
for n in "${H[@]}"; do
  a=$(S "$n" "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null; awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo")
  echo "$n MemAvailable=$a GiB"
  awk -v a="$a" -v m="$MIN_AVAIL_GIB" 'BEGIN{exit !(a >= m)}' || { echo "ABORT $n: MemAvailable below $MIN_AVAIL_GIB GiB" >&2; exit 11; }
done
T0=$(date +%s)
for r in 3 2 1 0; do
  n=${H[$r]}
  cid=$(S "$n" "bash /tmp/r7-run-$NAME.sh" 2>> "$WORK/launch-$n.err"); rc=$?
  echo "rank $r ($n) run rc=$rc cid=${cid:0:12}"
  [ $rc -eq 0 ] || { echo "ABORT launch $n: $(tail -2 "$WORK/launch-$n.err")" >&2; exit 12; }
  S "$n" "setsid nohup bash -c 'docker logs -f -t $NAME > /tmp/r7-serve-$NAME.log 2>&1' >/dev/null 2>&1 < /dev/null &"
  [ $r -eq 0 ] || sleep 3
done

# ---- 4. readiness ---------------------------------------------------------------------------------------------
echo "waiting for :$PORT (max ${BOOT_TIMEOUT}s)"
while :; do
  el=$(( $(date +%s) - T0 ))
  if curl -s -m 5 "http://$HEAD:$PORT/v1/models" | grep -q deepseek; then echo "READY in ${el}s"; break; fi
  for n in "${H[@]}"; do
    S "$n" "docker ps --format '{{.Names}}' | grep -qx $NAME" \
      || { echo "ABORT: $n died at ${el}s" >&2; S "$n" "docker logs --tail 60 $NAME 2>&1" | sed -E 's/(api-key[ =])[^ ]+/\1X/' | tee -a "$WORK/boot.log"; exit 13; }
  done
  [ $el -lt "$BOOT_TIMEOUT" ] || { echo "ABORT boot timeout" >&2; exit 14; }
  sleep 15
done
S "$HEAD" "docker logs $NAME 2>&1" | sed -E 's/(api-key[ =])[^ ]+/\1X/g' > "$WORK/boot.log"
grep -E "GPU KV cache size|late admission|Maximum concurrency|structural DSML closers" "$WORK/boot.log" | tail -3
for n in "${H[@]}"; do S "$n" "awk '/MemAvailable/{printf \"$n MemAvail_serving=%.1fGiB\n\", \$2/1048576}' /proc/meminfo"; done
echo "lane up as $NAME; boot log $WORK/boot.log"

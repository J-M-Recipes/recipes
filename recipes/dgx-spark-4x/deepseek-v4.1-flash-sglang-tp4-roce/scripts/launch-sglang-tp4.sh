#!/usr/bin/env bash
# launch-sglang-tp4.sh — Lane B: DeepSeek-V4.1-Flash-1017 on 4x DGX Spark, SGLang TP4 over RoCE (knapcio v2.4 + B1a override).
# Wraps knapcio's start-tp4.sh: renders scripts/env.tp4.recipe into $STACK_DIR/.env.tp4 with your hosts, checks the
# hazards that bit us, then execs start-tp4.sh. Run it on the HEAD node (rank 0); start-tp4.sh drives the workers over ssh.
#
# usage: SPARK_HOSTS="head w1 w2 w3" [WORKER_USER=$USER] [STACK_DIR=$HOME/dsv41-4x-spark] \
#        bash scripts/launch-sglang-tp4.sh {doctor|build|share|pack|serve|stop|status|logs|smoke}
#   first time: doctor -> build (BUILD_DOCKERFILE=Dockerfile.canary-roce, on every node) -> share -> pack -> serve
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-serve}"
read -r -a _H <<<"${SPARK_HOSTS:?set SPARK_HOSTS=\"head w1 w2 w3\" (rank order; ssh names of the four nodes)}"
[ "${#_H[@]}" -eq 4 ] || { echo "SPARK_HOSTS needs exactly 4 hosts (got ${#_H[@]})" >&2; exit 2; }
host(){ echo "${_H[$1]}"; }
STACK_DIR="${STACK_DIR:-$HOME/dsv41-4x-spark}"
KNAPCIO_REPO="${KNAPCIO_REPO:-https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4}"
# v2.4 content; our lane dir was a non-git copy consistent with this commit (not git-verified) — see README Software.
KNAPCIO_REF="${KNAPCIO_REF:-27616beca58690ea3120ad687e721c1c7e735412}"
DOCKERFILE_SHA256="dbd1cfbfb3d4f17c9d278a7b74dcb623ca5cf0109ba0a8192788ddfdc88e0c8c"

if [ ! -d "$STACK_DIR" ]; then
  git clone "$KNAPCIO_REPO" "$STACK_DIR" && git -C "$STACK_DIR" checkout --quiet "$KNAPCIO_REF"
fi
have=$(sha256sum "$STACK_DIR/Dockerfile.canary-roce" | cut -d' ' -f1)
[ "$have" = "$DOCKERFILE_SHA256" ] || echo "WARN: Dockerfile.canary-roce sha256 $have != measured $DOCKERFILE_SHA256 (different stack revision)" >&2

if [ "$CMD" != "stop" ] && [ "$CMD" != "status" ] && [ "$CMD" != "logs" ]; then
  sed -e "s|@WORKER_HOSTS@|$(host 1) $(host 2) $(host 3)|" -e "s|@WORKER_USER@|${WORKER_USER:-$USER}|" \
      "$HERE/env.tp4.recipe" > "$STACK_DIR/.env.tp4"
  # Hazard 1: an env without SKIP_PREPARE=1 makes boot.py download its hard-coded OLD revision INTO MODEL_DIR.
  rendered=$(cd "$STACK_DIR" && bash -c 'source .env.tp4; echo "$SKIP_PREPARE|$MODEL_DIR|$HF_REVISION|$EXTRA_CONTAINER_ENV"')
  IFS='|' read -r sp md rev ece <<<"$rendered"
  [ "$sp" = 1 ] || { echo "ABORT: rendered SKIP_PREPARE=$sp (must be 1)" >&2; exit 1; }
  [ "$rev" = 1bae0e1088993b9b82de08f59de5be5a34e42170 ] || { echo "ABORT: HF_REVISION=$rev" >&2; exit 1; }
  case "$ece" in *DSV41_INDEXER_LOGITS_BUDGET_BYTES=1073741824*) ;; *) echo "ABORT: B1a override missing" >&2; exit 1;; esac
  if [ "$CMD" = "serve" ]; then
    n=$(ls "$md"/model-*-of-00048.safetensors 2>/dev/null | wc -l)
    [ "$n" -eq 48 ] || { echo "ABORT: $md has $n/48 shards" >&2; exit 1; }
    # Hygiene (unified memory): page cache is CUDA-visible memory. Gate before every boot.
    for i in 0 1 2 3; do
      ma=$(ssh -o BatchMode=yes "$(host $i)" 'awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo')
      [ "$ma" -ge 110 ] || echo "WARN: $(host $i) MemAvailable ${ma} GiB < 110 — drop_caches + compact_memory first" >&2
    done
  fi
fi
cd "$STACK_DIR"
exec ./start-tp4.sh "$CMD"

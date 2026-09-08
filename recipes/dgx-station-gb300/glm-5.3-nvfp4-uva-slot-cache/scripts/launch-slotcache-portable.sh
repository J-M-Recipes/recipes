#!/usr/bin/env bash
# Portable slot-cache launcher for the recipe tree.
# Not live-tested from this packaged path; it preserves the measured sc13g flags.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "usage: $0 <run-name> <slots> [extra vllm args...]" >&2
  exit 2
fi
RUN="$1"; SLOTS="$2"; shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
RECIPE_DIR="$(cd "$HERE/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/exx/models/GLM-5.3-NVFP4-big}"
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-}"
CACHE_DIR="${CACHE_DIR:-$HOME/vllm-cache}"
API_KEY_FILE="${API_KEY_FILE:-$HOME/.glm_api_key}"
CAPTURE_DIR="${CAPTURE_DIR:-$RECIPE_DIR/capture}"
SLOT_CACHE_PER_LAYER="${SLOT_CACHE_PER_LAYER:-/w/configs/slots-8400.json}"
AT_KEY="${AT_KEY:-slotcache-S$SLOTS}"
SLOT_CACHE_DEFAULT_IMAGE_TAG="vllm-glm53-uva:v0.28.0-2cf0a691"
SLOT_CACHE_PINNED_LOCAL_IMAGE_ID="sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
SLOT_CACHE_IMAGE_EXPLICIT="${IMAGE+x}"
IMAGE="${IMAGE:-$SLOT_CACHE_DEFAULT_IMAGE_TAG}"
# Hash of the exact upstream gpu_model_runner.py source accepted by the patch generator.
SLOT_CACHE_PINNED_SOURCE_SHA="7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa"
SLOT_CACHE_PINNED_SOURCE_RUNNER="${SLOT_CACHE_PINNED_SOURCE_RUNNER:-/Users/jamesmeadlock/.hermes/profiles/milo/work/k2-v3-runtime-source/complete/vllm/v1/worker/gpu_model_runner.py}"
SLOT_CACHE_STAGED_PATCH_GENERATOR="$RECIPE_DIR/scripts/apply_slot_cache_instrumentation_patch.py"
SLOT_CACHE_PATCH_GENERATOR="${SLOT_CACHE_PATCH_GENERATOR:-$SLOT_CACHE_STAGED_PATCH_GENERATOR}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-big-$RUN}"
if [ -z "${COMPILATION_CONFIG:-}" ]; then
  COMPILATION_CONFIG='{"mode":3,"backend":"eager"}'
fi
# Context-profile overrides. Defaults preserve the measured sc13g flags (8 GiB KV / 65k).
# The live-tested 512K daily profile (see results/2026-09-07-ctx512k-live/) is launched with:
#   KV_CACHE_MEMORY=51539607552 MAX_MODEL_LEN=524288 MAX_NUM_SEQS=1 \
#   SLOT_CACHE_PER_LAYER=/w/configs/slots-5792-ctx512k.json bash scripts/launch-slotcache-portable.sh sc13g-mtp-ctx512k 112 ...
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-8589934592}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

_is_sha64() { [[ "$1" =~ ^[0-9a-f]{64}$ ]]; }
_is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
_sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
_path_has_symlink() {
  python3 - "$1" <<'PY'
import os
import sys
from pathlib import Path

p = Path(sys.argv[1]).expanduser()
if not p.is_absolute():
    p = Path.cwd() / p
parts = p.parts
cur = Path(parts[0])
for part in parts[1:]:
    cur = cur / part
    try:
        st = os.lstat(cur)
    except FileNotFoundError:
        # Missing final paths are handled by callers; missing parents are unsafe.
        if cur == p:
            break
        raise SystemExit(2)
    if os.path.islink(cur):
        raise SystemExit(1)
raise SystemExit(0)
PY
}
_require_no_symlink_path() {
  local label="$1" path="$2"
  if ! _path_has_symlink "$path"; then
    echo "$label must not contain symlinks" >&2
    exit 2
  fi
}
# Deterministic manifest hash of the recipe source artifact mounted at /w.
# Rows are sorted "relative-path NUL byte-count NUL file-sha256" entries; runtime
# output trees are excluded so the digest is not self-referential to launch output.
_recipe_manifest_sha256() {
  python3 - "$RECIPE_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

recipe = Path(sys.argv[1])
ignored_dirs = {".git", "__pycache__", "capture", "results"}
rows = []
for path in sorted(recipe.rglob("*")):
    rel_path = path.relative_to(recipe)
    rel = rel_path.as_posix()
    if any(part in ignored_dirs for part in rel_path.parts):
        continue
    if path.is_symlink():
        print(f"recipe artifact input must not be a symlink: {rel}", file=sys.stderr)
        raise SystemExit(2)
    if not path.is_file():
        continue
    if rel.endswith(".pyc"):
        continue
    data = path.read_bytes()
    rows.append(f"{rel}\0{len(data)}\0{hashlib.sha256(data).hexdigest()}")
manifest = "\n".join(rows).encode() + b"\n"
print(hashlib.sha256(manifest).hexdigest())
PY
}

mkdir -p "$CAPTURE_DIR"

instrument_env=()
instrument_mounts=()
if [ "${SLOT_CACHE_QUIESCENT_SNAPSHOTS:-0}" = "1" ]; then
  PATCHED_RUNNER="${SLOT_CACHE_PATCHED_RUNNER:-}"
  PATCHED_RUNNER_SHA256="${SLOT_CACHE_PATCHED_RUNNER_SHA256:-}"
  if [ -z "$PATCHED_RUNNER" ] || [ ! -r "$PATCHED_RUNNER" ] || [ -d "$PATCHED_RUNNER" ]; then
    echo "missing readable SLOT_CACHE_PATCHED_RUNNER" >&2
    exit 2
  fi
  _require_no_symlink_path "SLOT_CACHE_PATCHED_RUNNER" "$PATCHED_RUNNER"
  SLOT_CACHE_SOURCE_RUNNER="${SLOT_CACHE_SOURCE_RUNNER:-$SLOT_CACHE_PINNED_SOURCE_RUNNER}"
  if [ -z "$SLOT_CACHE_SOURCE_RUNNER" ] || [ ! -r "$SLOT_CACHE_SOURCE_RUNNER" ] || [ -d "$SLOT_CACHE_SOURCE_RUNNER" ]; then
    echo "missing readable SLOT_CACHE_SOURCE_RUNNER" >&2
    exit 2
  fi
  _require_no_symlink_path "SLOT_CACHE_SOURCE_RUNNER" "$SLOT_CACHE_SOURCE_RUNNER"
  if [ "$SLOT_CACHE_PATCH_GENERATOR" != "$SLOT_CACHE_STAGED_PATCH_GENERATOR" ]; then
    echo "SLOT_CACHE_PATCH_GENERATOR must be the staged recipe generator" >&2
    exit 2
  fi
  if [ -z "$SLOT_CACHE_PATCH_GENERATOR" ] || [ ! -r "$SLOT_CACHE_PATCH_GENERATOR" ] || [ -d "$SLOT_CACHE_PATCH_GENERATOR" ]; then
    echo "missing readable SLOT_CACHE_PATCH_GENERATOR" >&2
    exit 2
  fi
  _require_no_symlink_path "SLOT_CACHE_PATCH_GENERATOR" "$SLOT_CACHE_PATCH_GENERATOR"
  if ! _is_sha64 "$PATCHED_RUNNER_SHA256"; then
    echo "SLOT_CACHE_PATCHED_RUNNER_SHA256 must be a canonical lowercase SHA-256" >&2
    exit 2
  fi
  actual_runner_sha="$(_sha256_file "$PATCHED_RUNNER")"
  if [ "$actual_runner_sha" != "$PATCHED_RUNNER_SHA256" ]; then
    echo "patched runner sha256 mismatch" >&2
    exit 2
  fi

  if [ "${STATS_SEC:-20}" != "0" ]; then
    echo "STATS_SEC=0 is required for quiescent instrumentation" >&2
    exit 2
  fi
  if [ "${SLOT_CACHE_EXPECTED_LAYERS:-75}" != "75" ]; then
    echo "SLOT_CACHE_EXPECTED_LAYERS=75 is required" >&2
    exit 2
  fi
  SLOT_CACHE_RUN_ID="${SLOT_CACHE_RUN_ID:-}"
  if [[ ! "$SLOT_CACHE_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "SLOT_CACHE_RUN_ID must be explicit and safe" >&2
    exit 2
  fi
  if [ "$SLOT_CACHE_RUN_ID" != "$RUN" ]; then
    echo "SLOT_CACHE_RUN_ID must match launch run name" >&2
    exit 2
  fi
  SLOT_CACHE_SOURCE_SHA="${SLOT_CACHE_SOURCE_SHA:-}"
  SLOT_CACHE_RECIPE_SHA="${SLOT_CACHE_RECIPE_SHA:-}"
  SLOT_CACHE_IMAGE_SHA="${SLOT_CACHE_IMAGE_SHA:-}"
  SLOT_CACHE_ENGINE_GENERATION="${SLOT_CACHE_ENGINE_GENERATION:-}"
  SLOT_CACHE_WINDOW_STEPS="${SLOT_CACHE_WINDOW_STEPS:-}"
  SLOT_CACHE_K_MODE="${SLOT_CACHE_K_MODE:-}"
  SLOT_CACHE_SNAPSHOT_DIR="${SLOT_CACHE_SNAPSHOT_DIR:-}"
  if ! _is_sha64 "$SLOT_CACHE_SOURCE_SHA"; then echo "SLOT_CACHE_SOURCE_SHA must be a canonical lowercase SHA" >&2; exit 2; fi
  if ! _is_sha64 "$SLOT_CACHE_RECIPE_SHA"; then echo "SLOT_CACHE_RECIPE_SHA must be a canonical lowercase SHA" >&2; exit 2; fi
  if [ -z "$SLOT_CACHE_IMAGE_EXPLICIT" ]; then
    IMAGE="$SLOT_CACHE_PINNED_LOCAL_IMAGE_ID"
  fi
  if [[ ! "$SLOT_CACHE_IMAGE_SHA" =~ ^sha256:[0-9a-f]{64}$ ]]; then echo "SLOT_CACHE_IMAGE_SHA must be sha256:<64 lowercase hex>" >&2; exit 2; fi
  if [ "$IMAGE" != "$SLOT_CACHE_PINNED_LOCAL_IMAGE_ID" ]; then
    echo "IMAGE must be the pinned local Docker image ID" >&2
    exit 2
  fi
  if [ "$SLOT_CACHE_IMAGE_SHA" != "$IMAGE" ]; then
    echo "SLOT_CACHE_IMAGE_SHA must match IMAGE local ID" >&2
    exit 2
  fi
  if [ "$SLOT_CACHE_SOURCE_SHA" != "$SLOT_CACHE_PINNED_SOURCE_SHA" ]; then
    echo "SLOT_CACHE_SOURCE_SHA must match pinned gpu_model_runner.py source" >&2
    exit 2
  fi
  actual_source_sha="$(_sha256_file "$SLOT_CACHE_SOURCE_RUNNER")"
  if [ "$actual_source_sha" != "$SLOT_CACHE_PINNED_SOURCE_SHA" ]; then
    echo "SLOT_CACHE_SOURCE_RUNNER bytes must match pinned gpu_model_runner.py source" >&2
    exit 2
  fi
  actual_recipe_sha="$(_recipe_manifest_sha256)"
  if [ "$SLOT_CACHE_RECIPE_SHA" != "$actual_recipe_sha" ]; then
    echo "SLOT_CACHE_RECIPE_SHA must match recipe manifest" >&2
    exit 2
  fi
  generated_tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/slotcache-runner.XXXXXX")"
  cleanup_generated_tmpdir() { rm -rf "$generated_tmpdir"; }
  trap cleanup_generated_tmpdir EXIT
  generated_runner="$generated_tmpdir/gpu_model_runner.py"
  if ! python3 "$SLOT_CACHE_PATCH_GENERATOR" --source "$SLOT_CACHE_SOURCE_RUNNER" --output "$generated_runner" --expected-sha256 "$SLOT_CACHE_PINNED_SOURCE_SHA" >/dev/null; then
    echo "failed to regenerate deterministic patched runner" >&2
    exit 2
  fi
  if ! cmp -s "$PATCHED_RUNNER" "$generated_runner"; then
    echo "patched runner does not match deterministic generator output" >&2
    exit 2
  fi
  if ! _is_posint "$SLOT_CACHE_ENGINE_GENERATION"; then echo "SLOT_CACHE_ENGINE_GENERATION must be positive" >&2; exit 2; fi
  if [[ ! "$SLOT_CACHE_WINDOW_STEPS" =~ ^[1-9][0-9]*:[1-9][0-9]*(,[1-9][0-9]*:[1-9][0-9]*)*$ ]]; then
    echo "SLOT_CACHE_WINDOW_STEPS must be explicit start:end windows" >&2
    exit 2
  fi
  if ! python3 - "$SLOT_CACHE_WINDOW_STEPS" <<'PY'
import sys
for piece in sys.argv[1].split(','):
    start, end = map(int, piece.split(':'))
    if end <= start:
        raise SystemExit(2)
PY
  then
    echo "SLOT_CACHE_WINDOW_STEPS requires end > start" >&2
    exit 2
  fi
  case "$SLOT_CACHE_SNAPSHOT_DIR" in
    /wcap/*) ;;
    *) echo "SLOT_CACHE_SNAPSHOT_DIR must be under /wcap" >&2; exit 2 ;;
  esac
  case "$SLOT_CACHE_SNAPSHOT_DIR" in
    *../*|*/..|*/./*|*/.) echo "SLOT_CACHE_SNAPSHOT_DIR must not contain traversal" >&2; exit 2 ;;
  esac
  snapshot_rel="${SLOT_CACHE_SNAPSHOT_DIR#/wcap/}"
  snapshot_host="$CAPTURE_DIR/$snapshot_rel"
  if [ -L "$snapshot_host" ]; then echo "SLOT_CACHE_SNAPSHOT_DIR must not be a symlink" >&2; exit 2; fi
  if [ ! -d "$snapshot_host" ]; then echo "SLOT_CACHE_SNAPSHOT_DIR host path must already exist" >&2; exit 2; fi
  capture_real="$(cd "$CAPTURE_DIR" && pwd -P)"
  snapshot_real="$(cd "$snapshot_host" && pwd -P)"
  case "$snapshot_real" in
    "$capture_real"/*) ;;
    *) echo "SLOT_CACHE_SNAPSHOT_DIR escaped CAPTURE_DIR" >&2; exit 2 ;;
  esac

  spec_count=0
  spec_json=""
  prev=""
  for arg in "$@"; do
    if [ "$prev" = "--speculative-config" ]; then
      spec_json="$arg"
      prev=""
      continue
    fi
    if [ "$arg" = "--speculative-config" ]; then
      spec_count=$((spec_count + 1))
      prev="$arg"
    fi
  done
  if [ "$spec_count" != "1" ] || [ -z "$spec_json" ]; then
    echo "exactly one --speculative-config is required" >&2
    exit 2
  fi
  spec_k="$(python3 - "$spec_json" <<'PY'
import json, sys
try:
    data = json.loads(sys.argv[1])
except Exception:
    raise SystemExit(2)
if set(data) != {"method", "num_speculative_tokens"} or data.get("method") != "mtp":
    raise SystemExit(2)
k = data.get("num_speculative_tokens")
if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
    raise SystemExit(2)
print(k)
PY
  )" || { echo "--speculative-config must be a single mtp config" >&2; exit 2; }
  if [ "$SLOT_CACHE_K_MODE" != "K$spec_k" ]; then
    echo "SLOT_CACHE_K_MODE does not match speculative config" >&2
    exit 2
  fi
  if [ "${SLOT_CACHE_COUNTER_SCOPE:-target_slot_cache}" != "target_slot_cache" ]; then
    echo "SLOT_CACHE_COUNTER_SCOPE=target_slot_cache is required" >&2
    exit 2
  fi
  if [[ ! "${SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS:-0}" =~ ^[01]$ ]]; then
    echo "SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS must be 0 or 1" >&2
    exit 2
  fi

  instrument_mounts+=( -v "$PATCHED_RUNNER:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" )
  instrument_env+=(
    -e SLOT_CACHE_QUIESCENT_SNAPSHOTS=1
    -e SLOT_CACHE_STATS_MODE=quiescent
    -e SLOT_CACHE_EXPECTED_LAYERS=75
    -e "SLOT_CACHE_SNAPSHOT_DIR=$SLOT_CACHE_SNAPSHOT_DIR"
    -e "SLOT_CACHE_WINDOW_STEPS=$SLOT_CACHE_WINDOW_STEPS"
    -e "SLOT_CACHE_RUN_ID=$SLOT_CACHE_RUN_ID"
    -e "SLOT_CACHE_SOURCE_SHA=$SLOT_CACHE_SOURCE_SHA"
    -e "SLOT_CACHE_IMAGE_SHA=$SLOT_CACHE_IMAGE_SHA"
    -e "SLOT_CACHE_RECIPE_SHA=$SLOT_CACHE_RECIPE_SHA"
    -e "SLOT_CACHE_ENGINE_GENERATION=$SLOT_CACHE_ENGINE_GENERATION"
    -e "SLOT_CACHE_K_MODE=$SLOT_CACHE_K_MODE"
    -e "SLOT_CACHE_COUNTER_SCOPE=${SLOT_CACHE_COUNTER_SCOPE:-target_slot_cache}"
    -e "SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS=${SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS:-0}"
    -e PYTHONPATH=/w/patches
  )
fi

if [ ! -r "$API_KEY_FILE" ]; then
  echo "missing readable API_KEY_FILE: $API_KEY_FILE" >&2
  exit 2
fi
API_KEY="$(tr -d '\r\n' < "$API_KEY_FILE")"
if [ -z "$API_KEY" ]; then
  echo "API_KEY_FILE is empty: $API_KEY_FILE" >&2
  exit 2
fi

vllm_args=(
  /model --host 0.0.0.0 --port 30001
  --served-model-name glm-5.3-big
  --trust-remote-code
  --quantization modelopt
  --load-format safetensors
  --offload-backend uva
  --cpu-offload-gb 420
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight
  --gpu-memory-utilization 0.95
  --kv-cache-dtype bfloat16
  --kv-cache-memory "$KV_CACHE_MEMORY"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens 8192
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
)

has_compilation_config=0
for arg in "$@"; do
  if [ "$arg" = "--compilation-config" ]; then
    has_compilation_config=1
    break
  fi
done
if [ "$has_compilation_config" = "0" ]; then
  vllm_args+=(--compilation-config "$COMPILATION_CONFIG")
fi
vllm_args+=("$@")

docker_mounts=(
  -v "$MODEL_DIR:/model:ro"
  -v "$CACHE_DIR:/root/.cache/vllm"
  -v "$RECIPE_DIR:/w:ro"
  -v "$CAPTURE_DIR:/wcap"
  -v "$RECIPE_DIR/patches/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
  ${instrument_mounts[@]+"${instrument_mounts[@]}"}
)
if [ -n "$DRAFT_MODEL_DIR" ]; then
  if [ ! -d "$DRAFT_MODEL_DIR" ]; then
    echo "missing DRAFT_MODEL_DIR: $DRAFT_MODEL_DIR" >&2
    exit 2
  fi
  docker_mounts+=( -v "$DRAFT_MODEL_DIR:/draft:ro" )
fi

docker_extra=()
profile_env=()
container_cmd=("${vllm_args[@]}")
if [ "${NSYS:-0}" = "1" ]; then
  NSYS_INSTALL_DIR="${NSYS_INSTALL_DIR:-/opt/nvidia/nsight-systems/2025.6.3}"
  NSYS_TARGET_DIR="$NSYS_INSTALL_DIR/target-linux-sbsa-armv8"
  NSYS_OUTPUT="${NSYS_OUTPUT:-/wcap/e1-profile}"
  NSYS_CONTROL="${NSYS_CONTROL:-/wcap/nsys-control}"
  if [ ! -x "$NSYS_TARGET_DIR/nsys" ]; then
    echo "missing executable nsys target: $NSYS_TARGET_DIR/nsys" >&2
    exit 2
  fi
  case "$NSYS_OUTPUT:$NSYS_CONTROL" in
    /wcap/*:/wcap/*) ;;
    *) echo "NSYS_OUTPUT and NSYS_CONTROL must be under /wcap" >&2; exit 2 ;;
  esac
  # Nsight's target binary refuses direct invocation. Preserve the install tree at
  # its absolute path and invoke it through a symlink inside the writable capture mount.
  ln -sfn "$NSYS_TARGET_DIR/nsys" "$CAPTURE_DIR/nsys-cli"
  docker_mounts+=( -v "$NSYS_INSTALL_DIR:$NSYS_INSTALL_DIR:ro" )
  docker_extra+=( --cap-add SYS_ADMIN --security-opt seccomp=unconfined --entrypoint /wcap/nsys-cli )
  profile_env+=( -e "SLOT_CACHE_PROFILE_CONTROL=$NSYS_CONTROL" )
  container_cmd=(
    profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
    --cuda-graph-trace=node --capture-range=cudaProfilerApi
    --capture-range-end=stop --force-overwrite=true -o "$NSYS_OUTPUT"
    /usr/local/bin/vllm serve "${vllm_args[@]}"
  )
fi

DOCKER="${DOCKER:-docker}"
"$DOCKER" run -d --name "$CONTAINER_NAME" --gpus all --shm-size 32g --network host \
  ${docker_extra[@]+"${docker_extra[@]}"} \
  "${docker_mounts[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e "VLLM_AUTOTUNE_CACHE_KEY=$AT_KEY" \
  -e EXACT_PIN=1 \
  -e EXACT_PIN_FILE=/w/patches/exact_pin.py \
  -e "SLOT_CACHE=$SLOTS" \
  -e SLOT_CACHE_HOOK=/w/patches/slot_cache_hook.py \
  -e SLOT_CACHE_CAPTURE="${CAPTURE:-0}" \
  -e SLOT_CACHE_ROUTER="${ROUTER:-ffi}" \
  -e SLOT_CACHE_PER_LAYER="$SLOT_CACHE_PER_LAYER" \
  -e SLOT_CACHE_UNPACKED="${UNPACKED:-0}" \
  -e SLOT_CACHE_LOGIT_RING="${LOGIT_RING:-0}" \
  -e SLOT_CACHE_STATS_SEC="${STATS_SEC:-20}" \
  -e SLOT_CACHE_BYPASS_TOKENS="${BYPASS:-16}" \
  ${instrument_env[@]+"${instrument_env[@]}"} \
  ${profile_env[@]+"${profile_env[@]}"} \
  -e "VLLM_API_KEY=$API_KEY" \
  "$IMAGE" \
  "${container_cmd[@]}"

echo "launched $CONTAINER_NAME SLOT_CACHE=$SLOTS offload=420 per_layer=$SLOT_CACHE_PER_LAYER"

#!/usr/bin/env bash
# Safe rollback helper. Default mode is a verifiable dry-run; pass --execute to act.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RECIPE_DIR="$(cd "$HERE/.." && pwd)"
MODE="--dry-run"
FRESH_LAUNCH=0
PREVIOUS_RUN="${PREVIOUS_RUN:-v1}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$HERE/launch-bigv1.sh}"
HEALTH_CHECK_SCRIPT="${HEALTH_CHECK_SCRIPT:-$HERE/health-check.sh}"
CANDIDATE_CONTAINER="${CANDIDATE_CONTAINER:-${CONTAINER_NAME:-glm53-big-sc13g}}"
PRESERVED_CONTAINER="${PRESERVED_CONTAINER:-glm53-big-v1-keep}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30001}"
MODEL_NAME="${MODEL_NAME:-glm-5.3-big}"
HEALTH_RETRIES="${HEALTH_RETRIES:-720}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"

usage() {
  echo "usage: $0 [--dry-run|--execute] [--fresh-launch]" >&2
}

for arg in "$@"; do
  case "$arg" in
    --dry-run|--execute) MODE="$arg" ;;
    --fresh-launch) FRESH_LAUNCH=1 ;;
    *) usage; exit 2 ;;
  esac
done

cmd_stop=(docker stop "$CANDIDATE_CONTAINER")
cmd_start=(docker start "$PRESERVED_CONTAINER")
cmd_launch=(bash "$LAUNCH_SCRIPT" "$PREVIOUS_RUN")
cmd_health=(bash "$HEALTH_CHECK_SCRIPT")

if [ "$MODE" = "--dry-run" ]; then
  echo "ROLLBACK_DRY_RUN: no Docker actions executed"
  printf 'would run: %q ' "${cmd_stop[@]}"; printf '\n'
  echo "would inspect preserved container: $PRESERVED_CONTAINER"
  printf 'would run if preserved exists: %q ' "${cmd_start[@]}"; printf '\n'
  printf 'would run only with --fresh-launch if preserved is absent: %q ' "${cmd_launch[@]}"; printf '\n'
  printf 'would run health: BASE_URL=%q MODEL_NAME=%q %q ' "$BASE_URL" "$MODEL_NAME" "${cmd_health[@]}"; printf '\n'
  exit 0
fi

command -v docker >/dev/null 2>&1 || { echo "missing required command: docker" >&2; exit 2; }
[ -x "$HEALTH_CHECK_SCRIPT" ] || [ -f "$HEALTH_CHECK_SCRIPT" ] || { echo "missing health check script: $HEALTH_CHECK_SCRIPT" >&2; exit 2; }

if [ "$CANDIDATE_CONTAINER" = "$PRESERVED_CONTAINER" ]; then
  echo "refusing to stop preserved baseline; candidate container must differ from preserved baseline: $CANDIDATE_CONTAINER" >&2
  exit 1
fi

PRESERVED_EXISTS=0
PRESERVED_RUNNING="false"
if running="$(docker inspect -f '{{.State.Running}}' "$PRESERVED_CONTAINER" 2>/dev/null)"; then
  PRESERVED_EXISTS=1
  PRESERVED_RUNNING="$running"
else
  if names="$(docker container ls -a --format '{{.Names}}' 2>/dev/null)"; then
    if printf '%s\n' "$names" | grep -Fx -- "$PRESERVED_CONTAINER" >/dev/null; then
      echo "unable to inspect preserved baseline container: $PRESERVED_CONTAINER; rollback aborted" >&2
      exit 1
    fi
  else
    if [ "$FRESH_LAUNCH" = "1" ]; then
      echo "fresh launch requires confirmed preserved-container absence: $PRESERVED_CONTAINER; docker inspect/list failed; rollback aborted" >&2
    else
      echo "unable to confirm preserved baseline container state: $PRESERVED_CONTAINER; docker inspect/list failed; rollback aborted" >&2
    fi
    exit 1
  fi
fi

if [ "$PRESERVED_EXISTS" = "0" ]; then
  if [ "$FRESH_LAUNCH" != "1" ]; then
    echo "preserved baseline container not found: $PRESERVED_CONTAINER; pass --fresh-launch to run $LAUNCH_SCRIPT $PREVIOUS_RUN" >&2
    exit 1
  fi
  [ -f "$LAUNCH_SCRIPT" ] || { echo "missing launch script: $LAUNCH_SCRIPT" >&2; exit 2; }
fi

printf 'ROLLBACK_EXECUTE: stopping candidate %s (stop only, no rm)\n' "$CANDIDATE_CONTAINER"
if ! docker stop "$CANDIDATE_CONTAINER" >/dev/null 2>&1; then
  echo "failed to stop candidate container: $CANDIDATE_CONTAINER" >&2
  echo "rollback aborted; verify Docker status and set CANDIDATE_CONTAINER explicitly if needed" >&2
  exit 1
fi

if [ "$PRESERVED_EXISTS" = "1" ]; then
  if [ "$PRESERVED_RUNNING" = "true" ]; then
    echo "preserved baseline already running: $PRESERVED_CONTAINER"
  else
    echo "starting preserved baseline: $PRESERVED_CONTAINER"
    docker start "$PRESERVED_CONTAINER" >/dev/null
  fi
else
  echo "preserved baseline absent; fresh-launching via $LAUNCH_SCRIPT $PREVIOUS_RUN"
  bash "$LAUNCH_SCRIPT" "$PREVIOUS_RUN"
fi

BASE_URL="$BASE_URL" MODEL_NAME="$MODEL_NAME" HEALTH_RETRIES="$HEALTH_RETRIES" RETRY_SLEEP="$RETRY_SLEEP" bash "$HEALTH_CHECK_SCRIPT"
echo "ROLLBACK_OK $MODEL_NAME $BASE_URL"

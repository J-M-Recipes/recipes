#!/usr/bin/env bash
# Portable read-only health gate for the GLM-5.3 big vLLM endpoint.
# Performs HTTP reads only: no Docker/container actions.
set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:30001}"
MODEL_NAME="${MODEL_NAME:-glm-5.3-big}"
TIMEOUT="${TIMEOUT:-5}"
HEALTH_RETRIES="${HEALTH_RETRIES:-60}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"
API_KEY_FILE="${API_KEY_FILE:-$HOME/.glm_api_key}"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 2; }
}
need curl
need python3

if [ ! -r "$API_KEY_FILE" ]; then
  echo "missing readable API_KEY_FILE: $API_KEY_FILE" >&2
  exit 2
fi
API_KEY="$(tr -d '\r\n' < "$API_KEY_FILE")"
if [ -z "$API_KEY" ]; then
  echo "API_KEY_FILE is empty: $API_KEY_FILE" >&2
  exit 2
fi

auth_curl() {
  curl -fsS --max-time "$TIMEOUT" -H "Authorization: Bearer $API_KEY" "$@"
}

attempt=1
health_body=""
echo "health: GET $BASE_URL/health (up to $HEALTH_RETRIES attempts)"
while [ "$attempt" -le "$HEALTH_RETRIES" ]; do
  if health_body="$(curl -fsS --max-time "$TIMEOUT" "$BASE_URL/health" 2>/dev/null)"; then
    printf '%s\n' "$health_body"
    break
  fi
  if [ "$attempt" -eq "$HEALTH_RETRIES" ]; then
    echo "health endpoint did not become ready after $HEALTH_RETRIES attempts" >&2
    exit 1
  fi
  sleep "$RETRY_SLEEP"
  attempt=$((attempt + 1))
done

echo "models: GET $BASE_URL/v1/models"
models_json="$(auth_curl "$BASE_URL/v1/models")"
MODELS_JSON="$models_json" MODEL_NAME="$MODEL_NAME" python3 - <<'PY'
import json, os
payload = json.loads(os.environ["MODELS_JSON"])
if not isinstance(payload, dict):
    raise SystemExit("models response is not a JSON object")
items = payload.get("data")
if not isinstance(items, list):
    raise SystemExit("models response missing list field 'data'")
ids = []
for item in items:
    if not isinstance(item, dict) or not isinstance(item.get("id"), str):
        raise SystemExit(f"models response has invalid item: {item!r}")
    ids.append(item["id"])
want = os.environ["MODEL_NAME"]
print("served_models=", ",".join(ids))
if want not in ids:
    raise SystemExit(f"expected model {want!r}; got {ids!r}")
PY

if metrics_text="$(auth_curl "$BASE_URL/metrics" 2>/dev/null)"; then
  echo "metrics: present"
  printf '%s\n' "$metrics_text" | grep -E 'vllm:(num_requests_running|gpu_cache_usage_perc|spec_decode_num_(draft|accepted)_tokens_total)' || true
else
  echo "metrics: unavailable (non-fatal for local health script)"
fi

echo "HEALTH_OK $MODEL_NAME $BASE_URL"

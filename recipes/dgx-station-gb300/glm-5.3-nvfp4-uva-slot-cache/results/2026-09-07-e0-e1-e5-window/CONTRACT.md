# Frozen contract — E0 live telemetry · E1 nsys decomposition · E5 MTP K=2

Status: **FROZEN 2026-09-08T02:20:04-05:00 after safety review; PREPARED, NOT RELEASED.**

This file freezes scope, gates, ordering, and receipt names before any live action. Preparation does
not authorize a launch. The window may begin only after James explicitly says to launch and the
Station-side gate reads exact `CONTROL=RUN` plus exact `RELEASE=e0-e1-e5-20260907-v1`.

## Question

On the proven single-GB300 GLM-5.3 512K/MTP(1) slot-cache lane:

1. What is the real cache-hit rate once speculative verification is included in the denominator?
2. Is at least 2.0 ms per verification step directly attributable to slot-cache bookkeeping,
   scalar gathers, and their CUDA launch API work—enough to justify E4 gather compaction?
3. Does MTP K=2 increase accepted length enough to produce a quality-neutral C1 throughput win?

This is a bounded diagnostic and recipe-selection test. It is not a universal performance or
quality claim.

## Frozen source and incumbent

- Prepared from repository state after published commit `df28147` plus the files covered by this
  package's `PREP-SHA256SUMS`.
- Host: `milo@192.168.1.9`.
- Incumbent container: `glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907`.
- Image: `vllm-glm53-uva:v0.28.0-2cf0a691`.
- Model id: `glm-5.3-big`; maximum model length: `524288`.
- Exact observed command/environment subset: `../../research/window-live-baseline-2026-09-07.json`.
- Slot budget: exactly 5,792 across 75 routed MoE layers, using
  `configs/slots-5792-ctx512k.json`; its hash must equal the incumbent map hash before launch.
- FlashInfer autotune key: `slotcache-S112`.
- Nsight Systems: 2025.6.3.343-256337165561v0, full install tree
  `/opt/nvidia/nsight-systems/2025.6.3`.

## Hard safety rules

1. One candidate container at a time. Never remove the incumbent or any keeper.
2. `:30003` remains dark. No DSF restore or client routing changes are part of this window.
3. E0 runs against the already-live incumbent and causes no restart.
4. Before E1, stop—not remove—the incumbent. Every error path restarts that exact container.
5. Candidate names are exact: `glm53-big-e1-nsys` and `glm53-big-e5-mtp2`.
6. Every candidate uses the exact image, model, 512K context, 48 GiB KV allocation, one sequence,
   420 GiB routed-expert UVA offload, 5,792-slot map, router, compilation config, and MTP config
   shown here. K=2 may change only `num_speculative_tokens` from 1 to 2.
7. No live code edits. Stage the checksum-verified tree before release.
8. The profiler's `SYS_ADMIN` and unconfined seccomp are allowed only on `glm53-big-e1-nsys`.
9. No winner or ceiling claim from process exit alone. Missing, incomplete, distorted, or
   contradictory evidence is INCONCLUSIVE.
10. Final state requires authenticated `/v1/models` plus an exact `WINDOW_RESTORE_OK` completion
    from the restarted incumbent.

## Frozen gates

### E0 — measurement, no pass/fail

Collect these independent workload slices:

- standard prose benchmark, three scored repetitions at C1/C4/C8;
- standard code benchmark, one scored repetition at C1/C4/C8;
- one real Hermes `glm-big` session with at least four sequential tool calls and at least 20 minutes
  of elapsed interaction;
- the fixed four-prompt acceptance probe at 512 output tokens on incumbent K=1 as an E0 telemetry
  slice, not as the E5 comparator.

For each slice, retain raw logs and the exact pre/post speculative metric counters. The unchanged
incumbent hook prints hit rate with a legacy eight-route denominator. Correct it from observed MTP
verification metrics and state the formula. Report aggregate, best, worst, and median-layer values
only—the live hook does not expose a complete per-layer table. Do not invent one.

The E0 acceptance receipt must contain all four complete rows, nonzero verification steps, and no
missing speculative metric. The patched-tree K=1 receipt collected in E1 is the sole E5 comparator.

### E1 — profiler validity and E4 continuation

Before profiling, freeze the matched K=1 comparator from the exact patched candidate tree:

- fixed four-prompt acceptance probe at 512 output tokens;
- standard prose benchmark, three scored repetitions at C1/C4/C8;
- standard code benchmark, one scored repetition at C1/C4/C8.

Then run the same K=1 candidate twice with the fixed acceptance probe at 64 output tokens per prompt:
first while the profiler is armed but not collecting, then inside one START/STOP capture. Require at
least 50 verification steps in the captured probe.

Export and retain:

- `.nsys-rep`;
- `cuda_gpu_kern_sum` CSV;
- `cuda_kern_exec_sum` CSV;
- `cuda_gpu_trace` CSV;
- `cuda_api_trace` CSV;
- `nsys_bucket.py` JSON.

Kernel totals are aggregate GPU work, not serial latency. Report GPU stream overlap and unaccounted
wall time separately. Never force aggregate kernel duration to sum to wall time.

Validity requirements:

- capture includes the engine-core CUDA process and graph nodes, not only Docker/vLLM parent work;
- all seven kernel buckets are present in the JSON, even when zero;
- profiled decode throughput is no more than 20% below the adjacent unprofiled probe;
- captured verification-step denominator is complete;
- no profiler, CUDA, hook, or report-generation error appears in the logs.

Define recoverable E4 work as the sum per verification step of:

1. `fused_bookkeeping` GPU work;
2. `scalar_gather` GPU work (`index_put`/gather family);
3. CUDA launch API time directly associated with those kernels.

`masked_row_copy` is reported but excluded from the E4-recoverable sum because it represents the
necessary weight transfer, not gather-compaction overhead.

- **E1 PASS / authorize a separate E4 contract:** valid recoverable work >=2.0 ms/step.
- **E1 STOP / close this optimization campaign:** valid recoverable work <2.0 ms/step.
- **E1 INCONCLUSIVE:** any validity requirement fails, especially >20% profiler distortion. An
  inconclusive profile does not authorize E4 and does not support a ceiling claim.

### E5 — three serial gates

Use `scripts/dflash2_acceptance_probe.py` unchanged for K=1 and K=2: four frozen prompts, T=0,
`top_p=1`, low reasoning, 512 max output tokens each.

1. **Acceptance:** K=2 weighted accepted length must be at least the patched-tree E1 K=1 weighted
   accepted length +0.15. Otherwise STOP before throughput.
2. **Throughput:** using byte-identical packaged `fixtures/bench_big.py`, three prose repetitions
   and one code repetition in both matched candidate lanes, K=2 C1 prose mean must be >=1.05x the E1
   K=1 C1 prose mean. C4 and C8 are reported but are not promotion gates. Otherwise STOP before
   quality.
3. **Primary quality:** run the frozen 100-task HumanEval/GSM8K fixture twice per task using
   `results/2026-09-06-mtp-quality-audit/public-source/harness.py`, its frozen fixture
   `primary_100_seed20260906.json`, temperature 0, low reasoning, and 4,096 max tokens. Require:
   - 200/200 complete, valid rows;
   - >=190/200 correct repeats overall;
   - >=80/100 correct repeats in each 50-task category;
   - no unresolved transport, parser, sandbox, or protocol failure.

Only all three passes justify retaining K=2 for a later promotion discussion. Any failed gate stops
K=2 and restores the incumbent. This window itself never promotes a candidate to production.

## Exact staged paths

```text
ROOT=/home/milo/e0-e1-e5-window-20260907
REPO=$ROOT/repo
RECIPE=$REPO/recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache
OUT=$ROOT/receipts
CONTROL=$ROOT/CONTROL
RELEASE=$ROOT/RELEASE
RUN_ID=e0-e1-e5-20260907-v1
API_KEY_FILE=/home/milo/.glm_api_key
```

Staging creates `CONTROL` containing `HOLD` and no `RELEASE`. Experiment code must require exact
values before the first request and again before every experiment container stop/start. The exact
restore path is deliberately exempt: once launch is authorized, stopping either candidate and
restarting the preserved incumbent are always-authorized cleanup actions, even if the release files
are later missing, revoked, or corrupt.

The mandatory gate command is:

```bash
python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID"
```

It must print exactly `WINDOW_GATE_OK e0-e1-e5-20260907-v1`. Every experiment
state-changing command is joined to that gate with `&&`; do not run a detached second line after a
failed gate. The unconditional restore exception is defined below.

## Exact common candidate environment

```bash
export MODEL_DIR=/home/exx/models/GLM-5.3-NVFP4-big
export CACHE_DIR=/home/milo/vllm-cache
export API_KEY_FILE=/home/milo/.glm_api_key
export KV_CACHE_MEMORY=51539607552
export MAX_MODEL_LEN=524288
export MAX_NUM_SEQS=1
export SLOT_CACHE_PER_LAYER=/w/configs/slots-5792-ctx512k.json
export AT_KEY=slotcache-S112
export STATS_SEC=0
export COMPILATION_CONFIG='{"mode":3,"backend":"eager"}'
```

### E1 launch delta

```bash
python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID" && \
docker stop glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907

python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID" && \
CAPTURE_DIR="$OUT/e1" CONTAINER_NAME=glm53-big-e1-nsys NSYS=1 \
NSYS_OUTPUT=/wcap/e1-profile NSYS_CONTROL=/wcap/nsys-control \
bash "$RECIPE/scripts/launch-slotcache-portable.sh" e1-nsys 112 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  | tee "$OUT/e1/launch.txt"
```

The E1 launch above is followed by the readiness, matched K=1 comparator, and single capture block
below. Do not issue a separate preview capture. The E5 candidate is launched only by the gated E5
block after the E1 continuation verdict. A tested fake-Docker contract reproduces incumbent K=1
arguments and asserts that K=2 changes only the speculative-token integer.

## Exact execution commands after release

These commands run from the Station shell after exporting the frozen path and candidate environment
blocks above. `API_KEY` exists only in the operator shell environment and is never written to a
receipt.

### Readiness and matched E1 K=1 comparator

```bash
python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID"
export API_KEY="$(tr -d '\r\n' < "$API_KEY_FILE")"
HEALTH_RETRIES=120 RETRY_SLEEP=5 BASE_URL=http://127.0.0.1:30001 \
  MODEL_NAME=glm-5.3-big API_KEY_FILE="$API_KEY_FILE" \
  bash "$RECIPE/scripts/health-check.sh" | tee "$OUT/e1/health.txt"

python3 "$RECIPE/scripts/dflash2_acceptance_probe.py" \
  --max-tokens 512 --out "$OUT/e1/k1-acceptance.json"
for rep in 1 2 3; do
  python3 "$RECIPE/results/2026-09-07-e0-e1-e5-window/fixtures/bench_big.py" \
    | tee "$OUT/e1/bench-prose-rep${rep}.txt"
done
python3 "$RECIPE/results/2026-09-07-e0-e1-e5-window/fixtures/bench_big_code.py" \
  | tee "$OUT/e1/bench-code-rep1.txt"
python3 "$RECIPE/scripts/dflash2_acceptance_probe.py" \
  --max-tokens 64 --out "$OUT/e1/unprofiled-probe.json"
```

### E1 capture, report export, and bucketing

```bash
python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID"
python3 "$RECIPE/scripts/nsys_capture_control.py" \
  "$OUT/e1/nsys-control" START e1-profile-v1 | tee -a "$OUT/e1/nsys-control.log"
/usr/bin/time -f '%e' -o "$OUT/e1/profile-wall-seconds.txt" \
  python3 "$RECIPE/scripts/dflash2_acceptance_probe.py" \
    --max-tokens 64 --out "$OUT/e1/profiled-probe.json"
python3 "$RECIPE/scripts/nsys_capture_control.py" \
  "$OUT/e1/nsys-control" STOP e1-profile-v1 | tee -a "$OUT/e1/nsys-control.log"
python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID" && \
  docker stop -t 120 glm53-big-e1-nsys

docker logs glm53-big-e1-nsys > "$OUT/e1/container.log" 2>&1
NSYS=/opt/nvidia/nsight-systems/2025.6.3/bin/nsys
"$NSYS" stats --force-export=true --force-overwrite=true \
  --report cuda_gpu_kern_sum,cuda_kern_exec_sum,cuda_gpu_trace,cuda_api_trace \
  --format csv --output "$OUT/e1/e1" "$OUT/e1/e1-profile.nsys-rep"

read -r STEPS WALL_MS < <(python3 - "$OUT/e1/profiled-probe.json" \
  "$OUT/e1/profile-wall-seconds.txt" <<'PY'
import json, sys
probe = json.load(open(sys.argv[1]))
steps = int(probe["summary"]["verification_steps"])
wall_ms = float(open(sys.argv[2]).read().strip()) * 1000
print(steps, wall_ms)
PY
)
python3 "$RECIPE/scripts/nsys_bucket.py" "$OUT/e1/e1_cuda_gpu_kern_sum.csv" \
  --steps "$STEPS" --wall-ms "$WALL_MS" --output "$OUT/e1/nsys-buckets.json"
```

The E1 verdict must separately calculate attributable CUDA launch API time from
`e1_cuda_api_trace.csv`. Do not add all CUDA API time to the gate. If kernel/API attribution is
ambiguous, E1 is INCONCLUSIVE.

### E5 gates

E5 starts only after E1 receipts are complete and the E1 candidate is stopped.

```bash
python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID" && \
  CAPTURE_DIR="$OUT/e5" CONTAINER_NAME=glm53-big-e5-mtp2 \
  bash "$RECIPE/scripts/launch-slotcache-portable.sh" e5-mtp2 112 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    | tee "$OUT/e5/launch.txt"

python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID"
HEALTH_RETRIES=120 RETRY_SLEEP=5 BASE_URL=http://127.0.0.1:30001 \
  MODEL_NAME=glm-5.3-big API_KEY_FILE="$API_KEY_FILE" \
  bash "$RECIPE/scripts/health-check.sh" | tee "$OUT/e5/health.txt"
python3 "$RECIPE/scripts/dflash2_acceptance_probe.py" \
  --max-tokens 512 --out "$OUT/e5/k2-acceptance.json"
python3 "$RECIPE/scripts/window_verdict.py" acceptance \
  "$OUT/e1/k1-acceptance.json" "$OUT/e5/k2-acceptance.json" \
  "$OUT/e5/acceptance-verdict.json"
```

If and only if the acceptance gate passes:

```bash
python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID"
for rep in 1 2 3; do
  python3 "$RECIPE/results/2026-09-07-e0-e1-e5-window/fixtures/bench_big.py" \
    | tee "$OUT/e5/bench-prose-rep${rep}.txt"
done
python3 "$RECIPE/results/2026-09-07-e0-e1-e5-window/fixtures/bench_big_code.py" \
  | tee "$OUT/e5/bench-code-rep1.txt"
python3 "$RECIPE/scripts/window_verdict.py" throughput \
  "$OUT/e1" "$OUT/e5" "$OUT/e5/throughput-verdict.json"
```

If and only if the throughput gate passes:

```bash
python3 "$RECIPE/scripts/window_gate.py" "$ROOT" "$RUN_ID"
AUDIT="$RECIPE/results/2026-09-06-mtp-quality-audit"
SANDBOX_IMAGE="$(docker image inspect --format '{{.Id}}' python:3.12-slim)"
printf '%s\n' "$SANDBOX_IMAGE" > "$OUT/e5/sandbox-image-id.txt"
python3 "$AUDIT/public-source/harness.py" selfcheck \
  --fixtures "$AUDIT/primary_100_seed20260906.json" \
  --sandbox-image "$SANDBOX_IMAGE" | tee "$OUT/e5/sandbox-selfcheck.txt"
python3 "$AUDIT/public-source/harness.py" run \
  --base-url http://127.0.0.1:30001 --model glm-5.3-big \
  --key-file "$API_KEY_FILE" --lane e5-mtp2 --out "$OUT/e5/primary-k2.jsonl" \
  --fixtures "$AUDIT/primary_100_seed20260906.json" --repeats 2 --max-tokens 4096 \
  --reasoning-effort low --sandbox-image "$SANDBOX_IMAGE"
python3 "$RECIPE/scripts/window_verdict.py" quality \
  "$OUT/e5/primary-k2.jsonl" "$OUT/e5/primary-summary.json"
```

### Exact restore path

Run on normal completion and every error path. This cleanup path is intentionally independent of
`CONTROL` and `RELEASE`: revocation must prevent more experiment work, never prevent restoration.

```bash
set -euo pipefail
mkdir -p "$OUT/restore"
INCUMBENT=glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907
IMAGE=vllm-glm53-uva:v0.28.0-2cf0a691

for candidate in glm53-big-e1-nsys glm53-big-e5-mtp2; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$candidate" 2>/dev/null || true)" = true ]; then
    docker stop -t 120 "$candidate"
  fi
done

docker start "$INCUMBENT"

HEALTH_RETRIES=120 RETRY_SLEEP=5 BASE_URL=http://127.0.0.1:30001 \
  MODEL_NAME=glm-5.3-big API_KEY_FILE="$API_KEY_FILE" \
  bash "$RECIPE/scripts/health-check.sh" | tee "$OUT/restore/health.txt"

docker ps --filter "name=^/${INCUMBENT}$" > "$OUT/restore/docker-ps.txt"

python3 - "$OUT/restore/service-proof.json" "$INCUMBENT" "$IMAGE" <<'PY'
import json, subprocess, sys

container = json.loads(subprocess.check_output(["docker", "inspect", sys.argv[2]], text=True))[0]
args = container.get("Args") or container.get("Config", {}).get("Cmd") or []
name = container["Name"].lstrip("/")
image = container["Config"]["Image"]
running = bool(container["State"]["Running"])
proof = {
    "container_name": name,
    "image": image,
    "running": running,
    "started_at": container["State"].get("StartedAt"),
    "args": args,
}
open(sys.argv[1], "w").write(json.dumps(proof, indent=2, sort_keys=True) + "\n")
if name != "glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907":
    raise SystemExit(f"restore container mismatch: {name!r}")
if image != sys.argv[3]:
    raise SystemExit(f"restore image mismatch: {image!r}")
if not running:
    raise SystemExit("restore incumbent is not running")

def flag_value(tokens, *names):
    for i, token in enumerate(tokens):
        for name in names:
            if token == name and i + 1 < len(tokens):
                return tokens[i + 1]
            if token.startswith(name + "="):
                return token.split("=", 1)[1]
    raise SystemExit(f"missing required flag: {'/'.join(names)}")

if int(flag_value(args, "--max-model-len", "--max_model_len")) != 524288:
    raise SystemExit("restore max-model-len mismatch")
if int(flag_value(args, "--max-num-seqs", "--max_num_seqs")) != 1:
    raise SystemExit("restore max-num-seqs mismatch")
spec = json.loads(flag_value(args, "--speculative-config", "--speculative_config"))
if spec.get("method") != "mtp" or int(spec.get("num_speculative_tokens")) != 1:
    raise SystemExit(f"restore MTP config mismatch: {spec!r}")
PY

python3 - "$API_KEY_FILE" "$OUT/restore/models.json" <<'PY'
import json, sys, urllib.request

key = open(sys.argv[1]).read().strip()
request = urllib.request.Request(
    "http://127.0.0.1:30001/v1/models",
    headers={"Authorization": f"Bearer {key}"},
)
with urllib.request.urlopen(request, timeout=60) as response:
    raw = response.read()
open(sys.argv[2], "wb").write(raw)
models = json.loads(raw.decode())
ids = [item.get("id") for item in models.get("data", [])]
if "glm-5.3-big" not in ids:
    raise SystemExit(f"restore model id missing: {ids!r}")
PY

python3 - "$API_KEY_FILE" "$OUT/restore/completion.json" <<'PY'
import json, sys, urllib.request

key = open(sys.argv[1]).read().strip()
payload = {
    "model": "glm-5.3-big",
    "messages": [{"role": "user", "content": "Reply with exactly WINDOW_RESTORE_OK"}],
    "temperature": 0,
    "max_tokens": 16,
    "chat_template_kwargs": {"reasoning_effort": "low"},
}
request = urllib.request.Request(
    "http://127.0.0.1:30001/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    raw = response.read()
open(sys.argv[2], "wb").write(raw)
completion = json.loads(raw.decode())
content = completion["choices"][0]["message"]["content"]
if content != "WINDOW_RESTORE_OK":
    raise SystemExit(f"restore completion mismatch: {content!r}")
print(content)
PY
```

## Required receipt names

```text
preflight/
  docker-ps-a.txt  nvidia-smi.txt  disk.txt  incumbent-inspect-sanitized.json
  source-hashes.txt  slot-map-hash-match.txt  nsys-version.txt

e0/
  bench-prose-rep1.txt  bench-prose-rep2.txt  bench-prose-rep3.txt
  bench-code-rep1.txt  stats-log.txt  speculative-metrics-before.txt
  speculative-metrics-after.txt  k1-acceptance.json  hermes-tool-session.txt
  telemetry-summary.json

e1/
  launch.txt  container.log  k1-acceptance.json
  bench-prose-rep1.txt  bench-prose-rep2.txt  bench-prose-rep3.txt
  bench-code-rep1.txt  unprofiled-probe.json  profiled-probe.json
  nsys-control.log  e1-profile.nsys-rep
  e1_cuda_gpu_kern_sum.csv  e1_cuda_kern_exec_sum.csv
  e1_cuda_gpu_trace.csv  e1_cuda_api_trace.csv
  nsys-buckets.json  e1-verdict.json

e5/
  launch.txt  container.log  k2-acceptance.json  acceptance-verdict.json
  bench-prose-rep1.txt  bench-prose-rep2.txt  bench-prose-rep3.txt
  bench-code-rep1.txt  throughput-verdict.json
  sandbox-image-id.txt  sandbox-selfcheck.txt
  primary-k2.jsonl  primary-summary.json  e5-verdict.json

restore/
  health.txt  docker-ps.txt  models.json  completion.json  service-proof.json

README.md  SHA256SUMS  validation.txt
```

Absent downstream receipts are valid only when a prior serial gate stopped that branch; README must
name the exact gate and explain the omission.

## Restore and publication

On normal completion or error: stop the active candidate if any, start the exact incumbent, wait for
readiness, verify model/context plus exact `WINDOW_RESTORE_OK`, then copy and hash every receipt.
Do not alter the production blog during execution. Publish only after independent package review.

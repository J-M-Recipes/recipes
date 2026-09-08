# Contract — E1 v2 offline-repaired runner

Status: **PREPARED, NOT RELEASED.** This package is offline code only; it does not authorize Station mutation.

## Scope

Run only E1 on the single-GB300 GLM-5.3 512K/MTP(1) slot-cache lane:

1. Archive the runtime source snapshot and prove the exact incumbent identity before the first release gate or stop.
2. Stop the exact incumbent only after the v2 release gate passes.
3. Launch a patched-tree K=1 candidate under Nsight Systems from the archived runtime snapshot, not the incumbent `/w` tree.
4. Collect the patched K=1 acceptance/bench comparator and one START/STOP Nsight capture.
5. Stop the candidate.
6. Immediately and unconditionally restore and prove the exact incumbent.
7. Only after restore proof succeeds, run offline profiler analysis and produce the E1 verdict.

E5/K=2 is explicitly out of scope. The v2 runner must never pass a speculative config with `num_speculative_tokens=2` and must never launch `glm53-big-e5-mtp2` except as a best-effort restore stop target.

## Versioned release gate

```text
ROOT=/home/milo/e1-window-20260908-v2
RUN_ID=e0-e1-20260908-v2
CONTROL=$ROOT/CONTROL      # exact content: RUN
RELEASE=$ROOT/RELEASE      # exact content: e0-e1-20260908-v2
```

Every state-changing experiment action is preceded by the archived gate copy, for example:

```bash
python3 "$OUT/executed-source/scripts/window_gate.py" "$ROOT" "e0-e1-20260908-v2"
```

Gate failure must stop before any Docker mutation. Restore is intentionally gate-independent once the incumbent has been stopped.

An independently invocable failsafe restore command exists for a systemd timer and does not require `CONTROL`/`RELEASE`:

```bash
python3 "$RECIPE/scripts/window_e1_v2.py" --restore-only --out "$OUT"
```

## Pinned incumbent restore target

The runner verifies this exact incumbent identity before stopping it and again after restore; tag-only matches are insufficient:

```text
container_name=glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907
container_id=c5f345e092748912bee3774d46f3b58587d5fc1d566d5454f24ca3e0527a28ea
image=vllm-glm53-uva:v0.28.0-2cf0a691
image_digest=sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
served_model=glm-5.3-big
max_model_len=524288
max_num_seqs=1
speculative_config={"method":"mtp","num_speculative_tokens":1}
```

The `/w` incumbent mount observed live at `/home/milo/big-v1-campaign` is not mutated by this runner.

## Controlled launch environment

The E1 launch uses an explicit environment rather than inheriting operator shell state:

```text
MODEL_DIR=/home/exx/models/GLM-5.3-NVFP4-big
CACHE_DIR=/home/milo/vllm-cache
API_KEY_FILE=/home/milo/.glm_api_key
KV_CACHE_MEMORY=51539607552
MAX_MODEL_LEN=524288
MAX_NUM_SEQS=1
SLOT_CACHE_PER_LAYER=/w/configs/slots-5792-ctx512k.json
AT_KEY=slotcache-S112
STATS_SEC=0
COMPILATION_CONFIG={"mode":3,"backend":"eager"}
IMAGE=vllm-glm53-uva:v0.28.0-2cf0a691@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
ROUTER=ffi
CAPTURE=0
UNPACKED=0
LOGIT_RING=0
BYPASS=16
NSYS=1
NSYS_OUTPUT=/wcap/e1-profile
NSYS_CONTROL=/wcap/nsys-control
```

`DRAFT_MODEL_DIR` is scrubbed. The candidate command appends exactly:

```text
--speculative-config {"method":"mtp","num_speculative_tokens":1}
```

## Credential posture

The candidate launcher passes `VLLM_API_KEY` through the Docker environment, matching the incumbent launch posture. This is an accepted risk for v2: on the Station, Docker inspect access is restricted to root/docker-group operators, and avoiding this env path would redesign candidate auth differently from the incumbent. The runner must still redact Docker inspect stdout and any logged `VLLM_API_KEY=...`/`API_KEY=...`/bearer-token output in receipts.

## Readiness and safety budget

Full cold load is allowed to exceed the v1 10-minute health window. v2 keeps the general `--command-timeout-sec` subprocess bound at 1800s for non-readiness work and uses a separate runner-enforced `--readiness-timeout-sec` hard wall-clock timeout, default `2400`, for the health/readiness command. The health script still performs bounded HTTP attempts with `HEALTH_RETRIES=360` and `RETRY_SLEEP=5`; the runner timeout is the authoritative outer startup deadline.

James approved downtime until further notice, but the runner still has a bounded independent startup deadline.

## Evidence and release gates

Required E1 evidence:

- exact executed-source archive and `source-manifest.json`; all commands run from this archived snapshot and its patch/config/fixture/helper hashes are bound to the run;
- raw command logs under `raw-logs/`;
- patched K=1 `k1-acceptance.json` and benchmark receipts;
- adjacent unprofiled/profiled 64-token probes;
- a real `.nsys-rep` created by the launched Nsight path, never fabricated by the runner, plus exported CUDA kernel/API CSVs;
- `nsys-buckets.json`;
- explicit hash-bound `api-attribution.json` covering all profiler CSVs, `profiled-probe.json`, `profile-wall-seconds.txt`, and `nsys-buckets.json` before any PASS/STOP profiler claim;
- explicit engine-core/graph-node capture evidence before any PASS/STOP profiler claim;
- pinned preflight and restore proof.

State-changing experiment work uses a shared host operation lock (`/tmp/glm53-e1-v2-host-operation.lock`) in addition to per-output locking. Failsafe restore is not blocked by this lock.

`window_verdict.py e1` fails closed to `INCONCLUSIVE` if profiler evidence, bucket coverage, verification steps, slowdown, API attribution, or engine-core/graph-node evidence is incomplete. Missing API attribution must not be interpreted as zero API overhead. Missing buckets and non-finite numeric fields must produce issues, not uncaught exceptions.

## Archive exclusions

The source manifest includes exact executed sources: runner scripts, health helper, config, patches, and benchmark fixtures. It excludes `__pycache__`, `*.pyc`, and runtime Nsight symlinks so preparation/validation remains feasible and does not ship the 79 MB helper tree.

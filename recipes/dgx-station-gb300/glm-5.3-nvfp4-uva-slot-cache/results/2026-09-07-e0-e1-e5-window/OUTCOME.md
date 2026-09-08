# Outcome — 2026-09-08 live window

Final state: **E0 complete; E1 inconclusive; E5 blocked/not run; exact incumbent restored.**

## E0 — complete telemetry window

E0 ran against the unchanged incumbent and produced benchmark, acceptance, and slot-cache telemetry receipts.

- Corrected MTP(1) slot-cache hit-rate estimate: **0.74762** across **37** `SLOT_CACHE STATS` windows.
- Legacy reconstructed hit-rate estimate: **0.49524**.
- Weighted K=1 accepted length: **1.82694** over **2,048** completion tokens and **1,121** verification steps.
- Prose C=1 throughput across three repetitions: **45.75–46.33 tok/s**.
- Code C=1 throughput: **42.95 tok/s**.

Caveat: the 0.74762 value is a reconstructed estimate from legacy aggregate `misses/step/layer` lines rounded to two decimals, with the corrected 16-route MTP(1) assumption. The receipts retain aggregate before/after speculative counters and aggregate/best/worst/median slot-cache windows; they do **not** retain direct patched actual-route telemetry or enough per-slice/per-layer counters to close every telemetry acceptance criterion.

E0 remains telemetry-only and is not an E5 comparator.

## E1 — inconclusive before readiness

The exact patched-tree K=1 candidate was launched under Nsight Systems, but it exited before readiness. The one-off operator redundantly exported `COMPILATION_CONFIG`; the frozen launcher's nested default expansion appended a trailing `}`, and vLLM rejected the resulting JSON:

```text
--compilation-config {"mode":3,"backend":"eager"}}
```

No matched patched-tree K=1 comparator, Nsight capture, profile, kernel/API attribution, or E4 recoverable opportunity measurement was collected.

The failure also exposed a frozen operational mismatch: `HEALTH_RETRIES=120` with a five-second interval permits 600 seconds, while the observed full-model cold-load path remained in checkpoint loading beyond 600 seconds. The v1 contract was not silently revised.

Consequences under the frozen contract:

- E1 verdict: **INCONCLUSIVE**.
- Recoverable E4 opportunity: **not measured**.
- E5 authorization: `false` / not authorized.
- E5: **not run**.

## E5 — blocked, future work

E5 never ran. Besides the E1 inconclusive blocker, a late independent review identified an environment-parity blocker for any future retry: inherited `NSYS`, `IMAGE`, `ROUTER`, `CAPTURE`, `UNPACKED`, `LOGIT_RING`, `BYPASS`, and `DRAFT_MODEL_DIR` values could violate the intended K-only parity. A retry must scrub or explicitly pin that environment before any E5 release.

## Restore and release closure

The unconditional restore path restarted and proved the exact incumbent:

- Container: `glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907`
- Image: `vllm-glm53-uva:v0.28.0-2cf0a691`
- Served model: `glm-5.3-big`
- Maximum model length: `524288`
- Maximum sequence count: `1`
- Speculative configuration: MTP with `num_speculative_tokens=1`
- Authenticated completion: `WINDOW_RESTORE_OK`

`restore.log` includes an initial connection-refused attempt during restart/cold load; later restore receipts prove the served model and authenticated completion.

The exact release gate was then closed (`CONTROL=HOLD`, `RELEASE` removed), and the fail-safe timer was stopped after restore proof and local receipt replication.

## Evidence and publication caveats

- `live-receipts/SHA256SUMS` verifies the copied live receipt tree.
- `PREP-SHA256SUMS` verifies the frozen prep package.
- The receipt bundle includes Python bytecode files (`*.pyc`) because the frozen manifests hash them; they are retained to preserve manifest verification and are not required operator source.
- `live-receipts/operator-sha256.txt` records hashes for operator scripts on the Station at `/home/milo/e0-e1-e5-window-20260907/`; those source files are not included in this local publication bundle, so this repo cannot independently re-hash the Station-side operator sources.
- `live-receipts/e0/hermes-tool-session.txt` is a model-written final report, not a raw tool transcript. It supports the reported 20-minute Hermes wait only as a secondary summary unless the raw Hermes transcript is separately retrieved.
- No Nsight or kernel-attribution receipt is claimed.

## Follow-up boundary

A retry requires a separately reviewed/released contract revision that fixes the launcher expansion or avoids the redundant environment override, provides a realistic full-model readiness budget, and pins/scrubs the E5 parity environment. This v1 receipt set must not be reinterpreted as a profiler result or completed E5 screen.

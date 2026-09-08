# Publication notes — E0/E1/E5 live window

These notes document the post-release interpretation of the frozen v1 package and live receipts. They intentionally do not change `CONTRACT.md`, `PREP-SHA256SUMS`, or `live-receipts/SHA256SUMS`.

## Final state

- **E0:** complete telemetry/benchmark window.
- **E1:** **INCONCLUSIVE** before readiness.
- **E5:** blocked, not authorized, and not run.
- **Restore:** exact incumbent restored and proved with authenticated `WINDOW_RESTORE_OK` completion.

## Receipt-derived numbers

From `live-receipts/e0/telemetry-summary.json` and associated raw files:

- Corrected MTP(1) slot-cache hit-rate estimate: **0.7476215376521442** across **37** `SLOT_CACHE STATS` windows.
- Legacy reconstructed hit-rate estimate: **0.49524307530428846**.
- Corrected route assumption: **16** routes per step for MTP(1), replacing the legacy printed 8-route `HIT=` denominator.
- Layer steps: **17,582**.
- Estimated misses: **70,997.09**.
- Weighted K=1 acceptance length: **1.8269402319357717**.
- Median K=1 acceptance length: **1.8385265840379938**.
- Acceptance sample: **4** requests, **2,048** completion tokens, **1,121** verification steps.
- Prose C=1 throughput: **45.75–46.33 tok/s** across three repetitions.
- Code C=1 throughput: **42.95 tok/s**.

Telemetry caveat: the 0.74762 value is a reconstructed estimate from legacy aggregate `misses/step/layer` lines rounded to two decimals, with the corrected 16-route MTP(1) assumption. The receipts retain aggregate before/after speculative counters and aggregate/best/worst/median slot-cache windows; they do **not** retain direct patched actual-route telemetry or sufficient per-slice/per-layer counters to close every telemetry acceptance criterion.

## E1 failure and E5 blocker

`live-receipts/e1/failed-candidate-proof.json` shows the candidate exited with code **2** and `oom_killed=false`. The failed command contains:

```text
--compilation-config {"mode":3,"backend":"eager"}}
```

The trailing brace came from a redundant `COMPILATION_CONFIG` export interacting with the frozen launcher expansion. Because E1 failed before readiness, the bundle contains no matched patched-tree K=1 comparator, no Nsight capture, no profile, no kernel/API attribution, and no E4 recoverable-opportunity measurement.

E5 was not authorized and did not run. A late independent review also identified a future-work parity blocker: inherited `NSYS`, `IMAGE`, `ROUTER`, `CAPTURE`, `UNPACKED`, `LOGIT_RING`, `BYPASS`, and `DRAFT_MODEL_DIR` values could violate K-only parity unless scrubbed or explicitly pinned before any future E5 retry.

## Restore proof

Restore receipts identify the restored incumbent as:

- Container: `glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907`
- Image: `vllm-glm53-uva:v0.28.0-2cf0a691`
- Served model: `glm-5.3-big`
- Maximum model length: `524288`
- Maximum sequence count: `1`
- Speculative config: `{"method":"mtp","num_speculative_tokens":1}`
- Authenticated completion: `WINDOW_RESTORE_OK`

`restore.log` includes an initial connection-refused attempt during restart/cold load; later restore receipts prove the served model and authenticated completion.

## Evidence limitations and audit notes

- `live-receipts/SHA256SUMS` and `PREP-SHA256SUMS` verify the copied live receipt tree and frozen prep package.
- The bundle includes Python bytecode files (`*.pyc`) because the frozen manifests hash them; they are retained to preserve manifest verification and are not required operator source.
- `live-receipts/operator-sha256.txt` records hashes for Station-side scripts under `/home/milo/e0-e1-e5-window-20260907/`, but those source files are not included in this local publication bundle for independent rehash.
- `live-receipts/e0/hermes-tool-session.txt` is a model-written final report, not a raw Hermes transcript. Treat it as secondary evidence for the reported 20-minute Hermes wait unless the raw transcript is separately retrieved.
- A local credential-pattern scan of text receipts found no private keys, GitHub/OpenAI/Anthropic-style tokens, AWS access keys, bearer tokens, or email addresses. The package does expose private LAN/host/path metadata such as `192.168.1.9`, `127.0.0.1`, and Station paths.

## Follow-up boundary

A retry requires a separately reviewed/released contract revision that fixes the launcher expansion or avoids the redundant environment override, provides a realistic full-model readiness budget, and pins/scrubs the E5 parity environment. This v1 receipt set must not be reinterpreted as a profiler result, completed E4 measurement, or completed E5 screen.

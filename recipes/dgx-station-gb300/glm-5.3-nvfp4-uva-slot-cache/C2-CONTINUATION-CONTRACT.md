# C2 continuation release contract — K=2 measurement only

Status: **local/offline implementation contract; not a Station authorization and not a promotion decision.**

This continuation runner exists only after the already-completed K1 canary and C1 quality gate have been copied back and verified by exact SHA-256. It must not rerun C1, mutate frozen historical receipts, or infer live authorization from `c2_authorized=false` in the C1 receipt; that field is treated as expected preparation semantics.

## Bound prerequisite evidence

The runner refuses to proceed unless these artifacts match byte-for-byte:

- K1 canary evidence: `/tmp/k1-live-aa3c645/k1-canary-evidence.json` — `43945bb91fa3cd0e1176d0f7e4399d4b8891e901ccebc80bdb2f4c52a0912be7`
- K1 snapshot JSONL: `/tmp/k1-live-aa3c645/slot-cache-snapshots-412-f3b5fe513350.jsonl` — `26419c77a278e86e6e5aafeb2289209573be45872965bf13222dab62da80f5bf`
- C1 quality receipt: `/tmp/c1-live-76fff65/c1-quality-receipt.json` — `38a650666c747d36bd40f91d77a8d73a00bb3f4330a0b43d00fe4fa0a34bb0b8`
- C1 gate: `/tmp/c1-live-76fff65/c1-gate.json` — `0bf616eb2a7962ce54135065233d899c25fa6844fda8839aeb1ded52a1ca0994`
- C1 acceptance: `/tmp/c1-live-76fff65/c1/acceptance-512.json` — `64271ced8af2e96d3ffcb5392794fdb6157f291233bd522ce403679d01c6b7ae`
- C1 incumbent/candidate greedy references: `c8172e6f286d6394aca925ca969dc656fb38f724addd42425f6871036c0116d5`

## Runtime invariants

- Fresh empty output directory and host operation lock are mandatory.
- Every executed source/helper is archived before launch and helper binding is rewritten to archived source.
- C2 uses the exact pinned archived `sources/vllm/v1/worker/gpu_model_runner.py` source runner (`7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa`), patch generator (`8b4b3ae177618875378154681a43c16bf4cc265c6f073fb1dd6ef2562c45106b`), instrumentation adapter (`9f0c75b25438c63511a5b2580a4c0a77520f232e2affe109dd0ba3908477e453`), and launcher (`aebe4fab6272a8ded9d2e871d5b9c536b641634ae9b10232db9fa5c33bcac04d`). The generated patched runner must be exactly `2268a6dafda69566d4128bb9b589bdecb22e3e7eb8d0b7e1155f2bb1ce8e3cd4` and must be supplied to launch via `SLOT_CACHE_PATCHED_RUNNER`, `SLOT_CACHE_PATCHED_RUNNER_SHA256`, `SLOT_CACHE_SOURCE_RUNNER`, `SLOT_CACHE_SOURCE_SHA`, `SLOT_CACHE_RECIPE_SHA`, and generation/counter-scope env.
- Candidate name must be exactly absent before timer arming.
- Incumbent proof must match exact container ID, image digest, model, ctx `524288`, `max-num-seqs=1`, and MTP K=1.
- Candidate image must resolve to the exact digest and actual container `.Image` must exactly equal that digest; `Config.Image` must be one of the exact bare tag, exact digest, or exact `tag@digest` and all other tags, digests, ambiguous `tag@wrong-digest`, and non-string values fail closed.
- Independent system-scope restore-only timer must be armed and read back before incumbent stop; restore obligation is recorded before stop.
- Only C2 is launched, with exact `{"method":"mtp","num_speculative_tokens":2}`. Restore/incumbent remain K=1; `STATS_SEC=0`.
- Candidate must pass authenticated readiness, one real profiled 192-token verification workload between `START c2-continuation` and `STOP c2-continuation` that reaches at least engine step 164, exact 512-token acceptance outside the profiler window, exact 20/20 greedy against the bound C1 incumbent reference, real nonempty Nsight report export, and exact engine-owned start/end snapshots with canonical layers `3..77`.
- Launch environment is fail-closed against the real portable launcher: `IMAGE` and `SLOT_CACHE_IMAGE_SHA` must both equal the pinned local Docker image ID `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`, and the host `CAPTURE_DIR/snapshots/c2-continuation` directory must exist before launch.
- Snapshot acceptance is exact fail-closed: endpoints `start/end`, steps `100/164`, provenance sequences `[2,4]`, `run_id=c2-continuation`, generation `1`, K2, phase `spec_verify`, trace IDs must exactly match the archived instrumentation helper's finalized format `slotcache:c2-continuation:gen:1:seq:{seq}:step:{step}:boundary:step_complete:phase:spec_verify:flags:spec_verify,drafter:graph:unknown:k:K2`, canonical layer keys `3..77`, and nonnegative monotonic per-layer `misses/routes/steps` deltas.
- Candidate is stopped and proven inactive before incumbent restore. Incumbent is restored and authenticated before profiler export/analysis.
- Timer cancellation is allowed only after restore proof; failed restore leaves the timer armed.

## Verdict semantics

The receipt is measurement-only: `promotion_authorized=false`. G2 is finite positive C2 speed `>= 1.05 * 45.747`; NaN, Infinity, missing, or nonpositive speed is invalid and fails closed. G3 is exact greedy 20/20. Nsight CUDA/NVTX/GPU correlation remains `UNPROVEN` unless a future exact dynamic trace-ID correlator proves GPU linkage; missing/weak telemetry is fail-closed to `INCONCLUSIVE`.

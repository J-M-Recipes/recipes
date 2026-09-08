# Runtime canary acceptance for slot-cache quiescent instrumentation

Local CPU tests and source-anchor checks are not proven by local source-level tests to be GPU-safe. They only prove the adapter is opt-in, hash-guarded, compiles as a patched copy, preserves the intended call ordering in the pinned source, and regenerates the runtime copy deterministically from the staged generator.

Supported local instrumentation envelope: opt-in only, pinned `gpu_model_runner.py` SHA `7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa`, generated patched runtime SHA `2268a6dafda69566d4128bb9b589bdecb22e3e7eb8d0b7e1155f2bb1ce8e3cd4`, adapter SHA `9f0c75b25438c63511a5b2580a4c0a77520f232e2affe109dd0ba3908477e453`, staged generator SHA `8b4b3ae177618875378154681a43c16bf4cc265c6f073fb1dd6ef2562c45106b`, launcher SHA `aebe4fab6272a8ded9d2e871d5b9c536b641634ae9b10232db9fa5c33bcac04d`, pinned local Docker image ID `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` (local image ID, not a registry manifest digest), and single-GPU runtime topology only (`pipeline_parallel_size=1`, `tensor_parallel_size=1`, `data_parallel_size=1`, `decode_context_parallel_size=1`, and `use_ubatching=false`). When `SLOT_CACHE_QUIESCENT_SNAPSHOTS=1`, generated runtime init fail-closes outside that envelope before serving/target-forward launch, launcher default `IMAGE` becomes the pinned local image ID, and explicit tag/digest-tag/mismatched `IMAGE` values are rejected. Disabled/default runtime remains uninstrumented for topology purposes and the launcher defaults to image tag `vllm-glm53-uva:v0.28.0-2cf0a691` while preserving explicit `IMAGE` overrides.

Review invalidation scope: only the portable launcher hash changed from the prior offline-reviewed envelope; instrumentation adapter, staged generator, generated patched runner, slot-cache hook, pinned source, and local image identity hashes are preserved.

Launcher acceptance requires the staged recipe generator binding exactly: `SLOT_CACHE_PATCH_GENERATOR`, if set, must equal `scripts/apply_slot_cache_instrumentation_patch.py` in the mounted recipe tree. The launcher recomputes the current recipe source-artifact SHA from sorted `relative-path NUL byte-count NUL file-sha256` rows, excluding `.git`, `__pycache__`, `capture`, `results`, and `.pyc`, and requires it to match `SLOT_CACHE_RECIPE_SHA`. It then regenerates a temporary patched runner from the pinned source and byte-compares it with `SLOT_CACHE_PATCHED_RUNNER`. Required launch metadata also includes `SLOT_CACHE_PATCHED_RUNNER_SHA256`, `SLOT_CACHE_SOURCE_SHA`, `SLOT_CACHE_IMAGE_SHA`, `SLOT_CACHE_RUN_ID`, `SLOT_CACHE_ENGINE_GENERATION`, `SLOT_CACHE_WINDOW_STEPS`, `SLOT_CACHE_K_MODE`, `SLOT_CACHE_SNAPSHOT_DIR`, `STATS_SEC=0`, `SLOT_CACHE_EXPECTED_LAYERS=75`, and exactly one MTP `--speculative-config` matching `SLOT_CACHE_K_MODE`.

Excluded paths: pipeline parallel PP/non-last hidden-state returns, tensor/data/decode-context parallel collectives, and ubatching/DBO are not campaign-supported for this Station recipe; they require separate design and canary proof before being used for slot-cache claims.

Before campaign-valid K1/K2 numbers, Station canaries must prove:

1. **Build identity**: incumbent image SHA, source archive SHA, patch file SHAs, recipe SHA, and `SLOT_CACHE_QUIESCENT_SNAPSHOTS=1` are logged for the run.
2. **Warmup/capture exclusion**: first start snapshot occurs only after all75 registry readiness; serving rows are not `warmup` or `capture`.
3. **All75 readiness**: first campaign-valid row has exactly layers `0..74`, aligned `steps`, and `expected_layers=75`; any diagnostic short registry row is excluded.
4. **Current/default stream ownership**: endpoint device snapshots are enqueued from the engine-owned current/default stream, and async copies wait on that stream before D2H.
5. **no per-step sync**: there is no per-step `.item()`, `.cpu()`, `.tolist()`, `torch.cuda.synchronize()`, or background polling of live counter buffers. Endpoint synchronous D2H is acceptable only outside the measured window and must be reported as endpoint overhead.
6. **Private snapshot race safety**: async scheduling/high-concurrency windows show monotonic counters, no negative deltas, and no torn/repeated layer rows.
7. **Nsight/profiler correlation**: exact `metadata.trace_id` NVTX ranges are found in parsed profiler/Nsight events and bracket target/draft launch or graph replay work; host range containment alone is not acceptable.
8. **GPU timestamps must come from profiler traces**: JSONL snapshot rows must not invent host `start_ns`/`end_ns` as GPU timestamps.
9. **target/draft contamination**: compare drafter-disabled and K2 decode windows to prove whether draft forwards touch the same registry. If contaminated, rows must be marked `target_plus_draft_contaminated` and excluded from target-only hit-rate claims.
10. **abort/stale path**: abort an in-flight request during a window and verify analysis treats counters as scheduled device work, not successful generated tokens.

In this unqualified local build, generated raw snapshot metadata always carries `valid_for_campaign=false` with blocker `external_canary_not_proven`; a free-text `SLOT_CACHE_COUNTER_SCOPE=target_slot_cache` string is not proof of target-only scope. No historical result manifest was regenerated for this offline documentation reconciliation.

Remaining canary blockers: stream/capture behavior, graph replay correlation, async D2H race safety, and target-vs-draft counter scope are unproven until the Station canaries above pass.

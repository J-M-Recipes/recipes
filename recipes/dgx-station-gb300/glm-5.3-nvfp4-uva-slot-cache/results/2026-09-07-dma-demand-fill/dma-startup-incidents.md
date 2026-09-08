# Triton-graphs startup attempt notes

Date: 2026-09-07

## Attempt 1

The experiment launcher used a fresh FlashInfer autotune cache key. That unnecessarily triggered a full 21-profile retune, causing the readiness monitor to time out. The attempt was stopped and retained as `triton-graphs-first-start-fresh-autotune.log`. All experiment arms were corrected to use the previously validated `slotcache-S112` cache key.

## Attempt 2

The cached launch loaded the model and reached CUDA-graph capture, then failed with `cudaErrorStreamCaptureInvalidated`. The preceding root event was the slot-cache statistics thread attempting a CUDA operation while capture was active:

`SLOT_CACHE stats error AcceleratorError("CUDA error: operation not permitted when stream is capturing")`

This is telemetry interference, not a Triton-copy correctness result. For all three matched experiment arms, `SLOT_CACHE_STATS_SEC` is therefore fixed to `0`. The copy backend, slot geometry, model, MTP setting and benchmark workload remain unchanged.

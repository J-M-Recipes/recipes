# Experimental demand-fill DMA backend

Select `SLOT_CACHE_COPY_BACKEND=triton` (default) or `dma`. Only the transfer
mechanism changes: both use the existing fused GPU LRU bookkeeping, slot IDs,
scalar gathers, prefill bypass, and single routed MoE launch. There is no
speculative prefetch or new eviction policy.

From this recipe directory, with the usual model/cache/API-key environment:

```bash
# Existing graph-capable baseline.
SLOT_CACHE_COPY_BACKEND=triton bash scripts/launch-slotcache-portable.sh triton-graphs 112

# Eager control: isolate transfer changes from the cost of disabling graphs.
SLOT_CACHE_COPY_BACKEND=triton bash scripts/launch-slotcache-portable.sh triton-eager 112 --enforce-eager

# DMA: the portable launcher automatically adds --enforce-eager.
SLOT_CACHE_COPY_BACKEND=dma bash scripts/launch-slotcache-portable.sh dma-eager 112
```

Run candidates separately because these commands share the same serving port and
GPU budget. The historical `launch-slotcache.sh` uses external campaign files;
use the portable launcher for the packaged backend. With a custom launcher, pass
both the environment flag and `--enforce-eager`, and place `slot_cache_dma.py` and
`slot_cache_dma.cpp` beside `slot_cache_hook.py`.

## Implementation and dependencies

The DMA mode lazily builds a small C++ CUDA-runtime extension with PyTorch's
extension loader. It requires the CUDA-enabled PyTorch already used by the
recipe, a C++ compiler, CUDA toolkit headers/libraries, Ninja, and a writable
PyTorch extension cache (normally under `/root/.cache/torch_extensions`). There
are no CUDA kernels in the extension. The Triton mode never builds or imports it.

Each layer registers its original contiguous weight and block-scale banks.
No `.contiguous()` clone of a host bank is allowed in DMA mode. At registration,
`cudaPointerGetAttributes` determines the source allocation type. Registered host
memory uses CUDA's returned host alias and `cudaMemcpyHostToDevice`; same-device
memory uses `cudaMemcpyDeviceToDevice`. Destinations must be device allocations
on the selected GPU. Managed memory, unregistered/pageable sources, peer sources,
invalid shapes, and noncontiguous banks fail explicitly. Torch's `is_cuda` alone
does not classify mapped UVA backing. Per-bank attribute summaries and copy kinds
are logged at cache creation. See NVIDIA's
[pointer attributes](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaPointerAttributes.html)
and [memory-copy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).

After fused bookkeeping, the host reads the small source/destination/miss
descriptor array and compacts misses. One nonblocking `torch.cuda.Stream` with
priority `-1` per device submits `cudaMemcpyAsync` for each missed expert's two
weight rows and two block-scale rows. Small scalar gathers remain on the compute
stream. Ready/completion events order prior compute before slot writes and all
row writes before the consuming MoE. Calls for each layer remain serialized on
the inference compute stream; this is not a concurrent cache API. LayerCache
retains the tensor views and copy plan for the model lifetime; the existing UVA
offloader and `_keep_host` retain the pinned backing allocations.

## Limits and validation

The host descriptor read **synchronizes the current stream on every cached
forward, including all-hit steps**. Dynamic host-issued row copies are therefore
not CUDA-graph replay compatible. Capture is rejected before bookkeeping mutates
the cache. `--compilation-config '{"mode":3,"backend":"eager"}'` alone does not
disable CUDA graphs; use `--enforce-eager`. There is no silent fallback to Triton.

A demand fill remains a dependency of its own layer's MoE. A side stream does not
by itself hide that transfer or establish a throughput improvement. Actual
copy-engine/C2C activity and overlap must be verified on the target GB300 with a
CUDA profiler; no performance or model-quality result is claimed for this change.

From the repository root:

```bash
python -m pytest tests -q
# On the CUDA host/container (the UVA test additionally requires vLLM):
RUN_SLOT_CACHE_CUDA_TESTS=1 python -m pytest tests/test_slotcache_dma_cuda.py -q
```

CPU tests cover descriptor compaction, stream/event ordering, capture rejection,
and launcher flags. CUDA tests exercise real H2D/D2D copies, mapped UVA aliases,
invalid banks/indices, immediate slot reuse on a non-default compute stream, and
byte-exact parity with Triton through cold fills, all hits, duplicate routes,
evictions, block scales, and scalar gathers. An explicit CUDA test run fails if
CUDA dependencies are unavailable; only the vLLM-specific case can skip.

Before adopting DMA on the model, run the recipe's existing quality checks and
compare all three launch modes above with matching workloads. Report complete
decode latency including descriptor synchronization, not just transfer bandwidth.

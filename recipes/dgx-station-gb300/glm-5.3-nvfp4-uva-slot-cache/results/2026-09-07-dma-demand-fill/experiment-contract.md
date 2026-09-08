# GLM-5.3 GB300 slot-cache DMA experiment contract

Date: 2026-09-07
Status: frozen before target-GPU testing

## Inputs

- Repository: `J-M-Recipes/recipes`
- Baseline commit: `44e44aaee325545d3570b7fb4208fad06a835997`
- Fabian PR #1 head: `698e7d14e36dcdf3c40123faedb6e8f37b17799f`
- Image: `vllm-glm53-uva:v0.28.0-2cf0a691`
- Model: existing pinned `/home/exx/models/GLM-5.3-NVFP4-big`
- GPU: single NVIDIA GB300
- Constant speculative mode: native MTP(1)
- API model: `glm-5.3-big`

## Purpose

Determine whether replacing Triton SM demand-fill row-copy kernels with host-issued `cudaMemcpyAsync` improves end-to-end decode on the GB300. Do not infer success from bandwidth or startup alone.

## Fixed comparison

All arms use the same model, image, prompt harness, slot geometry, cache profile, MTP(1), output lengths and repetitions.

1. `triton-graphs`: `SLOT_CACHE_COPY_BACKEND=triton`, graph-capable launch.
2. `triton-eager`: same Triton backend plus explicit `--enforce-eager`.
3. `dma-eager`: `SLOT_CACHE_COPY_BACKEND=dma`; launcher adds explicit eager mode.

Primary 64K profile:

- `SLOT_CACHE=112`
- `SLOT_CACHE_PER_LAYER=/w/configs/slots-8400.json`
- `KV_CACHE_MEMORY=8589934592`
- `MAX_MODEL_LEN=65536`
- `MAX_NUM_SEQS=4`
- C1/C4/C8 prose: three repetitions each
- C1/C4/C8 code: one repetition each

## Preconditions

1. Preserve a deterministic restore handle for the currently verified 512K MTP lane.
2. Pass repository CPU checks and target-container CUDA parity tests.
3. Confirm DMA source banks classify as registered host/H2D and destination banks as device allocations.
4. Do not merge PR #1 before target evidence exists.

## Correctness gates

- CUDA row-copy tests pass, including mapped-UVA classification and byte-exact Triton/DMA parity.
- The service returns successful completions without NaN, crash or cache-integrity errors.
- Existing greedy/teacher-forced checks show no model-output regression attributable to the copy backend.

## Performance gates

### Continue gate

`dma-eager` must beat `triton-eager` by at least 5% at C1 end-to-end decode throughput. Otherwise stop DMA work and restore the proven 512K MTP lane.

### Promotion gate

DMA may remain live only if all are true:

- `dma-eager` beats `triton-graphs` at C1 by at least 5%;
- it has no greater than 5% aggregate regression at C4 or C8;
- correctness gates pass;
- Nsight Systems shows copy activity consistent with copy-engine/C2C transfer and no unexplained per-layer synchronization dominating TPOT;
- a matched 512K-profile decode-at-occupancy check beats the preserved 512K Triton/MTP baseline.

A 64K-only win is publishable research evidence but is not sufficient to replace the 512K daily lane.

## Profiling evidence

Capture at least one representative C1 decode under `dma-eager` with Nsight Systems CUDA tracing. Record:

- CUDA memcpy API calls and durations;
- GPU memory-copy activity;
- Triton copy-kernel absence/presence;
- CPU/device synchronization around descriptor reads;
- end-to-end request TPOT.

API selection alone is not proof of copy-engine execution.

## Stop and restore

On failed build, failed correctness, continue-gate failure, or promotion-gate failure:

1. Stop/remove the experiment candidate.
2. Restart the exact preserved 512K MTP container.
3. Verify authenticated `/v1/models` and an exact completion probe.
4. Record the negative result; do not silently omit it.

## Publication

After final restore/promotion proof:

- add receipts, comparison table, profiler summary and limitations to the recipe;
- credit Fabian (`onthehub97`, `@onthexitter69`);
- update PR #1 with measured target-hardware evidence before merge decision;
- update and deploy the relevant `al-engr.com` GLM-5.3/GB300 surfaces;
- verify GitHub state and public URLs after writes.

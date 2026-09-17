# O1-a — launcher/kernel address path

Worker: grok-4.6 (track1) · 2026-09-17 11:32 CDT
Read-only. No GPU. No container start/stop/rm/pull. Image cubin-cache listing taken with `docker cp` from stopped `dsv41-vllm-v14-1M-ksched-agent-BOUND-REF`.

Sources: FlashInfer tag `v0.6.18` @ `69ff11fc4954396d98326656dc85debd2223f637` (`scratch/flashinfer/`); cubin ABI headers fetched from the v0.6.18 BMM pin (`results/track1/srcdump/SOURCES.md`).

Line numbers below are from those trees. Image-equivalent Python paths match E1 (`flashinfer/fused_moe/core.py`).

---

## Q1. Cubin or source?

**Cubin.** `trtllm_fp4_block_scale_routed_moe` is not JIT-compiled CUDA source. The SM100 module is compiled from FlashInfer C++ *runners* that `cuModuleLoadData` a precompiled trtllm-gen Batched GEMM cubin fetched at runtime.

Call chain:

1. Python `trtllm_fp4_block_scale_routed_moe` → `get_trtllm_moe_sm100_module().trtllm_fp4_block_scale_moe(...)`  
   `scratch/flashinfer/flashinfer/fused_moe/core.py:6260–6445` (image: same path; E1 README L13).
2. JIT spec `gen_trtllm_gen_fused_moe_sm100_module` compiles the runners with `-DTLLM_GEN_GEMM_CUBIN_PATH="<bmm_path>"` and downloads `flashinferMetaInfo.h` + `trtllmGen_bmm_export/*` from artifactory.  
   `scratch/flashinfer/flashinfer/jit/fused_moe.py:251–342`.
3. Cubin repository default:  
   `https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/`  
   `scratch/flashinfer/flashinfer/jit/cubin_loader.py:35–38`.
4. BMM pin used by 0.6.18:  
   `8ec29a98612c3670f9f28825d1ed19f09496073b/batched_gemm-fa419f4-31ee4e5/`  
   `scratch/flashinfer/flashinfer/artifacts.py:141–143` (`ArtifactPath.TRTLLM_GEN_BMM`). Checksum of `checksums.txt` is `011635d3c36756addcdc148eea90c984f2d7611ba375626aaaf466d355c80e50` (`CheckSumHash.TRTLLM_GEN_BMM`, artifacts.py:169–171). Verified live against artifactory.
5. Load: `gemm::loadCubinData` calls `flashinfer::trtllm_cubin_loader::getCubin(TLLM_GEN_GEMM_CUBIN_PATH + "/" + FunctionName + ".cubin", sha256)` then `cuModuleLoadData`.  
   `results/track1/srcdump/bmm_export/GemmOptions.h:2221–2234`.  
   Python callback: `scratch/flashinfer/include/flashinfer/cubin_loader.h:52–57` + `scratch/flashinfer/flashinfer/jit/cubin_loader.py:237,314–322`.
6. Launch: `BatchedGemmInterface::run` → `loadCubinData` → `cuModuleGetFunction`.  
   `results/track1/srcdump/bmm_export/BatchedGemmInterface.h:756–762`.

**Cubin files (named, not present as blobs in the image cache):** `checksums.txt` lists **6846** `.cubin` (3378 `sm100f`, 3378 `sm107a`, 90 `sm103a`). MXFP4 SwiGLU dynamic-batch SM100 examples (120 names in `srcdump/cubin_list/mxfp4_swiglu_dynB_sm100f.txt`):

- `Bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x256_s6_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f.cubin`
- `Bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x256u2_s6_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f.cubin`

**Image `~/.cache/flashinfer`:** `docker cp` from stopped v14 REF → `/root/.cache/flashinfer/0.6.18/103a/` holds generated `flashinferMetaInfo.h` (14.7 MiB) + jit log. **Zero `.cubin` files.** Cubins land on first `getCubin` into `FLASHINFER_CUBIN_DIR` (env, else `flashinfer-cubin` package, else `~/.cache/flashinfer/cubins`; `scratch/flashinfer/flashinfer/jit/env.py:66–110`). Did not start a container to list a live `FLASHINFER_CUBIN_DIR`.

---

## Q2. Weight addressing in the launcher

**One base pointer + expert-index × stride.** Not a per-batch pointer array. BatchedGemm has **no** pointer-array mode.

Python/C++ handoff is a single `data_ptr()` for the whole `[E, …]` tensor:

- FP4 launcher bind: `args->gemm1_weights = gemm1_weights.data_ptr();` and `args->gemm1_weights_scale = gemm1_weights_scale.data_ptr();`  
  `scratch/flashinfer/csrc/trtllm_fused_moe_kernel_launcher.cu:3108–3118`.
- Shared-expert comment: permutation maps `global_id → weight row (global_id - local_expert_offset)`  
  same file `:1570–1573`.
- MoE runner passes that one pointer into PermuteGemm1 / Gemm2:  
  `scratch/flashinfer/csrc/trtllm_fused_moe_runner.cu:869–877` (`args.gemm1_weights`, `args.gemm1_weights_scale`) and `:921–927` (`args.gemm2_weights`, `args.gemm2_weights_scale`).
- PermuteGemm1::run → `mRunner.run(..., hiddenState, hiddenStateScale, weights, weightsScale, ...)` with `numBatches = numExperts`  
  same file `:505–511`.
- Because MoE sets `transposeMmaOutput = true` (`:561–570`), the batched-GEMM runner **swaps** A/B: weights become `mPtrA` / `mPtrSfA`.  
  `scratch/flashinfer/csrc/trtllm_batched_gemm_runner.cu:341–344`:
  ```
  mPtrA  = transposeMmaOutput ? b : a;   // b = weights
  mPtrSfA = transposeMmaOutput ? sfB : sfA;
  mPtrB  = transposeMmaOutput ? a : b;   // a = activations
  mPtrSfB = transposeMmaOutput ? sfA : sfB;
  ```

**BatchedGemmData ABI** (`results/track1/srcdump/bmm_export/BatchedGemmInterface.h`):

- `void const* mPtrA{nullptr};` L129 — for batchN (weights-as-A, MajorK): logical shape `[B, divUpMul(M, tileM), K/S]`, strides `[divUpMul(M, tileM)*K/S, K/S, 1]` (L111–113).
- `void const* mPtrB{nullptr};` L200 — same idea on the other side.
- No `mPtrA[]`, no `ptrArray`, no `void const**`. Repo-wide grep of the export headers for pointer-array spellings: **zero hits**.
- `BatchMode` is only `{BatchM, BatchN}` (`BatchedGemmOptions.h:82`). That is which GEMM dimension is the expert/batch axis, not a pointer-array switch.
- Device params: **one** TMA descriptor each — `CUtensorMap tmaA[1];` / `tmaB[1];` / `tmaSfA[1];` / `tmaSfB[1];`  
  `results/track1/srcdump/bmm_export/KernelParamsDecl.h:70,107,144,170`.  
  Plus `void const* ptrA` + `uint64_t strideInBytesA` (L193–197) and the same for B (L202–205).
- Host-side fill: `params.ptrA = ptrA; params.strideInBytesA = options.mK * dtypeGetNumBits(dtypeA) / 8;`  
  `results/track1/srcdump/bmm_export/KernelParams.h:491–497`.

Same contract is still on NVIDIA/TensorRT-LLM `main` (`cpp/.../batchedGemm/trtllmGen_bmm_export/BatchedGemmInterface.h`: single `mPtrA`/`mPtrB`, same `[B, paddedM, K]` strides). Not a FlashInfer omission.

---

## Q3. Any weight-side indirection today?

**None on the weight tensors.** Token-side tables exist; weight rows are dense `[expert, …]`.

Token-side (workspace, not weights) — `scratch/flashinfer/include/flashinfer/trtllm/fused_moe/runner.h:376–389`:

- `expanded_idx_to_permuted_idx`
- `permuted_idx_to_expanded_idx` / `permuted_idx_to_token_idx`
- `cta_idx_xy_to_batch_idx`, `cta_idx_xy_to_mn_limit`, `num_non_exiting_ctas`

`ctaIdxXyToBatchIdx` maps a CTA to **which expert (batch index)** to read from the *contiguous* weight tensor (`KernelParams.h:454`: `params.ctaIdxXyToBatchIdx[cgaOffset + cga] = b`). It does not relocate expert `b`'s storage.

**LoRA `gemm1_lora_delta` is not weight-pointer math.** It is `BiasType::Mn`: a 2-D buffer added to FC1 *output* (activation-side), mutually exclusive with `gemm1_bias` (`trtllm_fused_moe_kernel_launcher.cu:4013–4014`). Indexing, when used, is `mPtrPermutedIdxToBiasRowIdx` → row in the bias buffer (`BatchedGemmInterface.h:302–313`; runner.cu:860–862). Closest existing hook, but it gathers **bias rows**, not expert weight bases. Python LoRA path only clamps `expert_idx` into `[0, local_num_experts-1]` to scale the delta (`scratch/flashinfer/flashinfer/fused_moe/core.py` routed_fn dump L173–184 / image `core.py` ~same).

`expert_map` never reaches this kernel (E1 README).

---

## Q4. Minimal change

**CUBIN CHANGE (upstream).** Pointer-array mode does not exist in the kernel ABI, so it cannot be exposed by a FlashInfer-only launcher patch.

What would have to change:

1. **trtllm-gen Batched GEMM generator (NVIDIA/TensorRT-LLM `cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/`)** — the cubin itself. KernelParams today has `tmaA[1]` (one descriptor covering all experts via batch stride). Non-contiguous expert rows (HBM vs pinned host, arbitrary bases) need either:
   - an array of TMA descriptors / `void* const*` weight bases indexed by batch id, **or**
   - an `int64_t[E]` byte-offset table applied instead of `b * stride`.
   Same for scale TMAs (`tmaSfA[1]` / `tmaSfB[1]`).
2. After new cubins exist: FlashInfer launcher plumbing (`trtllm_batched_gemm_runner.cu` InputBuffers fill, `MoERunnerArgs`, Python `trtllm_fp4_block_scale_routed_moe`) to pass the table. That part is launcher-only **and is not sufficient alone**.
3. Autotune hash: **yes, it changes.** New cubin function names / shapes → new FlashInfer autotune hash (~75 min first boot on this box). Offload GiB/util would still not.

A launcher-only rebuild of a single TMA cannot point at mixed HBM/host allocations: CUDA TMA requires one tensor with regular strides. That is O1-b (single VA), not this path.

FlashInfer cannot invent a cubin field the generator does not emit. This is a TRT-LLM feature request; FlashInfer is the consumer.

---

## Q5. Scales

**Same addressing scheme as weights, with the FP4 block-scale layout on top.**

- Bound together: `gemm1_weights_scale.data_ptr()` next to `gemm1_weights.data_ptr()` (`kernel_launcher.cu:3108–3109`); runner passes `weightsScale` as `sfB` which becomes `mPtrSfA` under transpose (`batched_gemm_runner.cu:342`).
- ABI: `mPtrSfA` / `mPtrSfB` (`BatchedGemmInterface.h:154, 233`). For MxFp4/NvFp4 R128c4, logical `[paddedM, K/P]` with `paddedM = divUpMul(M, tileM) * B` when A is weights (`:140–149`). Device: single `tmaSfA[1]` / `tmaSfB[1]` (`KernelParamsDecl.h:144, 170`).
- DSV4.1 geometry (E1): `gemm1_weights_scale [E, 4608, 160]` after `nvfp4_block_scale_interleave` (leading E, same as `gemm1_weights [E, 4608, 2560]`). `gemm2_weights_scale [E, 5120, 72]`.
- Any expert indirection **must** cover weights **and** the interleaved scale tensors (and per-expert bias/alpha/beta if present). Scale layout is R128c4/swizzled, not a different *index* formula — still `expert * padded_row_stride` from one base.

---

```
O1-a VERDICT: CUBIN CHANGE (upstream)
- kernel delivery: cubin (files: 6846 in pin 8ec29a98…/batched_gemm-fa419f4-31ee4e5/; e.g. Bmm_E2m1_E2m1E2m1_Fp32_*_swiGlu_dynB_sm100f.cubin ×120; image ~/.cache/flashinfer has 0 cubin blobs — fetched on demand)
- weight address = single mPtrA/mPtrB + batchIdx × stride [B, divUpMul(M,tileM), K]  (BatchedGemmInterface.h:111-129, 200; KernelParamsDecl.h:51-56, 193-205; kernel_launcher.cu:3108; runner.cu:505-511; batched_gemm_runner.cu:341-344)
- pointer-array mode in BatchedGemm: no (BatchedGemmInterface.h InputBuffers; BatchMode is BatchM|BatchN only, Options.h:82; tmaA[1] not tmaA[E], KernelParamsDecl.h:70)
- weight-side indirection today: none (token-side cta_idx_xy_to_batch_idx / expanded_idx_to_permuted_idx only; LoRA gemm1_lora_delta = BiasType::Mn on activations, Interface.h:302-313)
- minimal change: cubin-generator in TRT-LLM batchedGemm (new TMA-desc array or int64 expert-offset table for weights+scales); FlashInfer launcher then plumbs it. Cubin touched: yes. Autotune hash: yes (new function names).
- scales: same scheme (mPtrSfA/mPtrSfB, tmaSfA[1], leading-E; R128c4 layout on top of the same expert index)
- confidence: high — verdict is from the cubin-export ABI this image actually loads (checksum-matched pin), not a guess from Python. TRT-LLM main still has the same single-pointer ABI.
- open questions: (1) live FLASHINFER_CUBIN_DIR after a serving boot not listed (would need a running container; stopped-image cache has 0 cubins). (2) O1-b single-VA remains the only bit-exact path that does not wait on NVIDIA. (3) Whether NVIDIA would accept an offset table vs per-expert TMA array — ask, don't guess.
```

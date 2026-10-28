# W11 Xid 31: the out-of-bounds read (desk analysis, 2026-10-28)

## TL;DR
- **The culprit is not DeepGEMM.** The faulting read comes from vLLM's block-verification sampler: `_compute_local_residual_mass_kernel`, `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:435` (af7f9488):
  `if tl.load(draft_sampled_ptr + logit_idx + 1) < 0:`
  For the last logit this reads `draft_sampled[num_logits]`, which is one int32 past the end. The guard at :429 compares against `num_speculative_steps` (5). Our k-schedule `[[1,4,5],[5,8,3],[9,32,3]]` runs k=3 at C9–C32, so the bonus row (local position 3) gets past the guard.
- **Why it faults at exactly 0 B past the end.** At C32 all-decode the batch has 32 × (3+1) = 128 logits. `draft_sampled` (`rejection_sampler.py:299`, int32) is therefore **512 B, exactly one allocator block**. In both W11 crashes (S6 and S7) that tensor sat at segment offset 0x1FFE00 and ended exactly at the segment end (`gap_to_end 0, req 512, rejection_sampler.py:299`, ~31 ms before the fault dump). Logit 127 reads byte 512, which is the first byte of the unmapped 14 MiB hole → `FAULT_PDE VIRT_READ`.
- **Upstream already fixed it:** vLLM **#60945**, "Fix out-of-bounds lookahead in adaptive block verification" (merged 2026-10-12, `ac28877f0e`). af7f9488 is from 2026-09-28 and is 830 commits behind it. The PR's compute-sanitizer report matches our fault exactly: *"Invalid __global__ read of size 4 bytes at `_compute_local_residual_mass_kernel` rejection_sampler_utils.py:435, block (127,0,0) … 1 bytes after … allocation … of size 512 bytes."* The fix is a 3-line guard and is bit-identical (see below). Backport: `patches/01-…DRAFT`.
- **The current production REF is exposed too.** `dsv41-vllm-1017-h1-REF` runs the same config: block + probabilistic + the same k-schedule + 32 seqs (from `args-before-…-h1-REF.json`). It only crashes when `draft_sampled` happens to land on an exposed segment tail during a full C32 decode step. Otherwise the 4-B over-read is silent and harmless (the row it feeds is never used).
- **DeepGEMM's `transform_weights_for_mega_moe` is a bystander.** Its per-layer shared-expert SF allocations create the small 2 MiB segments that sit in front of a 14 MiB unmapped hole. That makes their free tail the only place where a small over-read faults instead of reading a neighbour. Padding the SF buffers would only move `draft_sampled` elsewhere and make the bug silent. **Don't ship that as the fix.**

## 1. The transform buffers: arithmetic and why they're cleared
- Vendored file = vllm-project/DeepGEMM@e1f418c2 `deep_gemm/mega/__init__.py`. Line numbers match the trace: `_transpose_sf_for_utccp` :150 is `torch.empty_like(sf).copy_(…)`; calls at :164 (L1) and :167 (L2). They are reached from `deepseek_v4/nvidia/model.py:474 _finalize_shared_expert_weights`, so these are **shared-expert** SFs.
  - **737,280 B** = 4,608 rows × 40 int32 × 4 B. That is L1 = 2×2304 rows; K = 5120/32 = 160 UE8M0 = 40 packed int32.
  - **368,640 B** = 5,120 × 18 × 4. That is L2; K = 2304/32 = 72 UE8M0 = 18 int32.
  - The three blocks fill 1,843,200 B (0x1C2000). The remaining **253,952 B** (0x200000−0x1C2000) is the allocator's free tail, used as transient scratch every step.
- **163,072** is not a DeepGEMM size. It is the stale `requested_size` of the last transient that used that tail: a torch op at `flash_mla_mega_attn.py:734` (76 allocations of that size in the S7 trace). In other snapshots the same tail reads 131,072 (`rms_norm`).
- The fault is 253,952 B past the last SF block. That is not a "+1"-style overrun.
- Every SF read is TMA:
  - `make_tma_sf_desc` (`csrc/jit_kernels/impls/runtime_utils.hpp:262-280`) encodes extents equal to the checked tensor shape (`check_sf_layout`, `csrc/utils/layout.hpp:96-123`).
  - The shared-expert descriptors are built in `sm100_fp8_fp4_mega_moe.hpp:121-156`.
  - TMA zero-fills out-of-bounds boxes and does not fault.
- Also cleared while hunting:
  - **Mega mHC:** coefficient and residual loads are all TMA with `num_tokens` extents (`sm100_mega_mhc.hpp` `make_coeff_tma_desc`). Its 2,560-B `new_prev_mix` (`mega_mhc.py:69`) does also end at the segment end some steps.
  - **Mega gate:** `mask`, `bias` and `logical_count` reads are bounded by `num_valid_tokens` / `is_valid_expert` (`sm100_bf16_mega_gate.cuh:240-250,327-330`).
- The fault client `GPCCLIENT_T1_5` (L1TEX/LSU) fits a Triton `ld.global` and does not fit TMA.
- **Why only the single-process W11 run faults.** Each process has 8 of these per-layer SF segments, each followed by a 14 MiB hole; their ends all have low29 = 0x1F200000, which is the "R5" class. A 512-B transient lands in the last slot of one of them only after a whole ladder of allocation history.
- **Why split-KV ON/OFF makes no difference:** the faulting read is in the sampler.

## 2. Upstream check (worker report: `upstream/UPSTREAM.md`)
| Fix | Applies to our tree? | What it changes | Bit-identical? | Stops this fault? |
|---|---|---|---|---|
| **vLLM #60945** (`ac28877f0e`, 10-12) | Not in af7f9488; the hunk applies cleanly (dry-run OK) | `if logit_idx + 1 >= tl.num_programs(0): return` at the top of `_compute_local_residual_mass_kernel` | **Yes.** The rejection kernel reads residual rows only through `_compute_global_residual_mass(…, logit_idx+1)` under `i < num_draft_tokens-1`, so row ≤ end−2. The bonus row's residual is never consumed. | **Yes**, it removes the read |
| vLLM #61451 (perf, skip bonus-logit reduction for short drafts) | No | 20/10 lines, perf only | not verified | not needed |
| vLLM #61161 (block-verification random-draw reuse bugfix) | No | changes RNG draws | **No** (numerics) | unrelated. Separate decision; it affects block-verification quality |
| DeepGEMM #480 / #451 / #462 (fork #30) | No | #480 is the TMEM base in hc_prenorm; #451 is a combine-flag reset; #462 is a mega-kernel rewrite with no SF bounds change | #462 not bit-identical | **No.** No upstream DeepGEMM fix exists because there is no DeepGEMM bug here |

## 3. Mitigation that doesn't depend on the upstream kernel edit
- `patches/02-mitigation-draft-sampled-guard-slot.diff.DRAFT` is a Python-only change at `rejection_sampler.py:299`. It allocates `num_logits + 1` int32 and fills the extra slot with `-1`, then hands the kernel the `[:-1]` view.
  - The +1 read then stays in bounds and takes the kernel's existing "`-1` placeholder → return" branch.
  - Every in-bounds value is unchanged; checked on CPU (torch 2.8): equal, contiguous, chunk slices equal, no warnings.
  - Cost: one 1-element fill per step. No Triton recompile.
- Prefer patch 01 (the upstream line). Patch 02 is the fallback if James wants zero kernel-source change.
- Config-only escape hatches exist (k=5 in every band, or `draft_sample_method` greedy) but they cost speed or acceptance. Not recommended.
- **Rejected:** padding or large-pool allocation of the DeepGEMM SF buffers. It removes the 8 known guard pages (≈+41 MiB HBM), but other small segments border holes too (~50 per process; `evidence/exposed_small_segments.py`). It also leaves the 4-B over-read silently in place.

## 4. How to test in the next dark window (≈25 min; GPU worker only)
1. **Deploy.** Apply patch 01 to a copy of the image file and read-only bind-mount it over `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`, using the overlay-58132 pattern. Use the same att2e launch as W11. No autotune-hash impact: the change is a Triton kernel, which gets its own JIT cache key.
2. **Unit proof (~2 min, before boot).** The `-k shorter_than_configured_steps` test named in the PR body was never merged upstream, so use ours: `repro_residual_mass_oob.py.DRAFT` (untested, needs a GPU). It is the W11 shape: 32 reqs × 4 logits, steps=5, probabilistic, block.
   - Run it under `PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck --padding 4096 --error-exitcode 99`, unpatched and then patched.
   - Expect unpatched: an "Invalid __global__ read of size 4" at :435 block (127,0,0). Patched: clean.
   - Without the sanitizer, the saved `sampled`/`num_sampled` must be identical between the two runs.
3. **W11 repro (~10 min).** Re-run the exact W11 single-process ladder (all cells C1/C8/C32 × 0K/64K, split OFF and ON). Add `-e CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 -e CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_abort`; it is currently `=0`.
   - If anything still faults, `/usr/local/cuda/bin/cuda-gdb -ex 'target cudacore <core>' -ex 'info cuda kernels'` on the host names the kernel.
   - **Pass:** 0 new Xid, all cells complete, and the maptrace shows `rejection_sampler.py:299` allocations as 516 B (patch 02) or unchanged (patch 01).
4. **Bit-identity gate.** Same-window control vs patched at C8 (no fault shape), temp 1.0 with fixed per-request `seed`: identical output tokens on the per-class prompt set. Repeat at temp 0. The bonus-row residual is provably unused, so any diff means a broken deploy, not noise.
5. Then fold patch 01 into the h1/v27 lane recipe. The lane has the same latent fault.

Files: `patches/01-backport-vllm-60945-residual-mass-oob.diff.DRAFT`, `patches/02-mitigation-draft-sampled-guard-slot.diff.DRAFT` (apply with `patch -p1 -d …/dist-packages`), `evidence/W11-segment-forensics.txt` (+ scripts), `evidence/vllm-pr60945-upstream.diff`, `src-af7f9488/`, `upstream/`.

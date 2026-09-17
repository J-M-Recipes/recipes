# E1b — split-invoke follow-up (quiet GPU) + O1-b hand-off

Window 1, 2026-09-17 12:15–12:40 CDT. Worker: **claude-fable-5.1 via nous** (two grok-4.6 dispatches died on xAI outages with zero work done). GPU **quiet**: `docker ps` empty, GB300 22 MiB used before and after; GLM-5.3 big left stopped-and-kept, nothing else touched, no DSV4.1 boot. Spike container `pin-e1b-w1` (`--rm`, stopped at the end). Cap 0.25 (`set_per_process_memory_fraction`), abort if free < 4 GiB. Driver on the box is **595.91.07 / CUDA 13.2** (spec said 595.84). **All timings UNCONTENDED.**

Scripts: `spikes/e1b_straddle.py` (A1–A6; imports `spikes/e1_split_invoke.py` for the layer/shuffle/timing helpers), `spikes/e1b_a5b_decompose.py` (A5b). Numbers: `results/e1b/spike.json` (merged), per-section raw dumps in `results/e1b/raw/`.

## Source facts established before touching the GPU (flashinfer 0.6.18, checksum-matched tree in `scratch/flashinfer`)

- The routed entry (`trtllm_fp4_block_scale_routed_moe`, unpacked `(ids, weights)`) goes through `RoutingInputMode::UnpackedPrecomputed` → `routingPrecomputed` (`csrc/trtllm_fused_moe_kernel_launcher.cu:732-777`). The caller's weights are **memcpy'd into `metadata.expert_weights` unchanged** (L828-833). `routing_method_type` does not touch them.
- `-1` is a real skip: `routingPermutation` marks `isLocalExpert = 0 <= id-offset < local_num` and writes `expandedIdxToPermutedIdx = -1` otherwise (`include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh:340-345, 512-517`).
- Finalize (`csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu:678-717` and VecLoad variant L842-939): skips `permutedIdx == -1`, accumulates **fp32 in k order** `data += float(w_k) * float(bf16 gemm2 row)`, casts to bf16 **once**. `finalizeKernel` is used when `(H/256)·T < 1184` (T ≤ 59 at H=5120), `finalizeKernelVecLoad` above — same arithmetic order.
- `do_finalize=False` returns `[gemm2_output (bf16, permuted rows), expert_weights, expanded_idx_to_permuted_idx]` (`flashinfer/fused_moe/core.py:1695-1735`).

Prediction from the source: the E1 "2^35 garbage" cannot be renormalization, and a split that runs **two finalizes** can never be bit-exact on straddling tokens because each half is rounded to bf16 before the sum. Both predictions held.

## A1 — localise (T=6/24/96; synthetic E=16, hot 11 / cold 5, seed 20260917)

| T | all-hot | all-cold | straddle | straddle max-abs (dummy split) | straddle % bit-id | hot call vs ref-hot-contribution | cold call vs ref-cold-contribution |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 6 | 1 | 0 | 5 | 2^35 | 72.6 | **0.0 (100 %)** | **0.0 (100 %)** |
| 24 | 7 | 0 | 17 | 2^35 | 66.7 | **0.0 (100 %)** | **0.0 (100 %)** |
| 96 | 27 | 0 | 69 | 2^35 | 72.9 | **0.0 (100 %)** | **0.0 (100 %)** |

Every 2^35 row is a straddling token. Each partition call reproduces its reference contribution (full-E call with the other partition's slots set to `-1`) **bit-for-bit**. Neither call is garbage — **the sum is**. Control: summing the two *reference* contributions (`bf16 + bf16 → bf16`) reproduces the split's error statistics exactly (T=24: max-abs 2^35, 66.68 % bit-identical, identical `mean_abs`). Sentinel-only split (no dummy indexing) gives the same numbers. Zero-expert rows (cold call on all-hot tokens) are exactly 0.

## A2 — renormalize hypothesis: REJECTED

Full-E call with every routing weight × 0.5: output is **exactly** `ref × 0.5` (100 % bit-identical to `(ref.float()*0.5).bf16()`, median ratio 0.5000). Weights pass through; the kernel does **not** renormalize.

## A3 — why exactly 2^35

The synthetic layer (random MXFP4 nibbles, random e8m0 scales 2^-8..2^8) produces outputs up to **~5.6e12 ≈ 2^42.4**. bf16 has an 8-bit significand: one ulp at 2^42 is **2^35**. The "garbage" is **one bf16 ulp of double rounding**, magnified by the synthetic layer's scale. The E1 read "not ulp-level" was wrong; `max_rel` 7.3–54.7 comes from cancellation rows where `ref` is near zero.

- (a) Dummy row scale bytes after `nvfp4_block_scale_interleave`: {119..135} (2^-8..2^8), no 0x00, no 0xFF. Not a factor (the dummy is never indexed in the sentinel split, which shows the same numbers).
- (b) `-1` slots with weight 0.0 vs the original nonzero weight: **identical** (max-abs 0). `-1` is a true skip; the weight is ignored.
- (c) Tokens with zero experts in a call: full-E call with token 0 all `-1` → row 0 is exactly 0, other rows bit-identical to ref; hot call with token 0 all-dummy w=0 → exactly 0. Not the 2^35 rows.
- (c6) Compacting a straddler's hot experts into slots 0..n-1 (trailing `-1`): still bit-identical to the reference contribution — slot position is irrelevant to the per-partition result.
- (c7) Full-E reference with the six slots **permuted**: max-abs 2^29, 99.996 % bit-identical — i.e. the kernel's own finalize is order-sensitive at the ulp level, as expected of sequential fp32 accumulation.

## A4 — fixes (T=6/24/96)

| fix | straddle bit-identical? | all-tokens max-abs (T=6/24/96) | % bit-id (T=6/24/96) | note |
|---|---|--:|--:|---|
| E1 dummy-row split (baseline) | no | 2^35 / 2^35 / 2^35 | 72.7 / 76.4 / 75.2 | two finalizes |
| F1 `routing_method_type=TopK(5)` | no | 2^35 / 2^35 / 2^35 | 72.7 / 76.4 / 75.2 | identical to baseline; full-E route5 == route1 bit-for-bit (routing type is inert on the pre-routed path) |
| F2 compaction (per-partition token gather, dummy *token* pad, scatter-add) | no | 2^35 / 2^35 / 2^35 | 72.7 / 76.4 / 75.2 | identical to baseline; the tokens called were 6/5, 24/17, 96/69 |
| F3 weight-share compensation | no | 1.6e12 / 2.0e12 / 2.1e12 | 17.0 / 29.5 / 28.3 | wrong by construction (A2: no renormalization) — recorded, not a candidate |
| **F4 unfinalized split + one manual finalize (fp32 FMA, original k order)** | **YES** | **0 / 0 / 0** | **100 / 100 / 100** | `do_finalize=False` on both calls; gather bf16 gemm2 rows by `expanded_idx_to_permuted_idx`; one accumulation `acc = fma(w_k, row_k, acc)` k=0..5; one bf16 cast |
| F4 with plain fp32 `acc + w*row` (no FMA) | no (1 ulp on a few elements) | 1.3e8 / 1.3e8 / 8.6e9 | 99.997 / 99.995 / 99.997 | proves the kernel's finalize is FMA-fused |

F4 controls: (i) the manual FMA finalize applied to the **full-E** unfinalized call reproduces the kernel's finalized output 100 % (30720/30720, 122880/122880, 491520/491520); the mul+add variant misses 1/6/16 elements. (ii) bf16 gemm2 rows gathered from the two partition calls are **bit-identical** to the rows of the full-E call for every (token, k), straddlers included — batch composition does not change a row's GEMM result. The FMA emulation in the spike is fp64 (`(w·x + acc)` rounded to fp32), which is exact for fp32 fma; a real implementation is a ~20-line CUDA finalize kernel using `fmaf`.

**Headline: straddling tokens are bit-identical under F4.** Track 2 is no longer "fp-reorder-neutral"; with F4 it is bit-exact, same GEMM kernels, same cubins.

## A5 — uncontended timing (CUDA-graph replay, 50 warmup / 200 iters, median µs)

E=16 (11/5), per layer:

| T | ref | dummy split | added | ×40 ms/step |
|--:|--:|--:|--:|--:|
| 1 | 42.3 | 63.7 | +21.3 | 0.85 |
| 6 | 72.8 | 100.1 | +27.3 | 1.09 |
| 24 | 97.6 | 148.9 | +51.3 | 2.05 |
| 96 | 122.4 | 206.1 | +83.7 | 3.35 |

Uncontended E=16 is within noise of the contended E1 numbers — the E1 split overhead was never contention.

**E=384, real 295/89 partition** (6.72 GiB/layer + 6.72 GiB partitions, proc alloc 14.3 GiB), per layer:

| T | ref | sentinel split (2 finalized calls) | added | ×40 | two **unfinalized** calls (F4 kernel-side floor) | added | ×40 | F4 incl. torch finalize chain | added |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 42.4 | 53.7 | **+11.2** | 0.45 | 48.3 | **+6.3** | 0.25 | 156.2 | +113.8 |
| 6 | 147.9 | 167.3 | **+19.4** | 0.78 | 160.2 | **+12.4** | 0.50 | 283.0 | +135.1 |
| 24 | 439.7 | 461.1 | **+21.4** | 0.86 | 452.4 | **+13.1** | 0.52 | 585.2 | +145.5 |
| 96 | 1058.9 | 1082.5 | **+23.6** | 0.94 | 1061.8 | **+2.7** | 0.11 | 1231.1 | +172.2 |

At the real expert count the second call costs +11–24 µs/layer (≤ 25 µs bar met at every T), and the two unfinalized calls cost +3–13 µs because each skips its finalize. The +110–170 µs of the F4 row is the spike's **~30-node torch finalize chain** (gathers, `where`, six fp64 FMA steps) — measured alone it is 74–101 µs; a single fused finalize kernel replaces it (the kernel's own finalize is one launch of the same shape). Random top-6 over 384 gives 75/96 straddlers at T=96 — far above real traffic (295 hot ≈ 97 % of routing mass), so real-traffic F4 traffic is lighter than this synthetic worst case. F4 graph implementations were re-checked at E=384: 100 % bit-identical to the full-E call, straddlers included.

## A6 — pinned host bandwidth, uncontended (cold-only call, T=96)

Two cases: the spec's E=16/cold 5 (5 unique experts, 94 MB — fits L2) and cold 48 (48 unique experts, 902 MB — exceeds L2). Four host paths, each also read by a plain `torch.sum` from the GPU as a kernel-independent control. Every path's kernel output is bit-identical to the HBM output.

| case | path | kernel eager µs | kernel graph µs | GB/s raw (eager/graph) | GB/s net of HBM control | torch SM read of same bytes |
|---|---|--:|--:|--:|--:|--:|
| cold 5 (94 MB) | HBM control | 238 | 88 | — | — | 2.5 TB/s (L2) |
| | `pin_memory()` direct | 983 | 832 | 96 / 113 | 105 / 126 | **335** |
| | `get_accelerator_view_from_cpu_tensor(pin)` | 980 | 829 | 96 / 113 | 105 / 127 | 336 |
| | `cudaHostRegister` + dev ptr | 986 | 836 | 95 / 112 | 105 / 126 | 339 |
| | `cudaHostAlloc` (ctypes, Mapped\|Portable) | 984 | 833 | 96 / 113 | 105 / 126 | 340 |
| cold 48 (902 MB) | HBM control | 451 | 304 | — | — | 4.6 TB/s |
| | `pin_memory()` direct | 4460 | 4367 | 202 / 207 | 217 / 222 | 375 |
| | UVA accelerator view | 4546 | 4460 | 199 / 202 | 213 / 217 | 368 |
| | `cudaHostRegister` | 4510 | 4432 | 200 / 204 | 215 / 219 | 370 |
| | `cudaHostAlloc` | 4479 | 4380 | 201 / 206 | 216 / 221 | 371 |

Reading: the **allocation path is irrelevant** — all four are within 2 %. Plain SM reads of the same host bytes run at the 335–375 GB/s ATS/C2C ceiling on every path. The **MoE kernel** reads host-resident weights at ~100 GB/s when 5 experts are active and ~215 GB/s when 48 are — it is concurrency-bound (few TMA streams in flight per active expert tile over C2C latency), not bandwidth-bound. **Bar ≥ 250 GB/s: not met on any path.** Decode-shaped cold traffic (a handful of cold experts per layer) sits at the ~100 GB/s end. Any B-split cold read pays ~18 MiB / 100 GB/s ≈ 180 µs per cold expert unless the cold call gets more concurrency (or a prefetch to HBM staging, which is a different design).

## E1b VERDICT: B-split VIABLE (bit-exact via F4) — cold-read bandwidth is the open cost
- padding strategy: dummy row only where `E_part < top_k` (never indexed); `-1` is a confirmed skip (weight ignored); F2 compaction unnecessary
- nondeterminism floor (max|ref_a-ref_b|): 0 (unchanged)
- straddling bug: **not a kernel bug** — one bf16 ulp of double rounding from two finalizes on a synthetic layer whose outputs reach 2^42; both partition calls are bit-exact vs their reference contributions; kernel does not renormalize
- split error vs ref (max-abs, max-rel, % bit-identical) at T=6/24/96 — two-finalize split: 2^35 / 2^35 / 2^35 ; 72.7 / 76.4 / 75.2 % — **F4: 0 / 0 / 0 ; 100 / 100 / 100 %**
- all-in-one-partition tokens bit-identical? yes; **straddling tokens bit-identical under F4? yes**
- added time per layer (median, UNCONTENDED, cuda-graph), E=384 295/89, T=1/6/24/96: two finalized calls +11.2/+19.4/+21.4/+23.6 µs → ×40 = 0.45/0.78/0.86/0.94 ms/step; two unfinalized calls (F4 floor) +6.3/+12.4/+13.1/+2.7 µs → ×40 = 0.25/0.50/0.52/0.11 ms/step; E=16 +21/+27/+51/+84 µs (same as contended)
- pinned host tensor accepted directly? yes, all four paths; derived GB/s: 96–113 (5 experts) / 200–207 (48 experts) in-kernel vs 335–375 torch SM read of the same bytes — allocation path is not the limiter; bar ≥ 250 not met
- GPU 0 attempt: not repeated (RTX not visible to docker, E1)
- open questions for James/Milo: (1) implement F4's fused finalize as a small CUDA kernel (fmaf, k order) and re-time — the ≤ 25 µs bar is met before it; (2) cold-read cost: ~100 GB/s at decode concurrency means ~180 µs per cold expert-read per layer; decide whether E2a's design tolerates that or needs cold experts staged into HBM ahead of the call; (3) the E1 "straddle garbage" verdict in the ledger should be read as superseded.

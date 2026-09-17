# E1c — cold-read cost at decode shapes (quiet GPU, no boot)

Window 1 follow-on, 2026-09-17 12:52–13:05 CDT. Worker: **grok-4.6 via xai-oauth**. GPU **quiet**: `docker ps` empty, GB300 22 MiB before and after; GLM-5.3 big left stopped-and-kept; no DSV4.1 boot; no GLM start. Spike container `pin-e1c-w1` (`--rm`). Cap 0.25, abort if free < 4 GiB. Image `vllm/vllm-openai:deepseekv41-flash-0909`. **All timings UNCONTENDED**, CUDA-graph replay 50/200 median (C3b eager — not graph-safe).

Scripts: `spikes/e1c_cold_read.py` (imports `spikes/e1_split_invoke.py` helpers, same pattern as e1b). Numbers: `results/e1c/spike.json`. v1 Triton-UVA gather took `cudaErrorIllegalAddress` and is archived in `results/e1c/raw/`.

Geometry: E=384, 295/89 partition, S=32 staging slots, 17.93 MiB/expert. C3 gather = unrolled `copy_` from a UVA CUDA view of `pin_memory()` (labelled first cut; Triton-over-UVA disabled after v1). C2 finalize = E1b kernel-finalize estimate (half of two_finalized−two_unfinalized at E=384: 2.7/3.55/4.35 µs at T=1/6/24) — **no fmaf kernel written**. C3b = `cudaMemcpyAsync` per row HostToDevice.

Routing forces exactly `n_host` distinct cold experts; remaining slots hot. T=1 n_host=22 skipped (n_host > T·top_k). T=1 n_host=6 is the all-cold token (need==0); v2 first pass over-filled that row with 295 hots — rerun after the `need==0` guard; other cells were already `need>0` and were not rerun.

Bit-identity vs C0 (full-E HBM): **C1, C2-F4, C3, C3b all 100%** on every measured cell. C3 gathered rows bit-identical to HBM source rows.

## Table — µs/layer (graph median) and host GB/s

GB/s: C1 over unique_all·17.93 MiB; C2-cold and C3-gather over n_host·17.93 MiB.

| T | n_host | uniq all/cold | C0 | C1 | C2 (unfin+fin est) | C3 gather | C3 MoE | C3 sum | C3b DMA | C1 GB/s | C2-cold GB/s | C3-g GB/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1 | 6/1 | 42.5 | 308.8 | 96.9 | 61.7 | 42.8 | 106.6 | 70.6 | 365 | 309 | 305 |
| 1 | 2 | 6/2 | 42.8 | 312.7 | 134.7 | 121.5 | 42.8 | 163.8 | 130.3 | 361 | 341 | 309 |
| 1 | 6 | 6/6 | 41.9 | 316.8 | 327.0 | 361.0 | 42.5 | 404.0 | 368.8 | 356 | 360 | 313 |
| 1 | 22 | — | SKIP | n_host > T·top_k | | | | | | | | |
| 6 | 1 | 34/1 | 143.2 | 1707.4 | **192.7** | 62.1 | 142.9 | **207.4** | 71.4 | 374 | 302 | 303 |
| 6 | 2 | 35/2 | 145.8 | 1749.6 | 240.6 | 121.8 | 145.5 | 269.2 | 130.0 | 376 | 343 | 309 |
| 6 | 6 | 33/6 | 134.9 | 1645.4 | 421.0 | 360.1 | 135.0 | 496.3 | 369.2 | 377 | 367 | 313 |
| 6 | 22 | 36/22 | 147.4 | 1790.2 | 1170.2 | 1351.5 | 147.0 | 1501.0 | 1359.3 | 378 | 375 | 306 |
| 24 | 1 | 121/1 | 440.7 | 5986.1 | 497.4 | 61.5 | 440.6 | 506.7 | 70.3 | 380 | 304 | 306 |
| 24 | 2 | 119/2 | 436.6 | 5892.3 | 533.6 | 121.3 | 436.5 | 561.2 | 129.9 | 380 | 340 | 310 |
| 24 | 6 | 110/6 | 406.3 | 5455.1 | 688.0 | 359.4 | 406.0 | 768.6 | 367.7 | 379 | 367 | 314 |
| 24 | 22 | 125/22 | 454.9 | 6201.0 | 1492.7 | 1353.7 | 454.9 | 1814.0 | 1360.1 | 379 | 371 | 306 |

C3 MoE (E=327 HBM) matches C0 (E=384 HBM) at every T — the staging call is not the cost. C3 extra is the gather. C3b DMA is ~8–10 µs slower than C3 SM/UVA row-copy at n_host=1 and within 1% at n_host=22.

## C1 at decode is ~360–380 GB/s, not ~100

E1b's ~100 GB/s was a **cold-only call with a small partition** (E=16 / 5 unique, T=96). C1 here is today's positional analogue: the **whole 384-expert layer tensor on pinned host**. At T=1–6 the kernel reads unique_all·17.93 MiB at **356–378 GB/s** (ATS/C2C ceiling, same band as E1b's torch SM host-read 335–375). T=1 with only 6 unique experts is already 365 GB/s — TMA concurrency follows the E=384 tensor, not the 5-expert cold split.

The main post's 4 ms/step ≈ 258 GB/s effective on today's lane is **not an underestimate of C1**. C1 is *faster* per byte than that figure. The 100 GB/s number is the B-split's small-E cold call, not today's offloaded layer.

Fetch tax vs C0 at T=1 (k=0-like, 6 unique): C1−C0 = 266 µs/layer. ×11.7 offloaded layers ≈ 3.1 ms/step — matches the 4 ms k=0 story. At T=6 the C1 wall-clock is 1.6–1.8 ms/layer because unique_all is ~34 (synthetic, almost no cross-token overlap) vs production k=5 ~22.5 unique.

## ×40 step cost (pin-hot on every layer vs all-HBM C0)

Headline cell T=6, n_host=1 (closest measured point to in-domain ~0.6 cold / 22.5 unique):

| | µs/layer | vs C0 µs | ×40 ms/step vs C0 |
|---|--:|--:|--:|
| C0 all-HBM | 143.2 | 0 | 0 |
| C1 whole layer host | 1707.4 | +1564 | 62.6 (not the design — only ~11.7 layers are offloaded today) |
| C2 F4 split | 192.7 | +49.5 | **2.0** |
| C3 staged | 207.4 | +64.2 | **2.6** |

In-domain (~0.6 cold): use n_host=1. Cross-domain (~4.5 cold): linear interpolate T=6 n_host=2 and n_host=6.

- in-domain step tax vs C0: C2 **2.0 ms**, C3 **2.6 ms**
- cross-domain (~4.5 cold): C2 ≈ 353 µs/layer → **8.5 ms**; C3 ≈ 411 µs/layer → **10.8 ms**

At n_host=22 (worst synthetic) C2 1170 µs still beats C1 1790 µs; C3 1501 µs is gather-bound (~306 GB/s · 22·17.93 MiB ≈ 1.35 ms).

## E1c VERDICT: SPLIT (C2) beats STAGE (C3) beats C1 — per-layer cost at T=6,n_host=1: C1 1707 / C2 193 / C3 207 µs; projected step tax in-domain C2 2.0 ms / C3 2.6 ms, cross-domain C2 8.5 ms / C3 10.8 ms

C1 at decode shapes is **~370 GB/s, not ~100**. The prize for pin-hot is beating C1's 1.6 ms/layer host MoE with a ~50–65 µs add-on (one cold expert), not rescuing a 100 GB/s kernel.

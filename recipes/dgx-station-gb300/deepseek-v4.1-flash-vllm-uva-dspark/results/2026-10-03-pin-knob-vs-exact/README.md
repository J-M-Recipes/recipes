# 2026-10-03 — `pinned_max_cached_size_mb` vs exact-size `cudaHostAlloc` on Grace (vllm#58178 / #58185 follow-up)

**Question (gitbisector, vllm#58178, 2026-10-03):** on torch 2.13, `PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024`
stops the CachingHostAllocator's power-of-two rounding for tensors above the limit. Does it hold on Grace (GB300 Station,
494 GiB LPDDR5X, C2C), and how does it compare with the exact-size `cudaHostAlloc` path that vllm#58185 uses by default?

**Box:** DGX Station GB300, driver 595.91.07, kernel 7.0.0-1019-nvidia-64k. Image `vllm/vllm-openai:nightly-cd10ed6f…`
(`sha256:20a52b807cef…`), torch **2.13.0+cu130**, CUDA 13.0. Another experiment (Kimi-K3 expert-pack histogram) held the GPU
during this run, so this is **host-allocation + C2C read microbench only; decode throughput was not measured.**

**Method:** `pin_knob_bench.py`, one process per arm (the allocator setting is process-wide, read at first use). Fill a
CPU uint8 tensor, then pin it one of three ways; report process VmRSS delta, system MemAvailable delta and `/proc/meminfo`
Shmem delta (all three agreed within 0.15 GiB — table shows VmRSS). Re-pin tax: pin/free/pin of a 2 GiB tensor ×3 in the
same process. Read check: 48 × 17.9 MiB contiguous reads/iter through `get_accelerator_view_from_cpu_tensor`, CUDA events.

| requested | `Tensor.pin_memory()` default | `pin_memory()` + `pinned_max_cached_size_mb:1024` | exact `cudaHostAlloc` (#58185 default path) |
|---|---|---|---|
| 2.25 GiB (= DSV4.1 `w2_weight`) | **4.00 GiB**, 0.18 s | 2.25 GiB, 0.11 s | 2.25 GiB, 0.15 s |
| 16 GiB | 16.00 GiB, 0.72 s | 16.00 GiB, 0.73 s | 16.00 GiB, 0.74 s |
| 47.7 GiB | **64.00 GiB**, 2.77 s | 47.70 GiB, 2.19 s | 47.70 GiB, 2.16 s |
| 94.6 GiB (= one DSV4.1 Engram table) | **128.00 GiB**, 5.57 s | 94.60 GiB, 4.27 s | — |

- C2C stream read through the UVA view, 16 GiB region, GB300: default **359.5 GB/s**, knob **358.8**, exact `cudaHostAlloc` **353.4** — same memory type, same bandwidth (±2%, within run noise; T4 bench measured 355–359 on this box).
- Re-pin of a freed 2 GiB buffer: default `[0.096, 0.011, 0.011] s` (cached after the first), knob `[0.094, 0.093, 0.093] s` (never cached) → **~85 ms per 2 GiB** on Grace, vs ~0.53 s reported on GB10.
- Rounding is exactly next-pow2 above 1 GiB in every default arm; 16 GiB is a power of two, so it is the only size where the three paths coincide.

**Reading:** the knob does on Grace what it does on GB10 — exact-size host footprint, same bytes as #58185's default path,
no read-bandwidth change. The difference is who sets it: the knob is a process-wide, torch-version-specific env var the
operator has to know about (and it affects every other pinned allocation in the process, e.g. sleep-mode backups);
#58185 makes the UVA offloader itself allocate exact-size by default and keeps `pin_memory()` as the opt-in.

**Not measured:** decode tok/s with the knob vs exact-size (needs the lane; GPU was occupied).

Receipts: `receipts/*.json` (one line per arm). Script: `pin_knob_bench.py`.

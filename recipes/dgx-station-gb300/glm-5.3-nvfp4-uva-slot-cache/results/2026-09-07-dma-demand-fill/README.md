# GB300 demand-fill DMA experiment — stopped at continue gate

**Verdict: STOP / not promoted.** Fabian's PR #1 correctly moved slot-cache miss-row transfers from Triton SM kernels to registered-host `cudaMemcpyAsync` H2D copies and passed the target-GPU CUDA parity suite. End-to-end decode did not improve: DMA eager was **3.28% slower at C1** than matched Triton eager and roughly **84% slower than Triton with CUDA graphs**. The frozen continue gate failed, so profiler, teacher-forced, and 512K promotion stages were not run. The preserved 512K/MTP daily lane was restored.

## Attribution

- Idea and implementation: Fabian, [`onthehub97`](https://github.com/onthehub97) / [`@onthexitter69`](https://x.com/onthexitter69)
- Evaluated PR: [`J-M-Recipes/recipes#1`](https://github.com/J-M-Recipes/recipes/pull/1)
- Exact PR head: `698e7d14e36dcdf3c40123faedb6e8f37b17799f`
- Baseline commit: `44e44aaee325545d3570b7fb4208fad06a835997`
- Target hardware: one NVIDIA GB300
- Image: `vllm-glm53-uva:v0.28.0-2cf0a691`

Fabian disclosed before testing that he had not built the patch on his own machine. The target-hardware implementation was good enough to compile, boot, classify all 75 source layers correctly as registered-host H2D, and pass the CUDA suite. The negative result is about this execution shape, not the quality of the contribution.

## Frozen comparison

All arms used the same GLM-5.3 NVFP4 checkpoint, vLLM image, native MTP(1), 65,536-token window, 8 GiB bf16 KV cache, sequence cap 4, `SLOT_CACHE=112`, per-layer `slots-8400.json`, prompts, output lengths, and repetitions. Slot-cache statistics were disabled identically across arms after the telemetry thread was proven to invalidate graph capture during startup.

| Arm | C1 prose mean | C4 aggregate mean | C8 aggregate mean | C1 code | Result |
|---|---:|---:|---:|---:|---|
| Triton + CUDA graphs | **55.27** | **103.29** | **100.59** | 51.55 | production-class reference |
| Triton + eager | 9.04 | 35.31 | 34.89 | 9.83 | graph-loss control |
| DMA + eager | 8.75 | 34.00 | 33.40 | 9.24 | failed continue gate |

Relative DMA result:

| Comparison | C1 | C4 | C8 |
|---|---:|---:|---:|
| DMA eager vs Triton eager | **-3.28%** | -3.72% | -4.25% |
| DMA eager vs Triton graphs | **-84.17%** | -67.08% | -66.79% |

The predeclared continue gate required DMA eager to beat Triton eager by at least 5% at C1. It instead lost by 3.28%. The 64K promotion gate therefore failed without ambiguity.

## Correctness and implementation evidence

The CUDA suite ran inside the production vLLM image on the GB300:

```text
5 passed, 1 warning in 41.47s
```

The warning was only pytest's inability to write `.pytest_cache` into the intentionally read-only source mount.

At model startup, all 75 MoE layers reported:

- source allocation type: registered host memory;
- host and device aliases available;
- copy kind: H2D;
- `w13`: 12,582,912 bytes per expert row;
- `w2`: 6,291,456 bytes;
- `w13_scale`: 1,572,864 bytes;
- `w2_scale`: 786,432 bytes.

Total copied per missed expert is 21,233,664 bytes in this implementation. The backend therefore exercised the intended CUDA memory-copy API against mapped host backing; this benchmark does not claim profiler proof of which physical copy engine carried each transaction.

## Interpretation

The patch answered the first question cleanly: replacing Triton row-copy kernels with individual `cudaMemcpyAsync` submissions does **not** improve this eager execution path. The likely costs identified before the run remain:

1. a device-to-host descriptor read and synchronization on every cached layer;
2. four individual copy submissions per missed expert;
3. demand fills remain dependencies of the current layer rather than being prefetched;
4. forcing eager execution discards the graph path responsible for most of the measured throughput.

The result does **not** show that GB300 copy engines are useless. It shows that DMA alone, in this host-driven eager design, loses end to end. A credible next generation would need batched submissions, no per-layer host synchronization, graph compatibility, and/or correctness-preserving expert prefetch far enough ahead to hide transfer latency. Those are new experiments, not implied wins.

## Startup incidents

Two graph-baseline startup attempts were preserved rather than erased:

1. An experiment-only fresh autotune key triggered an unnecessary long FlashInfer retune. The launcher was corrected to use the validated `slotcache-S112` cache.
2. With slot-cache telemetry enabled, the statistics thread issued a CUDA operation during graph capture and invalidated it. `SLOT_CACHE_STATS_SEC=0` was then fixed across every matched arm.

See `dma-startup-incidents.md`. Neither incident was counted as a DMA performance result.

## Gate disposition

| Gate | Result |
|---|---|
| CPU/repository checks | PASS before target run |
| GB300 CUDA parity suite | **PASS: 5/5** |
| Pointer/copy-kind validation | **PASS: 75/75 layers H2D** |
| DMA eager > Triton eager by 5% at C1 | **FAIL: -3.28%** |
| DMA eager > Triton graphs | **FAIL: -84.17% at C1** |
| Nsight Systems proof | NOT RUN — blocked by continue gate |
| Teacher-forced quality | NOT RUN — backend lost before quality stage |
| 512K occupancy promotion | NOT RUN — 64K promotion gate failed |
| Restore preserved 512K/MTP | PASS; see restore receipts |

## Files

- `experiment-contract.md` — frozen gates and matched arms
- `comparison.json` — machine-readable percentages and gate booleans
- `*-summary.json` — parsed per-arm measurements
- `*-bench.txt` — raw benchmark output
- `cuda-tests.txt` — target-GPU test receipt
- `dma-backend-facts.json` — sanitized implementation/runtime facts
- `dma-startup-incidents.md` — startup amendments and causes
- `restore-live-facts.json`, `restored-models.json`, `restored-smoke.json` — final service proof
- `SHA256SUMS` — checksums for every file in this evidence package

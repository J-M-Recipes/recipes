# 2026-10-05-lib-t1-v21 — llm-inference-bench baseline on v21 Fused Split

First run of the pinned third-party bench against this lane. It is the **baseline** that later
recipe changes are compared against with `--compare-baseline`. It is **not** a re-measurement of
the v21 headline numbers: those use our own prompt classes (`knee.sh` prose) and stay in the
`2026-09-27-v21-fused-split` bundle.

## What ran

| | |
|---|---|
| Server | `dsv41-vllm-v21-fused-split-BOUND-REF`: the v21 container unchanged (0909 image `sha256:00d577a6…`, hook v21, `--cpu-offload-gb 60`, `--max-num-seqs 24`, DSpark k-schedule `[[1,4,5],[5,24,1]]`, 1M context). Boot 2026-10-06 14:57 CDT, up 15:02. |
| Bench | [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) **v0.7.7 @ `c71ec1f2`** (`bench-version.txt`), self-update disabled (`LLM_BENCH_NO_UPDATE_CHECK=1`), no upload. The repo carries no license, so it is not vendored here. Pin the tag and clone it. |
| Client | M4 Max on the LAN → Station `:30006`. Client-side timing; the bench's hardware panel was off (no `nvidia-smi` on the client). |
| Commands | `command.txt` (host scrubbed to `<station>`) |
| System | `system.json` (from `scripts/collect_system_snapshot.sh`, collected 2026-10-05 17:18 UTC; driver 595.91.07, kernel 7.0.0-1019-nvidia-64k) |

T0 gate before the bench: `t0-chat.json` (exact-word reply) and `t0-toolcall.json` (parsed `tool_calls[]`, `get_weather`) both passed.

## T1: decode matrix (`t1-perf.json`)

Aggregate decode tok/s. Workload = **the bench's synthetic padding text** (calibrated 6.23 chars/token), 30 s per cell, 3 s warmup. Speculative decoding is on, and acceptance depends on content, so this filler is a different prompt class from our prose knee.

| ctx \ conc | 1 | 4 | 8 |
|---|---|---|---|
| 0 | 203.5 | 344.5 | 561.0 |
| 32K | 173.7 | 367.7 | 572.1 |
| 128K | 190.2 | 376.3 | 558.0 |

Accept length (tokens per engine step): 2.42–2.78 at C1, 2.33–2.45 at C4, 1.65–1.67 at C8. That matches the k-schedule: k=5 up to 4 running, k=1 above.

Prefill (integrated scouts, N=1 each): 8K 0.53 s / 15,381 tok/s · 32K 2.08 s / 15,534 · 64K 3.48 s / 18,486 · 128K 6.93 s / 18,549.

The exact-loop guard was on, and no cell was flagged.

## T2: quality

| Test | Result | Notes |
|---|---|---|
| `needle-checksum` (`t2-needle.json`) | **EXACT 500 / 500** | 500 identical greedy requests, 8K retrieval + arithmetic answered through a tool call; C8; 0 hit max_tokens. Any wrong answer would have been serving numerics. |
| GSM8K, 200-item pinned subset (`t2-gsm8k200.json`) | **97.5 % (195/200)**, Wilson 95 % 94.3–98.9 | Bench default C30; the lane caps at 24 running, so ≤6 requests queued. That affects latency, not scoring. 0 truncated, 0 unparseable. The 5 misses are listed in `t2-gsm8k200.stdout.txt`. |

The GSM8K file is the `--compare-baseline` reference for paired McNemar tests on future engine, kernel, quant, or speculation changes on this lane.

## Contention

`*.engine-load.txt` holds the vLLM `loggers.py` lines for each run window. The max running was 8 during T1/needle, equal to the bench's own top concurrency, and 24 during GSM8K (the C30 bench capped by `--max-num-seqs`). No other client was on the lane.

## Boot note

v21 failed to boot three times on 2026-10-05/06 with `PIN_HOT abort layer 9 after: MemAvailable 9.38–9.68 GiB < 10 GiB`. Six 2026-09-27 boots of the same container passed that point with 10.96–12.03 GiB available. This boot passed with 10.29 GiB after two orphaned host processes (~1.2 GB) were killed and caches dropped and compacted. **The host-memory margin at the hook's floor is now about 0.3 GiB.** About 1 GiB of the drift since 09-27 is unexplained. See the recipe `limits`.

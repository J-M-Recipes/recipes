# 2026-10-06-lib-t2t3-v21: llm-inference-bench T2 + T3 on v21 Fused Split

Promotion-tier pass of the pinned standard bench against the v21 container, on a fresh host boot.
Tiers and pinning: [`docs/benchmarking.md`](../../../../../docs/benchmarking.md). T1 baseline: [`../2026-10-05-lib-t1-v21/`](../2026-10-05-lib-t1-v21/README.md).

## What ran

| | |
|---|---|
| Server | `dsv41-vllm-v21-fused-split-BOUND-REF`, container unchanged (0909 image `sha256:00d577a6…`, hook v21, `--cpu-offload-gb 60`, `--max-num-seqs 24`, DSpark k-schedule `[[1,4,5],[5,24,1]]`, 1M ctx) |
| Host | Rebooted 2026-10-06 17:40 to clear kernel memory. Boot at 17:43 passed the hook's layer-9 host check at **12.47 GiB** (floor 10). Pre-reboot boots read 9.38–10.29. `meminfo-prereboot-serving.txt` / `meminfo-postreboot.txt`: unreclaimable slab 2.21 → 0.89 GB, vmalloc 1.06 → 0.94 GB, percpu 0.20 → 0.12 GB. |
| Bench | llm-inference-bench v0.7.7 @ `c71ec1f2`; tool-eval-bench pinned `570951a7` (`bench-version.txt`). Client M4 over LAN, `LLM_BENCH_NO_UPDATE_CHECK=1`, no upload. Commands in `command.txt`. |
| Window | 2026-10-06 17:49–23:37 CDT, no other clients (`*.engine-load.txt`) |

## Results

| Test | Result | Notes |
|---|---|---|
| tool-eval-bench, leaderboard settings (hardmode, seed 42, T=0, parallel 4) | **92/100**, 169/184 points; 92 scenarios: pass 81 · partial 7 · fail 4 | Fails: TC-68 schema violation resistance, TC-74 stateful multi-turn corrections, TC-89 compensation after partial success (told the user an invoice was paid), TC-92 tenant isolation with same-named resources (disclosed another tenant's data). Full report under `t2-tooleval.tool-eval/`. |
| needle-checksum, 500 identical greedy requests | **EXACT 500/500** | Same as the pre-reboot boot |
| GSM8K full test set | **96.9 %** (1278/1319), Wilson 95.8–97.7 | Paired vs the 200-item T1 baseline: 193/200 vs 195/200, −1.0 pt, exact McNemar p = 0.5, 2 flips one way. This is a same-config re-run on a new boot, so it is **this lane's noise floor**. A candidate must beat it to count. |
| MMLU-Pro, pinned 1,000 | **87.1 %** (871/1000), Wilson 84.9–89.0 | 13 hit the 131,072-token max, 3 of them with no answer |
| GPQA-Diamond, 198 | **89.9 %** (178/198), Wilson 84.9–93.4 | 8 hit max tokens, 6 with no answer. Per-item model output text is omitted from `t3-gpqa.json` (the dataset asks that examples not be republished). Ids, answers and scores are kept. |

### Decode matrix (`t3-perf-matrix.json`), aggregate tok/s, synthetic filler, 30 s cells

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|---|
| 0 | 214.3 | 289.3 | 359.3 | 581.5 | 804.9 | 995.4 |
| 16K | 192.5 | 278.3 | 369.0 | 576.0 | 811.6 | 1021.7 |
| 32K | 196.0 | 298.2 | 410.4 | 587.4 | 817.6 | 1010.8 |
| 64K | 192.2 | 273.7 | 391.3 | 581.5 | 813.0 | 1003.1 |
| 128K | 183.8 | 298.7 | 374.0 | 594.0 | 791.6 | skipped (24 × 128K > KV pool) |

Prefill (scouts, N=1): 8K 15,931 · 16K 18,673 · 32K 18,976 · 64K 19,434 · 128K 18,914 tok/s (6.80 s TTFT).

### Burst cells and the loop detector: read before quoting

The bench's exact-loop detector flagged **16 of 29 burst cells** (`burst_results[*].loop_detected`). Those cells report aggregate −3.0, the bench's invalid-cell sentinel.

**These are not answer-time loops.** The bench decodes with `ignore_eos=true` by default, forcing every request to the full 8,192 tokens. All flags fire 2.5K–5K tokens in, after the model would normally have stopped. The repeated text is the kind that forced continuation produces: echoed prompt filler and meta-commentary about the prompt. Every eos-respecting test on the same boot is clean (needle 500/500, GSM8K, MMLU-Pro, GPQA, tool-eval). Burst cells send 5 × C requests, so high-concurrency cells get more chances to trip the detector. The C ≥ 4 clustering is therefore not evidence of a concurrency or speculation effect. The 0/29 sustained cells are not a clean control either: their 30 s windows end before the detector's 4 KB threshold.

The post-EOS text excerpts are removed from the published JSON. Period, cycle count, positions and sha256 of each cycle are kept. A `--respect-eos` re-run of the burst cells is queued. Its result replaces this paragraph.

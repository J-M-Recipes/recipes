## llm-inference-bench 0.6.2 (Local Inference Lab's client), aggregate tok/s, 30 s windows, ignore_eos, engine-default thinking

| arm | C1 ctx0 | C8 ctx0 | C32 ctx0 | C1 16K | C8 16K | C32 16K |
|---|--:|--:|--:|--:|--:|--:|
| Pg66 (before) | 382 | 1,241 | 2,415 | 377 | 1,267 | 2,379 |
| Pg66 (after G) | 377 | 1,259 | 2,431 | — | — | 2,390 |
| Pg66 (after Flare) | 381 | 1,257 | 2,440 | 490 | 1,247 | 2,393 |
| gracetier (ours, 1 GPU) | 254 | 892 | 1,868 | 271 | 890 | 1,813 |
| Flare gb300 11/20 (1 GPU) | 303 | 1,529 | 4,931 | 276 | 1,355 | 3,845 |

## Per-class C1 decode tok/s (T=1, top_p 1, thinking off, exactly 256 tokens/prompt, 40 prompts/class, mean of seeds 42/43/44; client-timed)

| arm | prose | structured | code | shell_ops | tool_json | mean | natural-length mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| Pg66 (before) | 415.7 | 456.7 | 557.0 | 571.3 | 524.8 | **505.1** | 542.2 |
| Pg66 (after G) | 408.7 | 447.8 | 557.3 | 579.9 | 517.3 | **502.2** | 546.7 |
| Pg66 (after Flare) | 401.5 | 445.4 | 563.4 | 578.4 | 516.2 | **501.0** | 536.6 |
| Pg66 via kwproxy | 406.4 | 447.4 | 560.9 | 556.6 | 506.1 | **495.5** | 535.3 |
| gracetier (ours, 1 GPU) | 255.1 | 301.6 | 337.2 | 387.4 | 347.1 | **325.7** | 341.6 |
| Flare gb300 11/20 (1 GPU) | 254.5 | 292.2 | 313.9 | 313.3 | 308.1 | **296.4** | 307.8 |

## Prefill (cold, distinct filler + cache_salt, max_tokens 1; median of 3) — prefill tok/s (TTFT s)

| arm | ~14K | ~57K | ~114K | fixed-prompt TTFT ms (2.7K) |
|---|--:|--:|--:|--:|
| Pg66 (before) | 39,380 (0.36) | 39,904 (1.42) | 39,510 (2.88) | 139 |
| Pg66 (after G) | 40,135 (0.36) | 40,076 (1.42) | 39,562 (2.87) | 137 |
| Pg66 (after Flare) | 39,740 (0.36) | 39,639 (1.43) | 39,638 (2.87) | 143 |
| gracetier (ours, 1 GPU) | 41,234 (0.35) | 43,870 (1.29) | 42,922 (2.65) | 182 |
| Flare gb300 11/20 (1 GPU) | 45,687 (0.31) | 66,862 (0.85) | 73,525 (1.55) | 164 |

## Quality and long context

| arm | GSM8K-200 (T=0.3, thinking on) | 200K needles | tool call | agent gauntlet | 1M needle (depth 0.5) | fixed prompt 2,048 tok "greedy" tok/s |
|---|--:|--:|--:|--:|--:|--:|
| Pg66 (before) | 98.0% | 3/3 | PASS | 12/12 | 1/1 | 714 |
| Pg66 (after G) | 98.0% | 3/3 | PASS | 12/12 | 1/1 | 683 |
| Pg66 (after Flare) | 97.5% | 3/3 | PASS | 12/12 | 1/1 | 705 |
| gracetier (ours, 1 GPU) | 98.0% | 3/3 | PASS | 12/12 | 1/1 | 337 |
| Flare gb300 11/20 (1 GPU) | 97.5% | 3/3 | PASS | 12/12 | 1/1 | 395 |

## vLLM-counter cells (bench_full timed windows; void on Flare: no vLLM /metrics)

| arm | C8 agg | C32 agg (median of windows) |
|---|--:|--:|
| Pg66 (before) | 1,428 | 2,548 |
| Pg66 (after G) | 1,444 | 2,650 |
| Pg66 (after Flare) | 1,442 | 2,624 |
| gracetier (ours, 1 GPU) | 960 | 1,588 |

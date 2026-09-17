# E2c — confirm pair, expanded parity, held-out cross-domain

Worker: **grok-4.6 / xai-oauth**. T1c windows **v15a → v14 → v15b**. CAT_ORDER `prose structured code shell_ops tool_json`. GLM stayed stopped-and-kept. `:30003` untouched. v15 left **up** on `:30006`. Both boots hash-hit `9ac7b387` Loaded 231 / autotune 5 s (v14 bind 268 s, v15 rebind 257 s). No `docker rm`.

## Confirm table (knee is the verdict instrument)

C1/C8/C16 are the mean of two `knee.sh` invocations per window (each invocation itself averages two internal reps).

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| v15a | 153.5 / 153.3 | 704.6 / 701.6 | 952.6 / 956.7 |
| v14  | 88.7 / 88.8 | 304.7 / 305.8 | 385.5 / 417.3 |
| v15b | 153.1 / 153.2 | 703.0 / 706.5 | 812.1 / 957.8 |

C1 two-window v15 mean **153.28** vs v14 **88.74** = **+72.7%**.

C16 four v15 knee-run means: **952.6, 956.7, 812.1, 957.8**. The 812 is one mixed pair: v15b-r1 internal reps **667.4 / 956.8** (same 667/951 split E2b saw). Other six internal C16 reps all sit at 952–958.

### Other instruments (same order every window)

| | v15a | v14 | v15b |
|---|--:|--:|--:|
| agent prose tok/s | 168.8 | 97.3 | 169.0 |
| structured | 207.6 | 121.8 | 205.0 |
| code | 213.7 | 152.7 | 215.4 |
| shell_ops | 250.4 | 159.2 | 251.7 |
| tool_json | 282.0 | 152.3 | 222.4 |
| weighted accept | 59.9% | 59.6% | 59.6% |
| T5 prose fixture | 152.6 | 90.7 | 152.7 |
| replay n=4 r1/r2 | 476.1 / 461.8 | 160.3 / 223.8 | 259.2 / 434.6 |

Replay swing is the known instrument (~20%+/run). No fixture category is worse than v14.

## Parity (greedy T=0, seed 42, max_tokens 256, 50 prompts, C1)

Prompts: 8 agent-fixture + 8 T5 prose + 10 code + 10 math + 8 non-English + 6 structured/tool. Compare content + tool name/args (ids ignored, as E2b).

| pair | content+tools exact | content-only | tools-only |
|---|--:|--:|--:|
| v15a vs v14 | **49/50** | 49/50 | 50/50 |
| v15a vs v15b | **47/50** | 49/50 | 48/50 |
| v14 vs v15b | 48/50 | 50/50 | 48/50 |

Non-tool prompts **44/44** exact on every pair. Misses are all `tool_calls` preambles/args:

- `agent-tool-1`: content `"the host"` (v15a) vs `"that host"` (v14 and v15b); tool args identical.
- `tool-json-status`: `nvidia-smi …nounits` vs without `nounits` (v15b vs the other two).
- `tool-weather-time`: `date -u && date` vs `date -u` (v15b vs the other two).

Same class as E2b's control-vs-itself tool-id jitter — not a pin/split numeric break.

## Gauntlet (round-7 twelve-task suite on v15, frozen `run_suite.py`)

3 repeats × 12 tasks, workers=4, temp 0, seed 42, reasoning `low`, sandbox `sha256:00d577a6…` with `NVIDIA_VISIBLE_DEVICES=void`. Isolation gate passed.

| rep | success | wall_s |
|---|--:|--:|
| 1 | 12/12 | 10.48 |
| 2 | 12/12 | 10.53 |
| 3 | 12/12 | 10.38 |

**36/36**, matching the recorded v14 36/36.

## Held-out cross-domain (streaming first→last token, 5×400 tok)

Not in the E2a profile (agent fixture / T5 prose / Hermes transcripts).

| workload | v14 | v15a | v15b | v15 mean vs v14 |
|---|--:|--:|--:|---|
| math/proof English | 129.78 | 182.87 | 184.20 | **+41.4%** (never-worse holds) |
| non-English JP+DE essays | 93.02 | 90.77 | 92.39 | **−1.55%** (bar was ≥ −1.5%; v15a −2.42%, v15b −0.68%) |

Math is a clear win. Non-English is a wash: two-window mean misses the −1.5% floor by **0.05 tok/s** (91.58 vs 91.62). Spec accept is lower on JP/DE than on English math; this is not a C1-prose regression.

## Host watch (cgroup `memory.current`)

| window | start GiB | end GiB | Δ GiB | MemAvailable |
|---|--:|--:|--:|---|
| v15a | 351.491 | 351.543 | +0.052 | 125 → 125 |
| v14  | 464.206 | 464.232 | +0.026 | 69 → 69 |
| v15b | 350.931 | 351.020 | +0.089 | 126 → 125 |

No window grew > 2 GiB. v14's higher cgroup (~464 vs ~351) is the positional UVA pin, not leak.

## Left running

`dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` on `:30006` (`dsv41-flash-uva`, 1M). v14 `-RETIRED-REF` stopped-and-kept. GLM / `:30003` down.

## E2c VERDICT: C1 v14 88.7 vs v15 153.4/153.2 (mean +72.7%) · C16 v15 runs 952.6/956.7/812.1/957.8 · parity 49/50 exact (v15 vs v14), 47/50 (v15 vs v15) · gauntlet 36/36 · held-out math 129.8→183.5 (+41.4%), non-English 93.0→91.6 (−1.55%) · host Δ ≤0.09 GiB

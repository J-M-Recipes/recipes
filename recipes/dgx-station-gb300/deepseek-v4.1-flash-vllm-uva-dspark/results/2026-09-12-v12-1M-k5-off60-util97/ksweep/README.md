# Speculation depth: static k=1 vs k=5 vs DSpark adaptive verification — 2026-09-12 morning

All on the v12 residency (`--cpu-offload-gb 60 --gpu-memory-utilization 0.97`) except adaptive, which needed off66 (see below). Same window: control knee + fixture on the live v12 REF at 06:55, k=1 boot bound 08:47, adaptive bound 11:04. Knee = `knee.sh` (192-token prose, C1/2/4/8/12/16); fixture = `agent_fixture.sh` (five categories, accept rate, accepted drafts per step).

| | k=5 static (v12 REF) | k=1 static | adaptive verification, k≤5, off66 |
|---|---|---|---|
| knee C1 / C8 / C16 | **90.1** / 244.9 / 316.9 | **110.8** / 312.6 / **422.2** | 101.8 / 293.7 / 396.4 |
| prose | 98.5 (1.49 acc/step) | **106.4** (0.67) | 104.1 (0.90) |
| structured | 123.7 | 106.0 | **124.4** |
| code | **155.0** | 109.5 | 144.2 |
| tool_json | **156.0** | 102.9 | 138.8 |
| shell_ops | **161.4** (4.54) | 107.1 (1.00) | 145.3 (4.07) |
| weighted accept | 0.596 | 0.848 | 0.519 |
| KV | 4.89 GiB | 6.15 GiB | 5.64 GiB (at off66) |

**Read.** No static k fits the lane: k=1 wins prose by 8% and every multi-stream point by 23–33%; k=5 wins the agent categories by ~a third. The 3.7× unique-expert multiplier of a 6-token verify window (see `../routing/`) lands on the Grace link, and only high-acceptance categories pay it back.

**Adaptive verification** (`enable_adaptive_verification: true`, V2 model runner, DSpark confidence head — `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`) does what it says: prose is trimmed to ~k=1 (within 2% of the static k=1 boot), shell keeps ~k=5. It is the only config with no losing category, but it beats the best static point only on structured and **misses the promotion bar** (prose ≥105, shell ≥150). Two reasons, both upstream-shaped: the V2 runner's cudagraph capture takes ~7 GiB (V1 ~1 GiB), which forced `--cpu-offload-gb 66` and cost ≈4.5 tok/s of residency everywhere; and the boot-profiled cost curves price verification on dummy batches that do not see expert streaming, so drafts are under-priced on an offloaded lane. **v12 static k=5 remains the reference.**

Boot notes: changing `num_speculative_tokens` or switching to the V2 runner changes verify shapes → new FlashInfer autotune hash → ~75-min boot each. Adaptive at off60 failed with `Available KV cache memory: -0.69 GiB`.

Files: `knee-*.json`, `agentfix-*.json` (raw), `campaign_*.sh` (the runners, paths scrubbed).

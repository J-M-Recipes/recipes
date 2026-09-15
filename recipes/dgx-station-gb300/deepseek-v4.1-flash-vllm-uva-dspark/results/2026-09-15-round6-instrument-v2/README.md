# Round 6 — replay measurements and offload bracket

Measurements: September 15, 2026, 04:08–05:49 CDT. **Interpretation corrected September 15, 2026 after auditing the scripts and full failure log.** No serving configuration change; v14 remains the reference. Raw JSON, logs and original instruments in this bundle are unchanged.

## Correction: what these results do and do not establish

The original writeup overclaimed production capacity, cache causality, uncertainty and the offload boundary. The observed throughput and cache-counter values remain valid for the recorded workloads. The following conclusions are withdrawn:

- ~~233–275 tok/s is the production capacity of four agents.~~ These are recorded-history replay rates. The harness does not execute returned tool calls or feed generated answers into later turns; it uses a short replacement system prompt, flattened tool history and truncated contexts.
- ~~The earlier fixture understated the lane by about 35% because it defeated the cache; configurations rank the same either way.~~ Both workload and cache behavior changed. The shuffled fixture contains 150 tool-call turns; the ordered fixture contains 277 and uses a larger context cap. Only v14 was measured on ordered sessions. The difference cannot isolate a cache effect or establish a transferred ranking.
- ~~The warm pass represents continuing sessions or a capacity ceiling.~~ It repeats the same recorded requests. Newly appended turns, live tool latency and task success were not measured.
- ~~The bootstrap intervals quantify wall-clock aggregate uncertainty.~~ They target a different busy-worker estimator and independently resample correlated turns. They are not valid confidence intervals for the published wall-clock rates.
- ~~OFFGB=60 is the exact floor and lowering context cannot help.~~ OFFGB=55 failed the 1M minimum-KV check; OFFGB=60 works. Intermediate settings and shorter-context binding were not tested in this round.

These corrections supersede the original claims in the PR title/body and contemporaneous commentary. The recipe and blog are updated to match this narrower interpretation.

## 1. Shuffled recorded-history proxy

300 assistant turns, 150 with recorded tool calls and 150 text turns, sampled across sessions with at most two per original session. Four workers; streaming; `max_tokens=400`; temperature zero; `chat_template_kwargs.thinking=false`; three simplified tools offered. Context cap is 24,000 characters, not a tokenizer-enforced token limit. The original builder reported 229 truncated session identifiers; those identifiers are not collision-proof.

| Configuration | Pass a (completion tok/s / wall) | Pass b | Mean |
|---|---:|---:|---:|
| v14 | 171.1 | 174.1 | 172.6 |
| v13 | 150.0 | 151.7 | 150.85 |

The mean difference is **+14.4% on this specific proxy**, with no transport exceptions reported. Two repetitions are a descriptive repeat check, not a characterized repeatability distribution or general agent-quality verdict. The full observed range is 1.74% of the mean for v14 and 1.13% for v13. Prefix-cache hit ratios from server counter deltas were approximately 5%.

Tool-emission booleans do not prove valid arguments, successful tool execution or correct answers. The harness did not retain complete responses or finish reasons, and missing usage would default to zero; these limits cannot be repaired retrospectively by a new summary.

## 2. Session-ordered recorded-history proxy

Twenty fixture entries, fifteen recorded turns each: 300 requests, 277 associated with historical tool calls and 23 text turns. Four workers process sessions serially. Context cap: 60,000 characters with the middle dropped when necessary; per-message truncation and flattened tool history remain. The twenty distinct context sequences use only nineteen truncated session labels because two labels collide. Do not group the old records by that label as if it were a unique session key.

| v14 pass | Completion tok/s / wall | Prefix-cache token hit ratio | TTFT median / p90 |
|---|---:|---:|---:|
| a: first measured pass after startup | **232.6** | **71.3%** | **0.45 / 0.77 s** |
| b: identical recorded requests repeated | **275.2** | **94.3%** | **0.32 / 0.47 s** |

Pass a follows a short warmup request; there was no explicit cache-reset receipt. Call it the first measured pass after startup, not a formally proven empty-cache condition. In pass a, median TTFT by turn index was 0.34 s at turn 0 and 0.56 s at turn 14; mean prompt length at turn 14 was 13,049 tokens. Accepted tokens per speculative step were 3.46 in both passes. These are descriptive observations, not proof of why acceptance differed from the shuffled corpus.

### Estimator mismatch

`agg_tok_s = sum(completion_tokens) / measured_makespan` includes idle worker tails. The original `agg_ci95_boot` resamples individual requests and computes `workers * sum(tokens) / sum(request_seconds)` instead. For pass a, that estimator centers at approximately 242.04 tok/s and reports 233.6–250.7, while the wall rate is 232.6. For pass b, it centers at 285.27 with 276.0–294.9, while the wall rate is 275.2. Do not attach either interval, or its half-width, to the wall rate.

A replacement instrument needs unique session IDs, strict capture validation, explicit cache-state receipts and independent repeated runs for wall-clock uncertainty. Session-level summaries must treat related turns as clustered observations.

## 3. OFFGB=55: failed 1M KV requirement, not an exact floor

The full terminal exception is in `campaign-round6-2026-09-15.log`, not merely the abbreviated `facts-R6-off55.txt`:

> To serve at least one request with the model's max seq len (1048576), 2.2 GiB KV cache is needed, which is larger than the available KV cache memory (0.65 GiB). Based on the available memory, the estimated maximum model length is 123264. Try increasing gpu_memory_utilization or decreasing max_model_len.

Two allocator OOM warnings also appear before this terminal error. The final failure was the minimum-KV validation at 1M context. OFFGB=60 has approximately 4.87 GiB KV in the reference configuration. The tested bracket is **55 failed at 1M; 60 works at 1M**. Intermediate offload flags remain untested. A shorter context may reduce required KV and allow a lower-offload configuration to bind, but does not automatically relocate experts; actual offloaded tensors, free memory, speed and stability require measurement.

## Files and reproducibility limits

- `replay2-R6-{v14,v13}-{a,b}-n4.json`: shuffled-proxy captures, including per-request timing/count records.
- `replays-R6-sess-v14-{a,b}-n4.json`: ordered-proxy captures; original session labels are not unique.
- `facts-R6-off55.txt`: abbreviated failure excerpt; use the full campaign log for the terminal cause.
- `campaign-round6-2026-09-15.log`, `campaign-round6b-2026-09-15.log`, `campaign_round6.sh`, `run_sessions.sh`: original orchestration receipts.
- `replay2.py`, `replay_sessions.py`, `build_fixture_v2.py`, `build_fixture_sessions.py`: original instruments, preserved as provenance rather than endorsed corrected tooling.

Private transcript fixtures are deliberately not published. These captures are inspectable timing evidence, not a publicly reproducible exact-corpus benchmark. Builders are historical scripts, not reviewed credential-sanitization or production-agent adapters.

# Online-updated DSpark verify-cost curve on the offloaded lane — 2026-09-14

Follow-up to `../2026-09-12-v12-1M-k5-off60-util97/ksweep/` (adaptive verification missed the bar; one stated
reason was that its cost curves are profiled on dummy batches at boot and never see expert streaming) and to
the discussion on vllm-project/vllm#38256. Experiment: replace the boot-profiled `verify_ms[num_tokens]` table
with a running median of *real* verify-step forward times and see what the adaptive controller does with it.

**Verdict: mechanism confirmed, benefit small and mixed. Not a promotion.** The boot table underprices real
verify steps by 11–40% at decode sizes; with the corrected table the controller trims slightly harder.
C1 +3.3% and C8 +1.7% (two same-window pairs), but **C2 −5.9% and C4 −8.3%**, C16 wash; tool-JSON +15.5% is
the only category that moves in both pairs. v12 static k=5 remains the reference; adaptive remains off.

## Setup

Identical to the adaptive run on 2026-09-12: DeepSeek-V4.1-Flash MXFP4 `df42c109`, image
`vllm/vllm-openai:deepseekv41-flash-0909` (dsv41-feat e47aa78), `--cpu-offload-gb 66 --gpu-memory-utilization 0.97`,
UVA expert offload + Engram host, 1M ctx, 16 seqs, `VLLM_USE_V2_MODEL_RUNNER=1`,
`--speculative-config {"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}`.
Control = that exact container restarted. Candidate = same launch + two files bind-mounted over site-packages
(`patch/`) + `VLLM_DSPARK_ONLINE_VERIFY_CURVE=1`. Both boots hit the cached FlashInfer autotune (3–5 min binds).
Knee = `knee.sh` (192-token prose, 2 runs per concurrency); fixture = `agent_fixture.sh`. Candidate ran one
discarded fixture pass first so the online curve had ~400 real verify steps before measurement.

## The patch (`patch/`, env-gated, default off, TP1 only)

`OnlineVerifyCurve` in `adaptive_verification.py`: CUDA events recorded on the current stream around the real
forward of every verify-only step (drafts > 0, no prefill in the batch), resolved later with non-blocking
`event.query()` — no host sync added. Keyed by padded token count; rolling window of 32 → median; after every
32 landed observations the verify table is rebuilt from the boot curve overlaid with the observed points
(`build_cost_tables_from_curves`, unchanged). `model_runner.py`: two hook lines around the real forward.
Draft curve and controller logic untouched.

## Boot curve vs observed (candidate pair 1, 2,497 real verify steps; `curve-log-lines.txt`)

| padded tokens | boot ms | observed median ms | Δ | n |
|---|---|---|---|---|
| 2 | 12.51 | 13.83 | +10.6% | 506 |
| 4 | 19.38 | 19.16 | −1.1% | 561 |
| 6 (C1, k=5) | 21.49 | 24.18 | +12.5% | 540 |
| 8 | 21.94 | 29.23 | +33.2% | 122 |
| 12 | 25.19 | 35.40 | +40.5% | 123 |
| 16 | 32.46 | 41.80 | +28.8% | 205 |
| 24 | 48.04 | 53.65 | +11.7% | 249 |
| 32 | 51.31 | 62.33 | +21.5% | 98 |

The boot dummy batches *do* stream experts (the boot curve is not flat), so "the profile does not see expert
streaming" as written on 2026-09-12 was an overstatement — it under-prices it. Marginal cost per verify token
at C1 goes from ~2.2 ms (boot) to ~2.6 ms (observed).

## Results — two same-window pairs (control → candidate, ~17 min each)

| tok/s | ctrl 1 | cand 1 | ctrl 2 | cand 2 | mean Δ |
|---|---|---|---|---|---|
| C1 | 101.5 | 107.0 | 101.9 | 103.2 | **+3.3%** |
| C2 | 133.8 | 128.8 | 140.4 | 129.1 | **−5.9%** |
| C4 | 203.1 | 188.3 | 212.0 | 192.1 | **−8.3%** |
| C8 | 294.7 | 299.6 | 295.6 | 300.6 | +1.7% |
| C12 | 333.9 | 355.1 | 338.7 | 352.5 | +5.2% |
| C16 | 407.4 | 410.6 | 411.9 | 399.7 | −1.1% |

| fixture tok/s (accept) | ctrl 1 | cand 1 | ctrl 2 | cand 2 | mean Δ |
|---|---|---|---|---|---|
| prose | 103.6 (0.18) | 104.5 (0.19) | 103.3 (0.19) | 104.7 (0.19) | +1.1% |
| shell_ops | 144.1 (0.81) | 143.9 (0.83) | 145.5 (0.81) | 144.1 (0.83) | −0.6% |
| code | 143.8 (0.57) | 137.0 (0.51) | 139.4 (0.55) | 141.8 (0.54) | −1.6% |
| tool_json | 138.3 (0.84) | 163.2 (0.84) | 146.0 (0.84) | 165.2 (0.86) | **+15.5%** |
| structured | 123.6 (0.42) | 124.2 (0.43) | 125.0 (0.43) | 124.4 (0.42) | 0.0% |
| weighted accept | 0.519 | 0.502 | 0.516 | 0.515 | |

**Read.** The controller behaves as the corrected table says it should — accepted drafts/step on prose rises
0.90 → 0.93 at the same accept rate (fewer wasted drafts). But the gain at C1 is inside the day-to-day drift band
for this lane (~3%) and the C2/C4 loss is larger than the C1 win. The C2/C4 knee runs are high-variance on
both arms (111–146 tok/s within a pair), so the loss is not established either — but it is the sign in both
pairs. tool_json (+15.5% both pairs, 318 tokens) is the one consistent mover and would need a longer run.

**Why so small.** The verify curve was one of two things missing from the cost model. The controller now knows
a draft token costs ~2.6 ms rather than 2.2 ms, which nudges the argmax. What it still does not know is that
extra draft tokens pull *extra unique experts* through the host link (3.7× at k=5, `../routing/`). A per-step
unique-expert term — the `union_peak` counter yasinyaman describes in #38256 — is the term that would let the
controller price a draft by its expert bytes rather than by its token count. That is the next lever; this one
was the cheap 40-line check.

## Files

- `knee-*-p{1,2}.json`, `agentfix-*-p{1,2}.json` — raw, per pair
- `curve-facts.txt`, `curve-log-lines.txt` — boot vs online curve tables as logged by the candidate
- `campaign.log` — full same-window runner log; `campaign_onlinevc.sh` — the runner (paths scrubbed)
- `patch/apply_patch.py` (applies to the two files from the 0909 image), `patch/*.diff`

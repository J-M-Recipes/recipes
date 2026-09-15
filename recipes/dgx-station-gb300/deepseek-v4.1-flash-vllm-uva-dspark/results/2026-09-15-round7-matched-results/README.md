# Round 7: matched replay and bounded tool execution

**Verdict:** v14's replay-throughput advantage over v13 persisted across all three alternating boot pairs. Keep the existing v14 reference for this tested four-worker recorded-history workload; this result does not establish general production capacity or broad quality equivalence. No configuration, routing, model, context, or offload change was made.

## Results

Values below are arithmetic means across three boots per profile. Pair deltas are descriptive; there is no significance test or turn-level bootstrap.

| Measurement | v14 | v13 | Mean paired v14 throughput gain |
|---|---:|---:|---:|
| Initial-namespace replay | 231.7 tok/s (230.2–234.0) | 196.2 tok/s (195.4–197.1) | +18.1% |
| Identical-request repeat | 260.4 tok/s (258.3–263.3) | 217.8 tok/s (216.8–218.9) | +19.5% |
| Twelve-task suite wall time | 11.78 s (11.59–11.93) | 14.79 s (14.02–15.30) | Not a token-throughput metric |
| Task successes | 36/36 attempts | 36/36 attempts | Same 12 unique tasks, repeated three times |

The suite wall-time reduction is **20.3%**, calculated from the two profile means. All three individual suite comparisons favored v14; this is a small deterministic fixture, not a broad agent benchmark.

| Boot pair | Initial replay gain | Repeat replay gain |
|---|---:|---:|
| 1 (v14 → v13) | +18.7% | +20.3% |
| 2 (v13 → v14) | +17.5% | +18.6% |
| 3 (v14 → v13) | +18.2% | +19.7% |

**Latency tradeoff:** v14's per-run median replay TTFT was 0.356–0.357 s initially and 0.224–0.226 s on repeats. v13 was quicker at 0.326–0.330 s and 0.187–0.191 s. Higher aggregate output rate does not imply lower first-token latency. Per-run p90/p95 and gauntlet request TTFT tails are in the JSON evidence.

## Frozen experiment

- Same preserved containers, checkpoint and image; only batch-size speculative schedule differed: v14 `[[1,4,5],[5,16,1]]`, v13 `[[1,2,5],[3,16,1]]`.
- Six boots: v14/v13, v13/v14, v14/v13. Per boot: 30 non-scored warmup requests, 300-turn initial replay, the same 300 requests again, then the twelve-task closed-loop suite. Both references stopped after each boot.
- Replay: 20 context sequences × 15 serial turns, four workers, temperature 0, seed 42, reasoning `low`, 400 output tokens maximum. An initial namespace still allows within-sequence prefix reuse; it is not a zero-cache run. Token cache-hit ratios were 72.5% initially and 92.3% on repeats, identical across profiles.
- Suite: four file chains, four code repairs, four structured-answer tasks, four workers, maximum 12 model turns / 1,500 output tokens per turn / 180 seconds per task. Tools execute and their results feed subsequent model turns.
- Source: [frozen instruments](../2026-09-15-round7-instruments/), reviewed head `974a299bbf3fb5cc9560f5bcd67da55061341faf`, source-manifest SHA256 `1a52c4270cb41063736a7e22e0369ab9e9d25886388c823ea5e16d093b20bb26`. Merged source commit `d92a433`.
- Image: `vllm/vllm-openai:deepseekv41-flash-0909`, immutable digest `sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d`.

## Audit and limitations

All **12 replay runs / 3,600 captures** and **6 gauntlet runs / 72 attempts** are present. Exact request hashes match within each pair and between initial/repeat phases. Raw SSE usage and finish reasons match the capture rows; token sums reconcile with recorded server metric deltas. Timestamps, serial ordering, finite values, coverage, per-run TTFT quantiles, release hashes, container identities, stop proofs and sandbox isolation receipts passed checks. Server access counts exactly match 600 replay + 30 warmup + 46 or 47 gauntlet requests per boot; all logged completion peers were loopback, with no extra logged completions. Recorded metric deltas, rather than original before/after Prometheus snapshots, are available.

The published source analyzer reproduces the profile means and ranges. The final task answers were regraded with the frozen grader, including first-successful-access chain semantics; recorded code-oracle receipts show every expected case completed. This audit did not rerun generated code after collection. No final schema, protocol, correctness, capture, or budget failures were observed. Intermediate failed tool evaluations were retained: three for v14, four for v13; these tasks recovered within the original budgets without score retries.

**Outputs were not bit-identical.** Excluding tool-call IDs, 279–290 of 300 paired response records differed per phase/pair (content, reasoning and tool-call payload comparison); this is not a semantic-quality score. Each profile hit the replay length cap on 199/1,800 responses. No gauntlet response hit its output cap. Reported token rate includes actual varying-length outputs, not a fixed identical generated-token workload.

The replay uses private flattened/truncated recorded history, not native Hermes conversations, and does not execute replayed tools or carry generated answers into the next request. The exact-repeat phase is not a newly continuing production session. Reasoning configuration differs from Round 6, so cross-round absolute numbers do not isolate an optimization. Twelve repeated public tasks cannot establish broad model quality or a general reliability rate. No automatic promotion is authorized by the captures.

Collection finished September 15, 2026 at 12:24:31 CDT. Independent shutdown verification confirmed both exact references preserved and stopped, port 30006 dark, and successful controller exit. The independent stop timer was disarmed only after that proof. No other incumbent was restored.

## Public evidence

- `throughput.csv`: allowlisted per-run replay values.
- `audit.json`: derived per-run/per-task aggregates, descriptive paired differences, source/capture hashes. No private replay content or SSE bytes.
- `traffic-and-variation.json`: allowlisted access counts and output-difference counts.
- `gauntlet-timing.json`: actual request-level TTFT tails and response counts.
- `audit_completed_campaign.py`: read-only post-run auditor. Set the private-results and frozen-release directory arguments on the collecting host; it does not launch inference or re-execute generated code.

Private history, replay completions and base64 captures are deliberately withheld. Source and derived measurements are public; the private replay is **not independently reproducible from this public bundle alone**. The synthetic gauntlet fixtures and evaluator are public in the linked instrument bundle.

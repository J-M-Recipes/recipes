# 2026-09-21 — v20 "Clean Nightly" promotion: bisect resolved, same-window pair, held-out sibling, cost

Three cards in one morning window on `:30006`, all with the v15 hook, `--cpu-offload-gb 54`, `--kv-cache-dtype fp8_ds_mla`.

## Card E1/E2 — the bisect finishes: #56633

| nightly | adds (vs previous row) | agent-tool top-1 flips | held-out | accept tool_json / shell / prose | C1 |
|---|---|--:|--:|---|--:|
| `2671fedf` 09-13 (r1) | model support only | 1.77% | 0.00% | 0.860 / 0.908 / 0.349 | 189 live / 180 loaded |
| **`dc36fcce` 09-14 (r4)** | #56512 Engram prefetch, #56682, #56464 DeepSelect (opt-in; `indexer_topk auto` unchanged) | **1.77%** | **0.00%** | 0.860 / 0.908 / 0.349 | 174 |
| **`cd10ed6f` 09-15 (r5)** | **#56633 fold mHC post into delayed pre projection**, #56741, #56683 | **12.66%** | **9.00%** | 0.878 / **0.829** / 0.319 | 186 |
| `af1c0149` 09-16 (r2) | #56962 Mega-mHC | 11.73% | 8.72% | **0.757 / 0.760 / 0.260** | 194 |

r4 is bit-identical to r1 on the 79,544-position teacher-forced corpus, so Engram prefetch and DeepSelect are exonerated. The drift enters with the mHC fold (one PR, same box, same weights, T=0 logprobs against the no-hook v14 reference), and Mega-mHC on top is what costs the DSpark acceptance on tool-JSON. Both rungs booted with a live FlashInfer tune (23–25 min) — no pinned set exists for their MoE shapes.

## Card E3 — the v20 window (r1 → v18 → r1, loaded autotune set `a34f9ad4`, 189 configs)

| | v20a (r1) | v18 control | v20b (r1) |
|---|---|---|---|
| C1 prose (knee ×2) | 180.9 (180.8 / 181.0) | 171.8 (171.6 / 171.9) | 180.3 (180.2 / 180.4) |
| C8 / C16 | 670 / 979 | 660 / 952 | 661 / 965 |
| fund C16 warm agent p50 / p95 | 0.54 / 1.61 s | 0.54 / 1.83 s | 0.55 / 1.83 s |
| fund C24 warm agent p50 / p95 | 0.56 / 2.83 s | 0.57 / 3.36 s | 0.59 / 2.42 s |
| fund C16 / C24 aggregate | 299 / 304 | 300 / 281 | 287 / 278 |
| cold 120K under 16 streams p50 | 8.41 s | 8.42 s | 8.61 s |
| tools C16 × 4 turns | 64/64 | 64/64 | 64/64 |

C1 +5.3% / +4.9% (pair mean +5.1%) against the ≥5% bar — met on the mean, not on boot b alone; recorded as such. C24 aggregate is noisy at ±8% window-to-window and is not a gate.

## Card F — held-out sibling and cost per solved task (first fill of the agent claim card)

Protocol: `harness/protocol.yaml` (T=0, no thinking, `tool_choice auto`, 1 retry, C8). Held-out = BFCL v4 `live_simple` (258) + `live_multiple` (1,053), frozen this morning and run once, here.

| | held-out 1,311 | live_simple | live_multiple | dev 600 (same boot) | wall | mean W | $ / 1000 solved @ $0.15/kWh |
|---|---|---|---|---|---|---|---|
| v20 (r1) | **1033 (78.8%)** | 82.2% | 78.0% | 556 (92.7%) | 392 s | 490 (pre 495 / post 484) | $0.008 |
| v18 | **1035 (78.9%)** | 81.0% | 78.4% | 558 (93.0%) | 378 s | ~486 *estimate* | ~$0.007 |

Δ −0.1 pt on held-out against the ≥ v18−1 rule: pass. The v18 W is an estimate — its first Card F leg died at engine startup (`wait_for_engine_startup` RuntimeError; rebound cleanly on retry), and the rerun's cost printer had a shell-quoting bug, so its BFCL rows are real (from the summary JSON) but the power sample was not taken; 486 W is the mean of the v20 range and is labelled on the card. The two-sample pre/post W mean is the known weakness of this cost number; a sampled trace is the next improvement.

## Verdict

**Promoted: v20 "Clean Nightly" = nightly `2671fedf` + v15 hook + off54 + fp8_ds_mla + v18's 24-seat / cudagraph flags, 2026-09-21 12:30 CDT.** v18 Many Seat is the rollback container, stopped-and-kept. The promotion is on a nightly that is eight days old, chosen because it is the last before the mHC fold; each later nightly inherits #56633, so upstream must move before this recipe can track nightlies again.

Receipts: `receipts/` — `VERDICT-cardE.txt`, `VERDICT-cardF.txt`, `cost.txt`, TF compare JSONs for r4/r5, agent fixtures, knee logs, fund harness logs, BFCL summaries, the r1 autotune-set sha256.

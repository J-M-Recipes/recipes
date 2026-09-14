# 2026-09-13 — MTP K=2 on the 256K profile: self-repeat + frozen quality ladder

**Verdict:** K=2 is deterministic, passes the frozen task-quality gate against K=1 (0 losses / 1 win / 99 ties, Clopper-Pearson 95% upper bound on losses 0.0295 < 0.05), and is a few percent faster — more on code than prose. It is **not** promoted to daily because it does not clear the v1 contract's secondary-gate and +10% short-C1 speed bars (neither does K=1; see limits). K1-256K stays daily. The September 9 "K=2 fails greedy 9/20" reading is corrected: that number is reproducible and is not a quality signal.

## What ran (one box, same image, one axis changed)

| Lane | Container | Config |
|---|---|---|
| control `k1-256k` | `glm53-big-sc13g-mtp-ctx256k-DAILY-20260913` | 262,144 ctx, 24 GiB bf16 KV, 7,360 slots, MTP `num_speculative_tokens=1` |
| candidate `k2-256k` | `glm53-big-sc13g-mtp2-ctx256k-K2-selfrepeat-20260913` | identical, `num_speculative_tokens=2` |

Image `vllm-glm53-uva:v0.28.0-2cf0a691` (digest `61fc8a89…`), model `glm-5.3-big` (GLM-5.3 NVFP4 full), `max-num-seqs=1`, eager compile mode 3, slot cache router `ffi`, no other process on the GPU. Lanes ran sequentially, each container alone.

## 1. Self-repeat (`selfrepeat/`)

Same 20 greedy prompts, three passes in-lane, then compared across lanes with `scripts/greedy_equiv.py`.

| Comparison | Result |
|---|---|
| K2 r1 vs r2, r1 vs r3 | **20/20, 20/20** identical |
| K1-256K r1 vs r2, r1 vs r3 (from `2026-09-13-context-slot-curve`) | 20/20, 20/20 |
| **K2 vs K1 (same base)** | **9/20** identical — 11 prompts diverge, first divergence at char 65–762 |

The 9/20 exactly reproduces the September 9 K=2 number on a different slot profile. Because K2 is bit-stable against itself, this is not kernel nondeterminism (the vLLM #54945 fused-finalize class is ruled out for this lane). All 11 divergences are late near-tie token flips ("comes from" vs "is believed to derive from", "trivial" vs "tiny", table rule length). Read: K=2 verifies 3-token blocks instead of 2, which changes GEMM shapes and therefore float reduction order in the NVFP4 path. Deterministic, but not bit-equivalent to K=1. The greedy-identical gate measures kernel-shape equivalence, not task quality, and is the wrong instrument for a K change.

512-token probe (4 prompts × 2 runs, `probe-512-{a,b}.json`): decode median **53.68 tok/s** (n=8; prose 49.5–53.0, code 54.4–56.1); accepted length 2.05–2.28 prose, 2.67–2.80 code (max 3). K1-256K in the same day's window: 51.29 median, accepted 1.84 (max 2). Slot-cache counters during the K2 probe: hit 0.455 mean, 4.36 misses/step/layer over 14 windows (vs 0.61 / 3.12 for K1-256K) — more routes per step, as expected.

## 2. Frozen quality ladder (`quality-ladder/`)

Same harness, fixtures (`primary_100_seed20260906.json`, 50 HumanEval + 50 GSM8K, 2 repeats), sandbox scorer, secondary fixtures, speed probe, and acceptance rules as the September 6 campaign (`CONTRACT.md`). Only `run_lane.py`'s lane table and image digest changed; `frozen-files.json` regenerated for that one file (`CONTRACT-ADDENDUM-20260913.md`). No prompt or extractor edits. No retries.

| | K1-256K control | K2-256K candidate | Rule | Verdict |
|---|---|---|---|---|
| HumanEval | 95/100 | **96/100** | ≥80% | pass |
| GSM8K | 95/100 | 95/100 | ≥80% | pass |
| Overall | 190/200 | **191/200** | ≥85% | pass |
| Per-task loss / win / tie (N=100) | — | **0 / 1 / 99** | CP95 upper on loss <0.05 | **0.0295 PASS** |
| Invalid / protocol errors | 0 / 0 | 0 / 0 | none | pass |
| Secondary grounded | 20/20 | 19/20 | all pass | miss (grounded-1, repeat 0 only) |
| Secondary structured | 9/20 | 9/20 | all pass | fail — same 7 fixtures as every lane since Sept 6 |
| Secondary tools (20-hop) | 4/4 | 4/4 | all pass | pass |
| Ladder speed short C1 (n=7) | 50.14 tok/s | 51.41 | ≥+10% | +2.5%, not met |
| Ladder speed long C1 (n=7) | 45.18 / 11.33 s | 48.62 / 10.53 s | no >10% wall regression | pass (+7.6%) |

The one differing task is HumanEval/116 (K1 0/2, K2 1/2). September 6 reference: sc13g no-MTP 190/200, sc13g-MTP(1) 190/200 — same instrument, same score band.

## Limits (read before quoting)

- **Secondary gate has never passed for any lane.** Structured-output fixtures score 9–12/20 across V1, sc13g, sc13g-MTP, K1-256K and K2-256K. That is a model/template problem, not a K axis; it blocks *promotion* under the v1 contract, it does not indicate K2 harm.
- **K2's speed gain is workload-dependent**: ~+2–3% on short prose at `reasoning_effort=low`, ~+7–8% on code and longer generations. It does not meet the +10% short-C1 bar the v1 contract set for MTP(1) vs no-MTP.
- One grounded miss on K2 is 1 of 40 secondary outputs, on one repeat of one fixture; it is inside repeat noise but it is a 19/20 and is reported as such.
- Public HumanEval/GSM8K may be training-contaminated; this is a bounded recipe-selection regression test, not model quality.
- Slot-cache hit/miss are raw 20 s window counters, not campaign-valid instrumentation.

## Files

- `selfrepeat/k2-256k-selfrepeat/` — greedy r1–r3, `selfrepeat-1v2.txt`, `selfrepeat-1v3.txt`, `probe-512-{a,b}.json`, `slotcache-stats.txt`, `window.log`, `nvidia-smi.txt`
- `selfrepeat/crosslane-k1-256k-vs-k2-256k.txt` — per-prompt divergence offsets
- `quality-ladder/` — `CONTRACT.md` (frozen v1), `CONTRACT-ADDENDUM-20260913.md`, `frozen-files.json`, `run_lane.py`, lane logs, `results/{lane}-{config,primary,secondary,speed}.*`
- `SUMMARY.json` — every number above, computed from the JSONL by script

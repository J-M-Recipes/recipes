# Offline cache-policy and prefetch screen — STOP

Date: 2026-09-07 CDT  
Live service impact: **none** — this was replayed on the M4 from a frozen trace copy

## Verdict

Both proposed branches failed their predeclared offline continue gates by large margins:

- **E2 cache policy/allocation: STOP.** The best candidate remained LRU with a slightly different 5,792-slot allocation. It improved held-out hit rate from **70.9736% to 70.9978%**, only **+0.0242 percentage points** against the required **+5.0 points**. The held-out tools segment improved only **+0.0142 points**. LFU, static placement, and all three static/LRU hybrids were worse than current LRU.
- **E3-A trace predictor: STOP.** The best arm was adjacent-layer prediction with one issued row per layer: **8.4085% precision** and **3.6721% recall**, versus the frozen **50% precision** gate. Its held-out `tools_low` precision was **15.7106%**, still far below the gate. Previous-token prediction was worse and was scored only on unambiguous consecutive C1 decode steps.

Neither branch earns implementation or a live GB300 benchmark. The current 512K/MTP lane remains unchanged.

## Corpus and split

| Item | Value |
|---|---:|
| Routed tokens | 79,119 |
| Decode tokens | 71,210 |
| Prefill tokens excluded | 7,909 |
| Captured steps | 48,363 |
| Shape | 78 layers × top-8 × 256 experts |
| Split | first 70% train / final 30% held out, independently per segment |
| Held-out expert accesses scored by E2 | 12,819,000 |
| Slot budget | exactly 5,792, layers 3–77 |

Segments: `bench_and_gates`, `prose_low`, `code_low`, `tools_low`, `prose_max`, and `code_max`. The small tool segment still contributed 548 training and 235 held-out decode tokens.

The raw 99 MB trace is not duplicated in Git. `input-manifest.json` records source paths, byte sizes, and SHA-256 hashes; local hashes were checked against the GB300 copy before replay.

## E2 — held-out cache results

| Policy | Frozen allocation hit rate | Training-optimized allocation hit rate | Read |
|---|---:|---:|---|
| LRU | **70.9736%** | **70.9978%** | Best arm; only +0.0242 points |
| 25% static + 75% LRU | 69.2746% | 69.2658% | Worse |
| 50% static + 50% LRU | 66.8221% | 66.7221% | Worse |
| 75% static + 25% LRU | 62.3086% | 62.2710% | Worse |
| LFU | 53.0766% | 53.1317% | Much worse |
| Static top-frequency | 45.2851% | 45.0832% | Worst |

Optimized LRU moved 16-slot blocks across 22 layers while preserving exactly 5,792 total slots. Across 12,819,000 held-out accesses it avoided only **3,104** additional misses. That is **0.48% of the required E2 gate**, not a deployable effect.

The negative result is stronger than it first appears: static and hybrid arms were allowed to learn their hot sets from each workload's own training prefix. Even with that favorable adaptation, they lost to ordinary LRU.

## E3-A — held-out prediction results

Predictor transitions were trained separately from each workload segment's chronological prefix. Predictions exclude experts already resident and do not alter replay cache state; this screen measures whether a prediction has enough signal to justify a real pollution/overlap prototype. Adjacent-layer scoring covers all held-out decode rows; previous-token scoring excludes C4/C8 because the physical KV-slot sidecar is not a reliable request identifier.

| Predictor | Rows issued / layer | Precision | Recall of true misses |
|---|---:|---:|---:|
| Adjacent layer | 1 | **8.4085%** | 3.6721% |
| Adjacent layer | 2 | 7.8187% | 6.8292% |
| Adjacent layer | 4 | 6.8542% | 11.9735% |
| Adjacent layer | 8 | 5.6696% | 19.8082% |
| Previous token (C1 only) | 1 | **3.8026%** | 1.6312% |
| Previous token (C1 only) | 2 | 3.7287% | 3.1990% |
| Previous token (C1 only) | 4 | 3.6029% | 6.1821% |
| Previous token (C1 only) | 8 | 3.4661% | 11.8950% |

Issuing more rows bought recall by wasting more bandwidth; precision monotonically declined. The best 8.41% result reached only **16.82% of the 50% gate**. This particular expert-ID transition model is not a viable prefetch signal.

This is not a universal verdict against learned MoE prefetch. The trace contains routed IDs, not router logits or hidden states. A future model that consumes richer on-device signals would be a different experiment. It should not be built merely to rescue this loss.

## What remains worth doing

1. **E0 graph-safe live telemetry** remains useful because the current 0.6166 hit figure is a planning estimate, while this replay measured 0.7097 on held-out historical traffic.
2. **E1 nsys decomposition** remains useful to locate the unexplained decode milliseconds and quantify bookkeeping versus MoE/attention/MTP time.
3. **E5 MTP K=2 acceptance** remains a cheap independent probe.

None of those is authorized or implied by this offline result. They require a separate live window because the 512K lane is currently serving.

## Live proof

At 2026-09-07 21:46 CDT, after the replay:

- running container: `glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907`;
- served model: `glm-5.3-big`;
- `max_model_len`: 524,288;
- completion smoke: exact `OFFLINE_WORK_OK`.

## Files

- `CONTRACT.md` — gates frozen before the result existed
- `offline-sim-v1.json` — complete machine-readable policy, allocation, segment, and predictor output
- `input-manifest.json` — source/input provenance and hashes
- `live-service-proof.json` — post-run service receipt
- `run-command.txt` — exact replay command and harness hash
- `verify-package.py` — rerunnable invariant and public-hygiene checker
- `validation.txt` — test and validation receipt
- `SHA256SUMS` — package checksums

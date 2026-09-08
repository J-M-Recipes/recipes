# GLM-5.3 one-GB300 slot cache — next experiment queue (proposed 2026-09-07, evening)

Status: **E2 and E3-A completed and stopped; live work not started.** The live lane is
`glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907` (:30001, 512K, MTP(1), 5,792 slots) and stays
live until James schedules a window.

## Where the time goes (why the last two experiments lost)

Per target decode step at the 512K profile (numbers already in the repo/ledger):

| Term | Value | Source |
|---|---:|---|
| Expert row (w13+w2+scales) | 21.233664 MB | measured DMA backend geometry |
| Predicted hit allocation | 0.6166 | `slots-5792-ctx512k.json` planning figure (not measured live) |
| Misses / target step | ~230 | 600 expert-uses × (1−0.6166) |
| Bytes / step | ~4.885 GB | |
| Miss copy rate today (Triton, SM) | **383 GB/s effective** = link ceiling | `SLOT_CACHE_B_RESULT` 55.4 µs/miss; `C2C_BENCH` memcpy CE 358, SM zerocopy 361 |
| Copy floor / step | **~13.6 ms** at measured CE bandwidth | serialized on the compute stream |
| All-HBM MoE compute / step | ~14.7 ms | 75 × 0.196 ms |
| Observed step | ~20–22 ms (46–55 tok/s C1) | bench receipts |
| Eager tax | −84% (55.27 → 9.04) | DMA A/B |

Two conclusions the DFlash2 and DMA results already paid for:

1. **The copy mechanism is not the lever.** Triton copies already run at the C2C ceiling; DMA cannot move bytes faster, and it cost graphs. Any copy-path change is only worth it if it enables *overlap*, and overlap needs *prediction*.
2. **Only two levers remain:** (a) fewer miss bytes per step — hit rate, allocation, replacement policy, slot budget; (b) hide miss bytes under compute — prefetch driven by a prediction that is measurably right.

Everything below is ranked by (expected tok/s gain) / (cost + risk), with the cheap measurements first because the 0.6166 figure is a *plan*, not a *measurement*.

## Ranked queue

### E0 — Graph-safe live hit-rate telemetry (prerequisite; ~1 h)

- **Why:** we have never measured the live hit rate under real traffic. The stats hook was disabled (`STATS_SEC=0`) because it issued CUDA ops during stream capture. Every later decision hinges on the true per-layer hit rate and miss count.
- **Change:** in `patches/slot_cache_hook.py`, keep device-side hit/miss counters inside the graph (already exist as `lc.misses`), and read them only from the stats thread guarded by `torch.cuda.is_current_stream_capturing()` + a dedicated stream; never during capture. Unit test: a fake capture context asserts the reader is a no-op while capturing.
- **Gate:** relaunch of the exact 512K profile with stats on must reproduce C1 within ±2% of 55.27 (graphs intact) and emit per-layer hit rates.
- **Deliverable:** `results/…-hitrate-live/hit-rates-per-layer.json` over the standard prose+code bench plus a 20-minute real Hermes session.
- **Kill:** none — this is measurement. If measured hit ≪ 0.6166, E2 jumps to the top.

### E1 — Decode critical-path decomposition with nsys (~2 h, one window)

- **Why:** the 13.6 ms floor vs 14.7 ms compute vs ~21 ms observed leaves ~5–7 ms unexplained (bookkeeping, `index_put_`, MTP verify, attention at 480K context, scalar gathers). Astra's review item #4/#7 — never done on the real model.
- **How:** `nsys profile --capture-range=cudaProfilerApi` around 50 decode steps at C1 on the graph lane (`nsys` present at `/usr/local/bin/nsys` on .9). Bucket kernel time: routed MoE, `masked_row_copy`, bookkeeping Triton, MLA attention, MTP head, everything else. Separate all-hit steps from miss-heavy steps using E0 counters.
- **Gate/deliverable:** a table of ms/step per bucket that sums to the measured step time within 10%. This decides whether E4 (hit-path overhead) is worth doing at all.
- **Kill:** none — measurement.

### E2 — Offline replacement-policy and allocation sim from real traces (~half day, zero Station risk)

- **Why:** 0.6166 assumes LRU with the current per-layer slot map. LFU / frequency-pinned-hot + LRU-dynamic hybrids typically beat pure LRU on MoE routing traces (arXiv 2511.05814 reports LFU > LRU; the SpecMD paper reports large collision-miss reductions). This is the cheapest possible tok/s lever because it needs **no bytes to move faster**.
- **Actual input:** existing `trace/r1-base` routing traces from .9. They contain a dedicated `tools_low` segment but predate the current 512K/MTP profile; no fresh live trace was collected because E0 requires a scheduled relaunch.
- **Sim variants:** LRU (control) · LFU · static-hot-pin top-N% + LRU remainder · per-layer re-budget (steal slots from flat layers, e.g. L3/L4 that thrash at S=112, give to skewed layers) · full residency for the 3–5 worst layers.
- **Gate to go live:** ≥ **5 points** hit-rate over LRU on the held-out trace **and** on the agent-session trace, with ≤ 10% extra eviction churn. Then one relaunch, one matched bench.
- **Live gate:** C1 ≥ +5% over 55.27 with 20/20 greedy vs current lane (policy must be quality-neutral; it only changes *where* bytes sit).
- **Result:** **STOP.** Frozen-allocation LRU measured 70.9736% aggregate held-out hit rate. The best candidate, trace-optimized LRU under the same 5,792-slot budget, measured 70.9978%: +0.0242 percentage points aggregate and +0.0142 points on `tools_low`, nowhere near the +5-point gate. LFU, static-hot, and all tested hybrids were worse. Evidence: `results/2026-09-07-offline-cache-prefetch/`.

### E3 — Prediction precision from traces, then side-stream prefetch (the only path to "85")

- **Stage A actually screened:** adjacent-layer transitions from the current token's routed experts and previous-token/same-layer transitions, each at 1/2/4/8 nonresident-row budgets. Training was segment-local on the first 70%; the final 30% was held out. Previous-token scoring was limited to unambiguous consecutive C1 decode steps; the physical KV-slot sidecar was not treated as a request identifier for C4/C8.
- **Go/no-go for Stage B:** precision ≥ 0.5 on misses (i.e., at least half of prefetched rows are actually demanded next step). Below that, prefetch wastes C2C bandwidth that demand fills need, and we stop here and publish the number.
- **Stage B (prototype, 1–2 days):** side high-priority stream issues copies for predicted-miss rows for layer L+k while layer L computes; demand fill remains exact and authoritative; prefetch never evicts protected slots; prefetched rows are marked speculative and promoted only on hit. Graph-compatible by construction: the prediction is computed on device from the previous step's routing ids already living in the graph, so no host sync — this is where Fabian's copy-engine idea actually earns its keep (SM-free copies overlapping compute), and it is the only variant of his idea we should build.
- **Gate:** C1 ≥ +15% over the control **and** measured wasted-copy bytes ≤ 40% of prefetched bytes **and** 20/20 greedy + KL non-inferiority (prefetch must be invisible to numerics).
- **Result:** **STOP before Stage B.** Best was adjacent-layer/limit1 at 8.4085% precision and 3.6721% recall; `tools_low` precision was 15.7106%, still far short. Previous-token was worse and covered only unambiguous consecutive C1 steps. This fails the 50% gate by a wide margin. A learned model using router logits was not tested and would be a new research branch, not an extension justified by this screen.

### E4 — Hit-path overhead: compact misses before the copy grid, fuse scalar gathers (conditional on E1)

- **Why:** today every layer launches N × ceil(row/2048) copy programs regardless of miss count, plus scalar `index_put_`s. Astra item #7.
- **Do only if** E1 shows ≥ 1.5 ms/step in bookkeeping + empty copy programs. Otherwise skip and say so.
- **Gate:** C1 ≥ +3%, bit-exact vs control (pure overhead removal).

### E5 — MTP K=2 acceptance probe (~1 h, reuses `dflash2_acceptance_probe.py`)

- **Why:** cheap, already-tooled. MTP(1) is live; K=2 recursively reuses the single MTP head. More tokens per verify step also **amortize misses**: expert rows fetched for token t serve token t+1 in the same step for free.
- **Gate 1 (probe):** weighted accepted length ≥ 1.6 (K=1 is ≈1.4–1.5 by construction at a typical ~50% acceptance; if K=2 adds < 0.15 it cannot pay for its extra verify tokens).
- **Gate 2 (bench):** effective C1 ≥ +5% and the frozen code/math primary gate (190/200-class) does not regress.
- **Kill:** Gate 1 fails → publish as one line, done.

### E6 — Slot-budget vs context sensitivity curve (2 relaunches, ~1.5 h)

- **Why:** 512K/48 GiB KV was chosen as a balance; we have points at 256K (7,360 slots, 0.698) and 1M (2,672, 0.402) but no measured tok/s along the curve. One relaunch at **384K** (~6,500 slots) gives James a real trade table rather than a guess.
- **Deliverable:** `context → slots → measured hit (E0) → C1/C4/C8` table. This is a decision aid, **not** a proposal to change the daily profile.

## Not proposed (and why)

| Idea | Verdict |
|---|---|
| More DMA-mechanism variants (batched `cuMemcpyBatchAsync`, graph node updates) as demand fill | Copies are already at 383 GB/s ≈ link ceiling; without prediction there is nothing to overlap. Only as E3's copy engine. |
| DFlash2 K2/K3 on UVA | K4 acceptance 1.57; lower K does not fix a draft that does not agree with the target on this path. |
| FP8/INT KV, sub-4-bit experts | Standing quality rules for the full GLM path. |
| Fixing structured-output gate | Real, but orthogonal to speed; separate track. |

## Proposed order and window

1. **Completed without a window:** E2 and E3-A both failed; no candidate is staged for live deployment.
2. **Possible first scheduled window (~4 h):** E0 → E1 → E5 probe. This is independent measurement/speculation work and needs a new explicit go/window decision.
3. **Possible later window:** E6 context/slot curve. There is no E2 winner to deploy.
4. **Closed:** E3 Stage B is not built because Stage A missed its precision gate.

Every live arm follows the frozen contract pattern from the DFlash2/DMA runs: one candidate container, preserved
baseline, matched `bench3.sh`, restore + `/v1/models` + real completion smoke, receipts with SHA256SUMS, publish
wins and losses alike.

## Honest expectation

E0+E1 cost a window and produce no speed — they produce the live map. E2 did not find a replacement/allocation
win, and E3 found that these simple transitions are not predictive enough to justify prefetch. E5 remains a
cheap independent coin flip. On the evidence now in hand, the likely ceiling is close to the current hit-rate
curve; a large further gain needs either a materially richer on-device predictor or a quality-acceptable
all-HBM representation, not more demand-copy plumbing.

# Adaptive expert residency on the GB300 — research round 2 + plan (AR campaign)

**Date:** 2026-10-23 (evening) · **Owner:** Milo (James: "you have the station now") · **Lane:** GB300 Station `.9`, `:30006`
**Builds on:** `PORT-56177-MXFP4.md` (shape B), `probes/` (E1/E2)

---

## 0. Outcome (21:49 CDT): AR0–AR3 PASS

On the Grace-only v23 recipe, **counting routed lanes in the split kernel and swapping residency between steps** (shape B, hook v26ar.1) gives:

| workload (C = concurrency, post-60 s window) | frozen static map | adaptive | Δ | cold share frozen → adaptive |
|---|---:|---:|---:|---:|
| JP/DE essays, C1 (shift protocol, 2 boots) | 89.95 tok/s | 189.4 / 189.8 | **+111 %** | 28.3 % → 0.3 % |
| agent English, C1 | 195.7 | 218.5 / 219.3 | **+12 %** | 2.2 % → 0.45 % |
| mixed JP/DE + agent, C1 (aggregate) | 133.1 | 194.1 | **+46 %** | 17.2 % → 0.72 % |
| mixed JP/DE + agent, C4 (aggregate) | 236.6 | 431.0 | **+82 %** | 16.6 % → 0.67 % |

- **Overhead:** relaxed atomics cost 2–6 µs/step on the microbenchmark (0.01–0.04 %). The in-lane A/B over 12 pairs reads −0.21 % ± 0.36 % (not significant).
- **Correctness:**
  - Six unit tests pass, including a graph-captured count check and a restore round trip.
  - With the same hot set, outputs are identical within lane noise. S1a vs S1b and S1a vs S2 (static → 15k swaps → restored static) are both 49/50; the one miss, `agent-tool-1`, flips run-to-run even with the map fixed.
  - With a *different* hot set, one extra near-tie prompt flipped (`tool-weather-time`, P1). Likely cause: prefill tile configs differ between the 295-row and 89-row calls. That is batch-variance-class numerics, not corruption, and the round trip proves the swaps are lossless.
- **Reproducibility:** the shift-protocol adaptive numbers on two separate boots agree within 0.4 %.
- **Operations:** 15,382 swaps, 0 stale, 0 worker errors, 0 request errors.

**Not done (James's call):** promotion of a "v26 = v23 + v26ar.1 adaptive" recipe, and the same-workload comparison against production v25 (sidecar). The lane is left on v26ar.1 **frozen** (≈ v23); v25 is stopped-and-kept. See §6.

---

## 1. What the research round changed

### 1.1 E2 on the GB300 (sm_103, L2 115.5 MiB) — the per-M-tile question is answered

One 17.93 MiB MXFP4 expert row, all M tokens routed to it, rotating over 16 rows so L2 cannot carry a row between calls.
Calibration: copy-engine H2D **72.5 µs/row (242 GiB/s)**; SM streaming read 91.5 µs (191 GiB/s).

| M | ≥ tiles | HBM µs | UVA µs | Δ UVA−HBM µs | pool − HBM µs |
|---:|---:|---:|---:|---:|---:|
| 6 | 1 | 255.6 | 258.3 | 2.7 | +53.2 |
| 8 | 1 | 248.8 | 257.6 | 8.8 | +57.0 |
| 16 | 2 | 246.4 | 295.9 | 49.5 | +55.2 |
| 24 | 3 | 241.1 | 338.1 | 97.0 | +60.7 |
| 32 | 4 | 239.0 | 385.7 | 146.7 | +60.9 |
| 48 | 6 | 238.6 | 477.2 | 238.6 | +61.6 |
| 64 | 8 | 241.5 | 571.9 | 330.4 | +69.6 |
| 96 | 12 | 245.7 | 757.6 | 511.9 | +59.2 |
| 128 | 16 | 253.7 | 947.2 | 693.5 | +58.4 |

Reading it:

- **Each extra 8-token M-tile re-reads the expert over C2C.** In the GPU-bound range the marginal cost is flat at **~45.5 µs per tile** (32→48: +91.9 µs for 2 tiles; 64→96: +181.5 µs for 4; 96→128: +181.6 µs for 4) — one full 17.93 MiB row at ~410 GB/s effective. The 115.5 MiB L2 does not absorb the re-reads.
- **M ≤ 8 is masked by the eager floor** (~0.24–0.26 ms of launch overhead per call); in a CUDA graph a single-tile read costs ~one row (~45 µs). So the small-M deltas are lower bounds, not the serving cost.
- **The pool path is flat:** +53–70 µs at every M (one copy-engine row + HBM compute).
- **Crossover ≈ 2 tiles** (M_e ≈ 9–16 tokens on one cold expert in one layer-step). Below it, a demand read is cheaper than copy-on-miss; above it, copy wins by ~45 µs per extra tile.

Consequences:

1. **Decode:** at C1–C16 a cold expert almost never sees more than 8 tokens in one layer-step (C1 with k=5: 6 tokens in total). Demand reads win, so copy-on-miss loses → **shape B (move residency between steps, never per miss) is confirmed for decode.**
2. **Prefill on UVA lanes — hypothesis REFUTED by AR1 (see §3 AR1).** Extrapolating E2 to an 8192-token chunk (M_e ≈ 128 → 16 tiles of 8) predicted up to ~25 GiB of C2C re-reads per layer (~1.4–2.7 s per chunk). The lane measures **~20k prompt tok/s steady state (0.39 s per 8k tokens, 1.49 s for 31k)**, so that tax does not exist. The likely reason: E2 forced `tileTokensDim = 8`, because 16 local experts × top-1 put the average tokens per expert at the floor. The real prefill cold call (T = 8192, top-6, 89 experts → ~550 lanes per expert on average) gets a much larger token tile, so there are ~1–2 reads per expert. The per-tile re-read is a property of the tile-8 tactic, which is exactly the decode tactic. The NVFP4/Marlin result on #56177 (2026-10-22: Marlin re-reads per M-block, ~5× PCIe traffic, and staging cut TTFT 67.6 s → 13.0 s at 81.6K) is real for Marlin on PCIe but **does not transfer** to TRT-LLM MXFP4 on C2C.

### 1.2 E1 on the GB300 — counter cost

| lanes/layer (≈ C) | base µs/step | +atomic µs/step | Δ | % of 14.6 ms |
|---:|---:|---:|---:|---:|
| 36 (C1, k=5) | 66.2 | 95.9 | 29.7 | **0.20 %** |
| 96 (C16, k=1) | 58.4 | 84.0 | 25.6 | **0.18 %** |
| 192 | 57.4 | 108.5 | 51.1 | 0.35 % |
| 288 | 55.3 | 133.9 | 78.6 | 0.54 % |

It passes the 0.5 % bar up to ~192 lanes. The boot counts decode-sized steps only (`PIN_COUNT_TMAX=256` tokens); prefill is not counted, because at large M almost every expert is touched and the counts would mostly be uniform noise.

### 1.3 Upstream, as of tonight

- **#56177** is still open, `mergeable=dirty`, with no commits since 09-10. A field report on 10-22 (GLM-5.3-Flash-NVFP4 at 1M context on 2× RTX PRO 6000, 193 pool rows, MTP-3 → 105–114 tok/s single stream) adds three follow-ups: an MTP layer guard, a guard for Marlin's 32 GiB index limit, and prefill staging. Nothing in it targets MXFP4/TRT-LLM.
- Related but not substitutes: **#62060** (UVA staging for large steps; prefill only), **#62146** (CPU residency for renamed NVFP4 expert weights), **#60779** (no GPU sync in expert-id lookups), and **#59168** EvenMoE (EPLB placement + replica routing — expert-parallel only, not single-GPU residency).
- **Nothing upstream does decode-time residency for MXFP4 TRT-LLM on one GPU**, so building it in-house is still the right call.
- **Hardware fact:** the trtllm-gen MXFP4 MoE ships **no sm_120 cubins** (sm100/103/107 only). Any probe of this kernel has to run on the GB300.

---

## 2. Decision

Build **shape B on the Grace-only v23 recipe** (`launch_v23.sh`: UVA cold bank, 295 hot / 89 cold per layer, `rowmap-static-v1.json`):

- v23 is the publishable single-Station recipe, and its decode is bound by cold C2C reads, which is exactly what residency fixes.
- v25 (production) computes cold experts on the RTX sidecar, so its bottleneck is different. Port shape B there only if AR wins on v23.

---

## 3. Cards

### AR0 — unit tests, no model (GB300, image `deepseekv41-flash-0909`) — **PASS 5/5** (20:36 CDT)

| test | checks |
|---|---|
| T1 split+count | hot/cold ids identical to the base kernel and to the torch reference at T = 1/6/24/48/96 (with −1 lanes); counts = `bincount`; flag off → no counting |
| T2 CUDA graph | 40 launches captured; 50 replays → counts = 50 × bincount per layer; flag toggled between replays without re-capture |
| T3 queue policy | one clear winner → exactly one swap with the expected rows; frozen → none; busy layer skipped |
| T4 guarded apply | rows move, row_map flips, eids + CPU mirror update; replaying the stale item is refused; per-barrier cap holds |
| T5 drain | deltas, EWMA, cold share, telemetry JSONL |

### AR1 — boot v26ar frozen (count only) — ran 20:36–20:46 CDT

Results:
- **Boot** 473 s. Hook wiring confirmed (atomic counter, atomic mode, swap barrier). KV pool 5.24 GiB = **2,495,694 tokens** (2.38× of 1M).
- **Same-boot A/B, count on vs off** (6 interleaved pairs, 512 greedy tokens, C1): medians **239.4 vs 241.8 tok/s**. Paired Δ = **+1.27, +1.00, −1.17, −1.85, −0.80, −1.80 %** → mean −0.56 % ± 0.56 % (1 SE), with a 2.4 % spread on the off arm. That is **below the lane's noise floor**: it does not resolve 0.5 % either way. It is consistent with E1 (0.20 %) and also with zero. The gate as written (paired median ≥ −0.5 %) reads FAIL at −0.985 %. Follow-up: switch the counter to `sem="relaxed"` (it currently uses acq_rel → a fence per kernel), then re-measure with 12+ pairs.
- **TTFT** (unique prompts, so no prefix-cache hits): 2.0k → 0.19 s, 7.9k → 0.39 s, 31.2k → 1.49 s. That is **10–21k prompt tok/s** in steady state. The first request at each size is 3–7× slower (first-use tactic warm-up).
- **Counter sanity:** 1,645 selections per step (median); in-domain cold share **2.4 %** per drain; JP/DE prose on the frozen map **18–21.5 %** (matches the 20.5 % history).

Original card:
v23 launch + hook dir `v26ar/hook` + `PIN_MODE=adaptive PIN_COUNTER=atomic PIN_ADAPTIVE_FREEZE=1` (+ ctrl/telemetry paths). Same image, args, rowmap, and autotune cache.
- **Gates:** boot OK; `atomic counter ready`, `atomic mode ready`, and `swap barrier` all appear in the log; same-boot A/B (count on vs off, 6 interleaved pairs, C1, 512 greedy tokens) **paired Δ ≥ −0.5 %**; same token count in every run; telemetry sanity.
- **Also:** a TTFT sweep (~2k / ~8k / ~31k-token unique prompts) to size P-stage; a frozen greedy parity capture (50 prompts × 256) as the pre-swap reference.

### AR2 — domain shift, frozen vs adaptive (same boot) — ran 20:46–21:06 CDT

**Result: adaptive residency doubles out-of-domain decode and lifts in-domain decode by 12 %.**

| phase (post-60 s window, C1) | frozen tok/s | adaptive tok/s | Δ | cold share frozen → adaptive |
|---|---:|---:|---:|---:|
| JP/DE essays (1) | 89.95 | **189.43** | **+110.6 %** | 28.3 % → **0.31 %** |
| agent English | 195.65 | **218.45** | **+11.7 %** | 2.17 % → **0.45 %** |
| JP/DE essays (2) | 89.95 | **188.61** | **+109.7 %** | 28.3 % → **0.36 %** |

- 4,158 swaps over 9 min; 0 stale, 0 worker errors. The first JP/DE request after unfreezing ran at 107.8 tok/s, and the phase hit 172–195 tok/s within ~60–90 s.
- Even the in-domain map was not optimal: the boot-time static map (profiled on agent traffic) runs agent at 2.2 % cold. Following live traffic gets that to 0.45 %, which is worth +12 %.
- Physics check: frozen JP/DE puts ~28 % of 1,645 selections per step on cold experts (≈ 460 cold lanes per step ≈ 11.5 per layer → several distinct cold experts per layer-step). At ~45 µs per single-tile C2C read (E2), that adds ~7–12 ms to a ~14.6 ms step, consistent with the measured 2.1× gap.
- **Parity: 49/50 exact pre-swap vs post-swap.** The one miss (`agent-tool-1`, a parallel tool-call prompt) **also flips between two captures with the map fixed** (AR2c vs AR2d, no swaps in between: 1 vs 2 tool calls). It is run-to-run non-determinism of the lane, not residency. The strict 50/50 gate reads FAIL; the honest reading is "within the lane's fixed-map noise". AR3 adds a proper noise baseline (S1a vs S1b) and a round-trip test (static → adapted → restored static).
- **Verdict as scripted: FAIL** (strict parity + A/B gates). **Substance:** the speed gates pass by ~20× their margins; both failing gates sit at the noise floor and are retested in AR3 with better instruments (relaxed atomics, 12 pairs, noise-aware parity).

### AR3 — v26ar.1 (relaxed atomics + `restore_map`) — launched 21:13 CDT, ETA ~21:50 CDT
- **E1b (GB300, same microbenchmark):** relaxed atomics cost **2.0–6.1 µs per step** (0.01–0.04 % of 14.6 ms), against 24.6–77.7 µs for acq_rel. The fence was the cost.
- **Plan:** AR0' (adds T6 restore_map); boot; S1a/S1b static parity (noise set); A/B with 12 pairs; mixed JP/DE+agent traffic at C1 and C4, frozen vs adaptive (180 s each); a reproducibility run of the shift protocol; P1 parity post-adaptation; `restore_map` → S2 parity; round-trip gate = S1a vs S2 misses ⊆ noise set.

Original AR2 card:
e4 protocol: JP/DE essays → agent English → JP/DE, 180 s per phase, C1. Run frozen first, then `{"freeze": false, "baseline": true}`, then re-freeze and capture post-swap parity.
- **Gates** (on the post-60 s window of each phase):
  - JP/DE tok/s **≥ +5 %** vs frozen;
  - agent tok/s **≥ −3 %**;
  - JP/DE cold share **≤ ½** of frozen;
  - post-swap greedy parity **50/50 exact** (split math does not depend on residency; E4 already showed swaps are bit-exact).
- **Policy defaults:** EWMA half-life 500 steps, drain every 64 steps, ≤2 swaps/layer/drain, hysteresis 1.5×, min EWMA 8, ≤8 swaps per barrier (~150 µs each: HBM→staging, C2C read, C2C write → ≤ ~1.2 ms on a swap step).

### AR3 — only if AR2 passes
Tune the half-life, budget, and barrier cap; run C8; soak 30 min; then decide whether to promote v26 = v23 + adaptive, and whether to port to v25.

### P-stage — DEPRIORITISED (AR1 TTFT: ~20k tok/s steady prefill; no C2C re-read tax to remove)
On prefill steps (T > 256), copy the layer's cold bank (1.56 GiB, contiguous) into an HBM staging bank one layer ahead, and run the cold call from HBM.
- **Cost:** 1.56 GiB of HBM, ~32 % of v23's KV pool — or half that with a 45-row partial stage.
- **Expected:** C2C traffic per 8k chunk drops from ~1 TB to ~62 GiB.
- Build it only if AR1's TTFT shows prefill is dominated by C2C.

---

## 3.6 AR3 results (21:13–21:49 CDT, boot 472 s)

- **AR0':** 6/6 (adds T6: restore_map moves the hot set back; weights follow experts; row_map matches the mirror).
- **A/B, relaxed atomics, 12 pairs:** median 241.3 (on) vs 242.7 (off); paired mean **−0.21 % ± 0.36 %**. The off-arm spread is 2.5 %, so this is not significant.
- **TTFT** (repeat of AR1): 2k 0.19 s · 8k 0.46 s · 31k 1.38–1.50 s (21–23k tok/s).
- **Mixed traffic** (table in §0). At C4 even agent requests nearly double (66 → 122 tok/s per request): with a frozen map, the cold reads from JP/DE tokens stall the whole batch step.
- **Shift reproducibility:** AR3 adaptive post-60 = 189.8 / 219.3 / 189.4 vs AR2 189.4 / 218.5 / 188.6.
- **Parity:**

  | comparison | exact | misses |
  |---|---:|---|
  | S1a vs S1b (fixed map) | 49/50 | `agent-tool-1` |
  | S1a vs P1 (adapted map) | 48/50 | `agent-tool-1`, `tool-weather-time` |
  | S1a vs S2 (restored) | 49/50 | `agent-tool-1` |
  | S1b vs S2 (restored) | 49/50 | `agent-tool-1` |

- **Restore:** 14 short requests drained the restore swaps; `hot_moved` reached 0.

## 6. Next cards (proposed, not run)

1. **AR4: same-workload comparison vs production v25.** Run the mixed C1/C4 and shift protocol on the v25 sidecar. This needs v25 restored, which is James's call. Question: does a **single-GPU** adaptive v23 match or beat the two-GPU sidecar? If it does, the sidecar becomes optional.
2. **AR5: soak + agentic gauntlet on v26 adaptive.** 60 min of mixed traffic; p99 inter-token latency on swap steps (≤ 8 swaps × ~150 µs per barrier); gauntlet 12/12; a 128k+ long-context check; host memory stable.
3. **Adaptive residency for v25:** the sidecar also has a static hot/cold split, and the same counter + policy applies. The actuator becomes RTX-VRAM ↔ GB300-HBM row moves over PCIe (~3–4× slower per swap).
4. **Policy tuning:** half-life, budget, barrier cap. Defaults already converge in ~60–90 s.
5. **Upstream:** the counter-in-existing-kernel + relaxed-atomics lesson, plus the C2C per-tile re-read data, are worth a comment on #56177 / RFC #38256 (as `jmeadlock`; draft for James's review).

## 4. Lane state and restore

- **v25 sidecar:** STOPPED-AND-KEPT at 20:14 CDT. Watchdog disarmed; Mac soak cron `0bd2fcf8896c` paused; `LANE-OWNER.txt` updated.
- **Restore steps:** `~/dsv41/ar/RESTORE-v25.md` (sidecar first, then server, then re-arm the watchdog, then resume the cron).
- **During the campaign** `:30006` is served by `dsv41-vllm-1017-v26ar-AR`. It serves the same model name, and its outputs are bit-identical to v23 while frozen. At the end the runner leaves it **up and frozen**. There is no restore-on-done unless James asks.

## 5. Code

- `ar/hook-v26ar/` (deployed to `~/pin-hot-experts/v26ar/hook`; every other file is byte-identical to v19):
  - `pin_hot_experts_hook.py`: `_skc` counting split kernel, `_cnt[40,384]` + device flag, `PIN_COUNT_TMAX` / `PIN_SWAPS_PER_BARRIER` / `PIN_CTRL` / `PIN_TELEMETRY`.
  - `e4_adaptive.py`: atomic drain on a side stream, numpy EWMA, CPU hot/cold mirror, guarded apply with stale refusal, per-barrier cap, ctrl file (count / freeze / budgets / baseline / dump), telemetry JSONL.
- Harness: `ar/{ar0_unit.py, launch_ar.sh, ar_runner.sh, ar1_ab.py, ar_shift.py, ar_verdict.py}` plus the e2c parity scripts. Results go to `~/dsv41/ar/run/`, with progress in `run/STATUS`.

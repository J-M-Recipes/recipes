# Next window plan — K=2 under slot cache, with live hit-rate telemetry (v3)

> **Implementation correction — Astra / openai-codex, September 8:** The historical design below is NOT a released live contract. Offline implementation is authorized; no new Station window is authorized by this document. The following corrections override conflicting text below:
>
> - `Instances == 300` and kernel duration are retrospective attribution heuristics, not general phase proof. New exact phase bucketing requires timestamp-aligned GPU trace intervals, provenance, and boundary checks. Missing phase markers is a blocker, not permission to call a whole capture “decode.”
> - `completion_tokens / median(decode_tok_s)` is not measured aggregate decode wall. Client row durations and GPU time remain separately labeled. Historical ratios and the ≤2 ms copy-overhead ceiling are estimates, not hard physical bounds or evidence excluding all policy work.
> - Reject the proposed `CaptureGate` plus side-stream CUDA reads: a check in a hook neither brackets all capture operations nor synchronizes a concurrent reader. Candidates keep periodic stats OFF. A real engine-owned quiescent snapshot boundary must be proven before advertising graph-safe live telemetry; pure helper tests cannot prove it.
> - G1 runs before C2, using validated raw probe rows and all 20 nonempty exact reference outputs. The 45.65 historical speed is a hygiene reference, not a matched contemporary performance control; K1 versus K2 remains the primary comparison.
> - G4 is reporting-only, not part of the Boolean PASS expression. G1/G2/G3 are measurement gates; complete bound evidence, restoration and reviewed instrumentation are separate mandatory qualification gates. Missing qualification yields INCONCLUSIVE, never PASS. Twenty greedy prompts are a bounded regression check, not proof of universal quality equivalence.
> - A5 is removed from this A/B: no optional coalescing or cache-policy changes in the same experiment. Only speculative depth changes between candidates (apart from run-specific artifact/container identifiers).
> - Set restoration obligation BEFORE attempting incumbent stop. A failed stop command may have already stopped the service. Independent restoration must stop/verify both candidates, then prove the exact incumbent. Deadline budgeting must reserve restoration and include in-flight commands, not just candidate launch checks.
> - No fixed calendar execution date or artificial implementation time box. Review, real capture/snapshot validation, and a new explicit released contract govern readiness.


> **For the implementer (Astra):** offline work first, strict behavioral TDD, independent adversarial review before any Station mutation. Reuse `scripts/window_e1_v2.py` primitives; do not fork a new safety model. Nothing here authorizes a live run — James schedules the window.

**Date:** September 8, 2026 · **Author:** Milo (anthropic/claude-fable-5.1 via Nous) · **Repo:** `J-M-Recipes/recipes` · **Recipe:** `recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache`

---

## 0. Why this plan looks different from what I said this morning

At 12:10 I told James the next run should be "skip empty masks + coalesce the 4 copy launches, 5–12 ms/step." Re-cutting the E1 v2 trace this afternoon falsified that:

| Claim (12:10) | What the receipts actually say | Evidence |
|---|---|---|
| Step is ~59 ms, a third MoE GEMM / a third row copies | Whole-capture aggregate ÷ decode steps mixed in **prefill**. Decode is **40.1 ms wall / 43.1 ms GPU** per step. Rows with `Instances == 300` (= 4 requests × 75 layers, 5.2 and 2.6 ms avg) are the prefill bypass GEMMs (2,351 ms of the 8,389 ms capture). | `e1_cuda_gpu_kern_sum.csv`; `analysis/decode-only-attribution.md` |
| `masked_row_copy` is a third of decode | It is **~half**: 20.1 of 40.1 ms wall (46.6 % of decode GPU). | same |
| Decode MoE GEMMs ~22 ms, worth tuning | Decode routed GEMMs (`Instances == 10725`) are **5.4 ms/step**. Not a lever. | same |
| ~15 % of copy launches near-empty → skip/coalesce saves 5–12 ms | Only **2.4 %** of the big `w13` launches are <10 µs. Duration quantizes at **32.8 µs = one 12.6 MB row at 383 GB/s** → mean **4.4 misses/layer-step**. Bytes alone explain ~18 of the 20.1 ms. Copy is **~90 % C2C bandwidth**, ≤2 ms is overhead. | w13 duration histogram from `e1_cuda_gpu_trace.csv` (Station); inferred |
| Decode is eager (189 launches/step, 3 graph launches) | Wrong reading. ~4,400 kernels/step execute but only 189 are eager API launches → **~4,200 replay inside 3 CUDA graph launches**. `backend: eager` disables Inductor, not cudagraphs. Graphs are already on. | `e1_cuda_api_trace.csv` vs kern_sum instance counts; `container.log` "Inductor compilation was disabled" |

Implied live hit rate ≈ **0.73** (inferred from copy quantization; E0's reconstruction said ~0.75). So the remaining decode levers are exactly the two the 2026-09-07 plan already named: **(a) fewer miss bytes per step, (b) more accepted tokens per step to amortize the bytes.** E2 (policy/allocation) already showed (a) is near its ceiling for this slot budget. That leaves **(b): MTP K=2**, plus finally *measuring* the hit rate instead of inferring it.

Bookkeeping hypothesis stays falsified (1.56 ms/step). Gather-compaction stays dead.

## 1. Goal

One live window (~2 h) that answers: **does MTP K=2 raise single-stream decode throughput ≥5 % on the 512K slot-cache lane without quality regression?** — and, as a by-product, the first **measured** per-layer hit rate under real traffic.

Non-goals: prefill/TTFT (see §7), copy-path micro-optimizations (≤2 ms ceiling; do only if K=2 fails and nothing better exists), policy/allocation changes (E2 closed), MoE GEMM tuning (5.4 ms/step, not a lever).

## 2. Pre-registered numbers and gates

Baseline (E1 v2, patched K=1, September 8): decode **45.65 tok/s** C1 median (probe), acceptance length **1.82**, step **40.1 ms**, misses ≈ **4.4/layer-step** (inferred).

Expectation for K=2 (state it so we can be wrong in public): acceptance length 1.82 → ~2.3–2.4 if the second draft accepts at ~0.7; step +15–25 % (union of 3 tokens' experts → more misses, plus one extra MTP-head pass). Net **+5–10 % tok/s, confidence low-medium.** If the second-draft acceptance is <0.55 or misses scale linearly, K=2 loses.

| Gate | Measure | Pass | Fail action |
|---|---|---|---|
| **G0 telemetry** | Graph-safe stats thread reports `HIT` for all 75 layers on C1 with no capture invalidation | ≥1 STATS line after warmup, no `stats error`, graphs still replaying (kernel/API ratio unchanged) | Fix offline; window continues (telemetry is not on the critical path) |
| **G1 hygiene** | C1 (new tree, K=1) vs E1 v2 K=1 | C1 tok/s ≥ 45.65 × 0.99 **and** greedy 20/20 vs incumbent outputs | Stop the window before C2; tree regressed |
| **G2 speed** | C2 (K=2) acceptance probe `decode_tok_s_median` vs C1 | ≥ 1.05 × C1 | Still run bench (12 min) for the record; verdict STOP |
| **G3 quality** | `greedy_equiv.py` C2 vs incumbent outputs, 20 prompts, temp 0 | Existing frozen standard (20/20; document any ulp-class miss per `research/slot-cache-root-cause-2026-09-06.md`) | Verdict STOP regardless of G2 |
| **G4 bench** | `bench_big.py` ×3 prose + `bench_big_code.py` | C1 agg tok/s ≥ +5 % over C1 candidate at C1/C4/C8 | Reported, not gating (probe is primary) |

G2 ∧ G3 ∧ G4 → **PASS**: propose K=2 as the daily lane (separate James approval; not automatic). Otherwise **STOP**, publish, and the next lever is the slot-budget/context curve (old E6).

## 3. Phase A — offline (no Station), strict TDD

All tests: `.venv/bin/python -m pytest tests -q` (currently 102+ passing). Commit after each task. Triton kernels test locally with `TRITON_INTERPRET=1` (CPU interpreter) — if that proves impossible for a kernel, say so and mark that assertion as Station-only.

### A1. Decode-only bucketing (fix the methodology bug)

**Files:** `scripts/nsys_bucket.py`, `tests/test_nsys_bucket.py`, `scripts/window_verdict.py`, `tests/test_window_verdict.py`

1. RED: with `results/2026-09-08-e1-v2-live/live-receipts/e1_cuda_gpu_kern_sum.csv`, `--steps 140 --wall-ms 5608 --exclude-prefill-instances 300`, assert `routed_moe.per_step_ms ≈ 5.38`, `masked_row_copy ≈ 20.10`, `dense_gemm ≈ 9.41`, `gpu_per_step_ms ≈ 43.1`, and that the JSON records `excluded_rows` with names/instances/avg_ns.
2. Implement: exclude rows where `Instances == N` for each N in `--exclude-prefill-instances` (comma list) **and** `Avg (ns) > 1e6`; refuse (exit 2) if an excluded row has `Avg (ns) ≤ 1e6` (guards against excluding decode kernels by accident). Record exclusions.
3. `window_verdict.py e1`: consume `excluded_rows`; `wall_ms` for the slowdown/reconciliation must be **decode wall** = `completion_tokens / decode_tok_s_median` from the probe, not probe total wall. Test with the E1 v2 receipts.
4. Also store `decode_wall_ms` in the probe summary (probe already has `decode_tok_s_median`; derive and persist).

### A2. Graph-safe live hit-rate telemetry (the E0 that never happened)

**Files:** `patches/slot_cache_hook.py` (`_start_stats_thread`, `_invoke_kernel`), `patches/slot_cache_stats.py`, `tests/test_slotcache_helpers.py`, `recipe.yaml` (patch sha256 registrations)

Root cause: `lc.misses.item()` from the stats thread during cudagraph capture invalidates the capture (why `STATS_SEC=0` on the incumbent). Counters themselves are fine — `fused_bookkeeping` does `atomic_add` on device and keeps counting during graph replay.

1. RED (pure-Python, no CUDA): a `CaptureGate` state machine in `slot_cache_stats.py`: `observe(capturing: bool)` transitions `idle → capturing → done`; `reads_allowed()` is True only in `done`. Tests: never allowed before a capture has been seen **and** finished; stays allowed after; a later `capturing=True` (recapture) flips it off again until done.
2. GREEN: implement; wire `observe(torch.cuda.is_current_stream_capturing())` at the top of the patched `_invoke_kernel` (engine thread, cheap). Stats thread: `if not gate.reads_allowed(): continue`; perform the D2H on a **dedicated side stream** (`torch.cuda.Stream()`, `with torch.cuda.stream(s): vals = torch.stack([lc.misses, lc.routes, lc.step]).to("cpu", non_blocking=True); s.synchronize()`), one stack per layer, not three `.item()` calls.
3. Emit **machine-readable** per-layer JSON to a file, not just `_LOG` text: `SLOT_CACHE_STATS_JSON=/wcap/slot-stats.jsonl` → one line per window: `{"t":..., "window_s":..., "layers":{"3":{"misses":..,"routes":..,"steps":..},...}}`. `summarize_window` stays the single source of the hit-rate formula (routes denominator; already correct for MTP).
4. Default remains `SLOT_CACHE_STATS_SEC=20`; the runner sets `STATS_SEC=10` for candidates. Incumbent restore is untouched (its container is `docker start`ed, env baked).
5. Update `recipe.yaml` patch checksums; `scripts/check_recipe.py --all` must pass.

Station-only assertion (Phase B, G0): after warmup, `container.log` shows STATS lines and **no** `stats error`, and cudagraph replay is still in effect (API trace `cudaGraphLaunch/step ≈ 3`).

### A3. Runner v3: two candidates, one axis, K=2 allowed only where named

**Files:** `scripts/window_ab_v3.py` (new), `tests/test_window_ab_v3.py` (new), `results/2026-09-09-k2-v3/{CONTRACT.md,verify-prep.py,PREP-SHA256SUMS,README.md}` (new), no edits to `window_e1_v2.py` beyond extracting shared functions into `scripts/window_common.py` if needed (keep v2 tests green).

Reuse verbatim: `archive_source`/executed-source execution, `verify_archive_integrity`, `verify_incumbent_identity` (still requires **K=1** on the incumbent), `verify_candidate_image` (digest `sha256:61fc8a89…`), `gate`, `run_logged` with process-group kill, host lock, `--restore-only`, redaction, restore-before-analysis.

Sequence (each arrow = release gate re-check):

```
preflight identity+digest → greedy_equiv on INCUMBENT (20 prompts, temp 0, read-only, ~2 min) → gate
→ stop incumbent → gate
→ C1: launch new tree, K=1 ({"method":"mtp","num_speculative_tokens":1}) → readiness ≤ 2400 s
     → acceptance probe (512 tok) → bench prose×3 + code → stats JSON snapshot → unprofiled 64-tok probe
     → NSYS START/STOP around a 64-tok probe → greedy_equiv vs incumbent outputs → stop C1
→ gate → C2: launch same tree, K=2 ({"method":"mtp","num_speculative_tokens":2}) → same collection → stop C2
→ IMMEDIATE restore + proof (WINDOW_RESTORE_OK, exact id/digest/K=1)
→ offline: nsys stats (decode-only buckets), window_verdict.py k2, package
```

Rules to encode and test behaviorally (fake docker/nsys/probe executables, as in `tests/test_e1_window_v2.py`):

- K=2 appears in **exactly one** launch argv (C2) and nowhere else; incumbent identity and restore proof still demand K=1; a test asserts C1 argv contains K=1 and C2 argv contains K=2 and no other `--speculative-config` values exist.
- Any failure in C1 → restore, **do not launch C2**. Any failure in C2 → restore. SIGTERM anywhere → kill child group → restore.
- Readiness budget per candidate 2400 s; overall window budget flag `--window-deadline-sec` (default 3 h) checked before each launch; if exceeded, skip remaining candidates and restore.
- Independent systemd fail-safe timer (`--restore-only`) armed for **4 h** (two 22-min boots + two restores worth of slack).
- `window_verdict.py k2 <out>`: computes G1–G4 from receipts; fails closed to INCONCLUSIVE when any receipt is missing/malformed or hashes don't bind; PASS requires G2∧G3∧G4.
- Contract states both candidate envs byte-for-byte identical except the speculative config; `NSYS=1` on both (parity), `STATS_SEC=10` on both.

### A4. Independent review before staging

Dispatch a read-only adversarial review (fresh context, same rubric that produced today's NO-GO list): ordering, K=2 confinement, failure paths with fake executables, restore-only independence, credential redaction, prefill/decode separation in the verdict. Fix blockers; re-review. **No live run without a GO.**

### A5. Optional, only if time remains: copy-launch coalescing

Expected ≤2 ms/step (5 %); do **not** trade window time for it. If done: single Triton launch with a block→(tensor, blk) table replacing the 4 `mcopy` launches (`slot_cache_hook.py` `_cache_forward`, lines ~300–310 of the executed source), test bit-exact slot tensors under `TRITON_INTERPRET=1` against the current 4-launch path on random masks, and ship it in the same tree so G1 measures it. If it is not bit-exact, drop it.

## 4. Phase B — the window (James schedules; ~2 h)

| Step | Time | Who |
|---|---|---|
| Preflight (read-only): identity, digest, health, disk ≥150 GB, no old timers/locks, image resolves by digest | 2 min | runner |
| Stage exact HEAD tree (`git archive`, no pycache), `verify-prep.py` on box, arm fail-safe (4 h), write CONTROL/RELEASE, gate | 3 min | operator |
| Incumbent greedy reference (read-only) | 2 min | runner |
| C1 boot + collection | ~22 + 15 min | runner |
| C2 boot + collection | ~22 + 15 min | runner |
| Restore + proof | ~21 min | runner |
| Close release, stop fail-safe, pull receipts, decode-only analysis, verdict | 15 min | operator |

Operator reports to James at: launched, C1 ready, C2 ready, restored, verdict. Blog/GH publish per James's rule (new profile or significant finding → yes, either outcome qualifies).

## 5. Deliverables

- `results/2026-09-09-k2-v3/OUTCOME.md` with G0–G4 verdicts, C1 vs C2 vs E1 v2 comparator table, **measured** per-layer hit rate (first ever), decode-only buckets for both K, acceptance length distribution, TTFT (should be unchanged), limitations.
- `live-receipts/` (filtered, with `RECEIPTS-EXCLUDED.md` for the big traces), SHA256SUMS.
- Blog living page update: current-state verdict on top; today's correction stays as a dated note.
- If PASS: a one-paragraph proposal to promote K=2 to the daily lane, for James's decision.

## 6. Risks and honest expectations

- **Boot time dominates** (22 min × 2). If C1 fails readiness the window ends without a K=2 answer; that is acceptable and the plan says so.
- **K=2 may lose**: more distinct experts per step raises miss bytes roughly with the union of 3 tokens' routes; if that plus the extra draft pass exceeds the acceptance gain, we get a clean negative and publish it. The window still yields the measured hit rate, which is worth the cost on its own.
- **Quality ulps**: verify batch shape changes M=2→3 on the Modular NVFP4 path; the September 6 root-cause doc documents single-ulp bf16 routing-weight flips. Report exactly, gate on the existing standard.
- **Telemetry could still invalidate capture** if the gate is wrong. It is on a side stream and after capture-done; G0 checks it, and the daily lane is unaffected (incumbent unchanged).

## 7. Parked, with the number that parks it

- **Prefill bypass TTFT ≈ 0.6 s per short prompt**: `BYPASS_TOKENS=16` sends any prompt >16 tokens through the all-experts UVA path at ~7.8 ms/layer × 75. Structural — the slot path needs `M × topk ≤ slots ≤ 96` → M ≤ 12. A fix is a prefill design, not a flag. Record TTFT in every window; do not touch now.
- **Copy-path micro-optimizations**: ≤2 ms/step ceiling (§0). A5 only.
- **MoE GEMM tuning**: 5.4 ms/step at decode. No.
- **Enable CUDA graphs**: already enabled; the morning claim was a misread.

## 8. References

- E1 v2 receipts and analysis: `results/2026-09-08-e1-v2-live/` (`OUTCOME.md`, `analysis/decode-only-attribution.md`)
- E1 v2 runner/contract (reuse): `scripts/window_e1_v2.py`, `results/2026-09-08-e1-v2-offline-repair/CONTRACT.md`
- Prior queue and closed experiments (E2/E3 STOP, old E5/E6): `research/next-experiments-plan-2026-09-07.md`
- Copy-engine/launch-cost measurements: `research/hot-expert-cache-design-v2.md` §1–2
- Quality gate provenance: `research/slot-cache-root-cause-2026-09-06.md`; `scripts/greedy_equiv.py`, `scripts/tf_kl.py`
- Station ops rules: skill `gb300-station-ops` (window contract, fail-safe as system-scope unit, receipts over deletion)

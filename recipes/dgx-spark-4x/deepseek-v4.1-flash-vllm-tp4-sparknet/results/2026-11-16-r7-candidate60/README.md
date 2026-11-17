# 2026-11-16 — r7: candidate 60 in our shape, on Sparks 5–8

Five boots on one night, two four-Spark pools, one instrument. The question was narrow: **does Christopher
Owen's candidate-60 engine, run in our shape (switch, 4 KiB pages, 32 seats, KV 10.5 GiB/rank, late admission at
layer 20), beat the r6x lane we run today — and which of its two new scheduling defaults do we keep?**

Everything below is read from `receipts/` and `post/` in this directory. Lane definitions are in `config/arms/`;
the exact `docker run` for the promoted arm is `rendered-spark5.sh`.

## The five lanes

| lane | pool | engine | shape | prefill yield | overlays mounted |
|---|---|---|---|---|---|
| `r7-ctl-r6x` | Sparks 5–8 | r6x (`dsv41-karmic-kraken`, the kept 2026-10-23 config) | 32 seats / 8,192 / threshold 3,856 / KV 10.5 GiB | n/a (r6x has none) | none |
| `r7-a2` | Sparks 5–8 | candidate 60 | same | **off** | `so-bitmask` + `dsml-c60` |
| `r7-a3b` | Sparks 1–4 | candidate 60 | same | short, 10 s | `so-bitmask` only (stock DSML parser) |
| `r7-a3` | Sparks 5–8 | candidate 60 | same | **short, 10 s** | `so-bitmask` + `dsml-c60` — **promoted** |
| `r7-a3c61b` | Sparks 1–4 | candidate 60 + c61 decode overlay (6 files) | same | short, 10 s | `so-bitmask` + `dsml-c60` + c61 |

Two pools exist because the GhettoRak rack is split: Sparks 5–8 (Lane C) and Sparks 1–4 (Lane B, normally the
SGLang lane, released for this window). `r7-a3b` and `r7-a3c61b` ran on pool B, the rest on pool A. The same config
on the two pools (`r7-a3` vs `r7-a3b`) reads C1 +0.37%, C16 +0.76%, 131K prefill +0.92%, mixed-short −3.2%,
mixed-warm +2.5% — **the pools agree within ~1%**, so a pool-B arm is comparable to a pool-A control for the rows
in this table. `r7-a2` at 19:22 and `r7-a3` at 20:04 ran in the same slot on the same nodes, sequentially.

Arms A1 (Owen's own sizing: 16 seats, 16,384 batched tokens, threshold 16,288, late admission 30) and A4 (our
shape with late admission 30) were written and validated (they resolve cleanly) but **not booted**: A1 is not our
shape and A4 is a ±tens-of-ms knob on a row this instrument cannot resolve boot-to-boot. Both files are in
`config/arms/` if the next round wants them.

## Battery rows

Same instrument for all five: `v41bench_tf.py` prompt set v1 (8 categories, T=0, thinking off), `prefill_reps.py`
(unique nonce, token-id prompt), `mixed_probe.py` / `mixed_warm.py` behind a 413K cold prefill, `long_check.py` at
1M, `qeval.py` (75 auto-scored tasks), `agent_turn` (200K cached session + 1.5K fresh turn). C1 and C16 are the
mean of 3 reps on one boot; the bracket is min–max of those reps.

| row | `r7-ctl-r6x` | `r7-a2` (yield off) | `r7-a3b` (pool B) | **`r7-a3` (promoted)** | `r7-a3c61b` |
|---|---:|---:|---:|---:|---:|
| C1 per stream, tok/s | 103.33 [103.24–103.36] | 123.18 [121.22–125.68] | 122.72 [120.58–124.94] | **123.18 [123.02–125.36]** | 123.71 [121.80–125.33] |
| C1 aggregate, tok/s | 92.06 | 105.36 | 104.78 | **105.12** | 105.24 |
| C16 aggregate, tok/s | 471.76 [467.24–479.24] | 551.22 [524.88–555.78] | 541.56 [522.83–545.31] | **545.67 [535.74–556.48]** | 528.50 [524.09–555.23] |
| cold prefill 32K, tok/s | 5,481.0 | 8,507.3 | 8,442.2 | **8,554.5** | 8,366.4 |
| cold prefill 131K, tok/s | 5,314.8 | 8,702.2 | 8,645.8 | **8,725.7** | 8,587.1 |
| cold prefill 300K, tok/s | 4,956.0 | 8,510.9 | 8,470.7 | **8,554.0** | 8,399.7 |
| cold prefill 1M, tok/s (1 rep) | 3,853.7 | 7,807.7 | 7,769.5 | **7,848.8** | 7,707.4 |
| short turn behind a cold 413K prefill, s | 1.012 [0.890–1.021] | 1.511 [1.456–1.694] | 0.727 [0.598–0.728] | **0.704 [0.653–0.761]** | 0.848 [0.661–0.894] |
| warm-cache turn (68K resident + 1.5K), s | 2.063 [1.892–2.988] | 2.343 [2.325–2.879] | 1.998 [1.924–2.775] | **2.048 [1.758–2.478]** | 1.959 [1.793–2.696] |
| agent turn, 200K cached ctx, s | 0.791 | 0.490 | 0.509 | **0.511** | 0.492 |
| 1M needle | 3/3 in 263.3 s | 3/3 in 131.6 s | 3/3 in 132.4 s | **3/3 in 131.2 s** | 3/3 in 133.8 s |
| qeval content | 71/75 | 71/75 | 71/75 | **71/75** | 71/75 |
| KV pool, tokens (engine line) | 8,580,566 (8.18× at 1M) | 12,034,343 (11.48×) | 12,034,343 | **12,034,343 (11.48×)** | 12,034,343 |
| MemAvailable floor during C16, GiB | 22.2 | 25.1 | 24.9 | **25.2** | 25.2 |
| boot to HTTP ready, s | (container was already up) | 1,392 (cold JIT) | 1,395 (cold JIT) | **135 (warm)** | 211 (warm, recompiles) |

Every qeval run misses the same four tasks: `code_interval_intersect`, `json_count`, `json_escape`, `math_m9`.

**The KV line is not a like-for-like memory saving.** Both engines reserve the same
`--kv-cache-memory-bytes 11274289152` (10.5 GiB) per rank; the startup line moved from `kv_cache_utils.py:2854` on
r6x to `:2879` on c60 and now reports per-group pages (the log prints group sizes of 128 and 32). Treat it as the
engine's own accounting. What we tested at 1M is a single-sequence needle, not eleven concurrent 1M sequences.

Two spreads are wide enough to say so out loud: C16 on this rig moves ±2% boot-to-boot (551.22 / 541.56 / 545.67
across three c60 boots), and `mixed_warm` has a slow tail (its 3-rep ranges overlap at every lane). Only C1 is
tight enough (±0.2% within a boot) to read a 1% delta; we do not claim one anywhere else.

## Gates

Bars are the promotion gates set before the run (campaign `CAMPAIGN.md`), all against the same-night r6x control.

| gate | bar | `r7-a3` | evidence |
|---|---|---|---|
| Determinism | two identical greedy runs on one boot | **PASS** 40/40 | `post/r7-a3/greedy-1.json`, `greedy-2.json` |
| Determinism across boots | candidate output identical on a second boot | **PASS** 40/40 across `r7-a3` (pool A) and `r7-a3b` (pool B) | the two `greedy-1.json` files |
| Determinism under load | idle vs 15 background streams | **PASS** 40/40 | `post/r7-a3/greedy-load.json` |
| Quality | qeval ≥ control − 1 | **PASS** 71 vs 71, same four misses | `receipts/r7-a3/qeval.json` |
| Quality | 1M needle 3/3 | **PASS** 3/3 | `receipts/r7-a3/needle.json` |
| Quality | smoke + tool-call round | **PASS** `FUNCTIONAL_GATES PASS`, 0 preemptions | `post/r7-a3/gates.log`, `gates-functional.json` |
| Quality | DSML tool-call probe | **PASS** `ALL_PASS` | `post/r7-a3/dsml-live.log` |
| Mixed traffic | short turn behind 413K cold ≤ control + 20% | **PASS** 0.704 s vs 1.012 s (−30.4%) | `receipts/r7-a3/mixed_short.json` |
| Mixed traffic | warm-cache turn ≤ control + 20% | **PASS** 2.048 s vs 2.063 s (−0.7%) | `receipts/r7-a3/mixed_warm.json` |
| Speed | C1 not worse than control − 2% | **PASS** +19.21% | `receipts/r7-a3/c1.json` vs `receipts/r7-ctl-r6x/c1.json` |
| Speed | a real gain somewhere ≥ +5% | **PASS** C16 +15.67%, 131K prefill +64.18%, 1M prefill +103.67% | `receipts/r7-a3/{c16,prefill}.json` |
| Memory | MemAvailable floor ≥ 4 GiB on every node during C16 | **PASS** 25.2 GiB | `receipts/r7-a3/c16.json` (`memavail_during`) |
| #1194 guard | 64 concurrent structured-output requests, two waves | **PASS** 64/64, server alive | `post/r7-a3/so_probe-*.json` (control: `post/so-probe/r7-ctl-r6x.json`, also 64/64) |

Post-boot suite detail: `post/r7-a3/post-r7-a3.log` (gates, greedy ×2, greedy under load, so_probe, DSML live),
and the same suite on the two pool-B arms — `r7-a3b` and `r7-a3c61b` each also read 40/40, 64/64, `ALL_PASS`.

## What was declined, and why

**A2 — prefill yield off (`r7-a2`).** Speed is not the problem: C1 123.18 (identical to A3), C16 551.22 (the
highest of the three c60 boots, inside the ±2% spread), 131K prefill 8,702.2. The problem is the rows this shape
exists for. With yield off, a short turn behind a cold 413K prefill reads **1.511 s against the control's 1.012 s
(+49.3%)** and a warm-cache turn 2.343 s against 2.063 s (+13.6%). Both fail the "not worse than control by >20%"
bar. `--prefill-yield short` is what buys those rows back: 0.704 s and 2.048 s. A2 is the clean isolation of that
one lever — nothing else differs between A2 and A3.

**c61 — the decode overlay (`r7-a3c61b`).** Not adopted, and not because it is slower. Against `r7-a3b`, its
same-pool twin, C1 reads **123.71 vs 122.72 (+0.81%)** — inside the C1 spread of every boot in this table — while
C16 is **528.50 vs 541.56 (−2.41%)**, mixed-short 0.848 vs 0.727 (+16.6%) and 131K prefill −0.68%. The c61 README's
own claim is a C1-only effect gated on `num_reqs == 1`; our instrument cannot see it at the resolution it needs, and
the aggregate rows move the wrong way. Against the promoted `r7-a3` the C1 delta is +0.43%. Adding six mounted files
to a lane for a result inside the noise floor is not a trade we make. The six files and their hashes are in
`config/arms/`-adjacent notes (`~/hermes/work/.../results/R2-c61-overlay/`, cited in the campaign), not shipped in
`overlays/` — an unadopted overlay in the recipe's overlay directory is a footgun.

**Not a candidate this round:** teacher-forced non-inferiority against r6x. The r6x baselines come from the kept
r6x container, and a different engine build invalidates the per-token baseline — that comparison needs a fresh
reference boot, not a re-use. Quality here is qeval + needles + determinism + the tool probes.

## The #1194 fix and the DSML port

**spark-ds41f#1194.** c60 lets a copy draft verify more rows than `num_speculative_tokens` — the boot logs
`Copy drafts widen the CUDA-graph capture range from 192 to 203 rows`. `grammar_bitmask` sized its per-request
bitmask from `num_speculative_tokens` alone, so a structured-output request served by copy drafts could index past
it. `overlays/so-bitmask/` is **Zee-B33's three-line fix, verbatim**. We did not try to reproduce the crash;
`scripts/so_probe.py` (64 requests, two waves, answers that copy 40 records out of the prompt) is the guard probe. It
passed 64/64 on the r6x control (no copy drafts, so not exposed) and on every c60 arm, all of which carried the fix;
stock c60 without the fix was never probed. Treat the overlay as insurance,
not as a demonstrated fix on our rig.

**DSML structural closers.** `overlays/dsml-c60/` is our re-port of the structural rule onto c60's parser (R4's
work; the r6x file cannot be mounted on c60 — it lacks `DeepSeekV41ServingParser`). In-image gate:
`image-gate-run.log` — 25 cases, stock fails B/C/D/E/E2/J, the overlay fixes all six and changes only N, N2, O2, Q, R,
`ALL GATES PASS`, kill switch byte-identical to stock. The **live** probe passes on stock c60 too (`r7-a3b` mounted
no DSML overlay and read `ALL_PASS`): the live case is one quoted closer, which c60's token-provenance rule already
survives. The overlay's own value is the six synthetic genuine-atom cases the live probe does not build. Offline kit:
`tests/{matrix.py,compare.py,run_matrix.sh}`.

## Files

```
receipts/<lane>/          every phase's summary JSON + the raw bench logs it was read from (5 lanes)
post/<lane>/              gates.log, gates-functional.json, greedy-{1,2,load}.json, metrics-after-gates.txt,
                          dsml-live.log, so_probe-*.json, post-<lane>.log
post/so-probe/            the r6x control's so_probe (64/64)
config/arms/              A1–A4 as resolved for this campaign, plus the two pool-B variants
config/nodes-switched.*.json   the switched node maps the arms point at (pool A = 10.0.0.5–8, pool B = 10.0.0.1–4)
rendered-spark5.sh        the exact docker run for rank 0 of the promoted arm (no api-key line: the lane is keyless)
boot/arm-a3-boot.log      the launcher's own log: MemAvailable gate, launch order, READY 135 s, MemAvail_serving
boot/r7-a3-kv-and-init.txt     the KV/init lines from the c60 head log (pool 12,034,343; 203-row capture; overlay active)
boot/r7-ctl-r6x-kv.txt    the same lines from the r6x control (pool 8,580,566)
image-gate-run.log        the DSML overlay's in-image gate (25 cases)
tests/                    the DSML gate kit + live probe used by post_arm.sh
system.json               spark5 snapshot (scripts/collect_system_snapshot.sh), 2026-11-17 02:42 UTC
```

`system.json` was collected on spark5 after promotion (kernel 7.0.0-1025-nvidia, 4 KiB pages, driver 580.178.04,
CUDA 13.0); Sparks 6–8 match on kernel, page size and driver.
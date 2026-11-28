# 2026-11-22 → 2026-11-27: Pg65, Pg66, Pg67a, Pg67b, Pg67c, and a same-box one-user check against Flare

Five promotions took the lane from Pg64 to **Pg67c**, the config serving `:30006` since 2026-11-27 19:22:55 CST. This
bundle holds what each step changed, the receipts each promotion rested on, the exact files that serve, and Card F3:
Pg67c and Local Inference Lab's Flare package measured one after the other on the same Station.

Read the weak points first:

- **Pg67a, Pg67b and Pg67c were promoted on a "live-gates" path.** Each promote script's own gate block failed for a
  scripting reason (log source, a KV-bar constant, a smoke probe that tripped on a cold-build artifact). Each time the
  gates were then read off the running boot by hand and the promotion was declared in the lane-owner log. The
  failing logs are in [`promote/`](promote/) and [`cards/pg67-live-gates/`](cards/pg67-live-gates/) as they were written.
- **Card PF's verdict was HOLD, not PASS.** Its KV gate asked for an identical KV token count and read
  11,660,754 vs 11,659,893 (861 tokens); KV bytes, graph estimate, free memory and peak activation were byte-equal.
  James accepted the byte-equal-pool reading and PF1 was promoted. [`cards/PF-w1/PF-VERDICT.md`](cards/PF-w1/PF-VERDICT.md).
- **Card F's analysis step failed** (`ANALYZE FAILED: no arm dirs`). The F1a table below was recomputed for this bundle
  from the two arms' probe files, which are included.
- **No T2/T3 quality suite has run on any of the five configs, and no 48 h soak has completed.** The evidence is
  per-card: determinism 8/8 on each Pg67 boot, the F1/F1b numerics and gauntlet for Pg66, and for Pg65 a desk test
  suite plus the boot log line.
- **The published launcher does not build Pg65–Pg67c.** The serving tree was assembled on the box by the promote
  scripts here. [`patches/`](patches/) holds every delta, and `patches/SERVING-SHA256SUMS.txt` the hashes of what serves.

## What each step is

| config | promoted (CST) | = previous + | KV pool, tokens | rollback script |
|---|---|---|--:|---|
| Pg65 | 2026-11-22 08:37:56 | DSML markers count as structure only when they come from the real `｜DSML｜` token: a port of vLLM PR [#65019](https://github.com/vllm-project/vllm/pull/65019) (BorisLoveDev, head `cad5796a`, open) onto v0.31.0, merged with our structural-closers overlay. Seven parser overlay files. Kill switch `VLLM_DSV41_DSML_TOKEN_PROVENANCE=0` | 11,696,582 | [`rollback-pg65-to-pg64.sh`](promote/rollback-pg65-to-pg64.sh) |
| Pg66 | 2026-11-24 22:51:56 | two components of Garner McCloud's engine that our policy had left off: `w8oproj` (attention output projection + MXFP8 quantize in one kernel, 1–8 rows) and `lowb_aux` (attention side projections on the low-latency BF16 path at small row counts). Both change numerics by design. One line of `policy.env` | 11,660,754 | [`rollback-pg66-to-pg65.sh`](promote/rollback-pg66-to-pg65.sh) |
| Pg67a | 2026-11-27 12:24:58 | **F1a**, routing-glue fusion: the cold path's per-layer routing arithmetic becomes two 385-entry table lookups. [`fuse1_route.py`](patches/fuse1_route.py) + a hook hunk | 11,660,754 | [`rollback-pg67a-to-pg66.sh`](promote/rollback-pg67a-to-pg66.sh) |
| Pg67b | 2026-11-27 19:03:02 | **PF1**, a direct call plan for the eager `mm_mxfp8` projection GEMMs: runner and tactic are resolved once per shape key instead of on every call. [`pf1_mm.py`](patches/pf1_mm.py) + a hook hunk. Coupled to FlashInfer 0.7.0.post1; any mismatch leaves the stock path | 11,659,893 | [`rollback-pg67b-to-pg67a.sh`](promote/rollback-pg67b-to-pg67a.sh) |
| **Pg67c** | 2026-11-27 19:22:55 | capture rung 384 added to the CUDA-graph ladder (top was 256) and `--gpu-memory-utilization` 0.95 → 0.9531. Two lines | 11,813,533 | `WORK=<work-Pg67b> launch-pg61.sh down`, then `up` (no script) |

Two attempts did not hold. Pg67 (F1a + rungs 384/512/768) booted at 11:47 and was rolled back 12:05–12:13: its KV pool came
up at 11,324,211 tokens, under the 11.5M bar written into its promote script. A three-rung variant with the utilization pin (17:38) kept the KV pool
(11,665,488 tokens) but left 11.84 GiB free after capture against a 12.0 GiB bar; it failed at 17:46 and Pg67a
was restored. The first Pg66 attempt (22:28) failed its own "`w8oproj` not ACTIVE" check at 22:35 and
restored Pg65 by 22:42; the second passed.

## Pg66: Cards F1 and F1b

Two windows, each bracketed by the live lane as control. Per-class C1 step time, temperature 1.0, 3 seeds, 5 classes
([`pg66/F1-ANALYSIS.txt`](pg66/F1-ANALYSIS.txt), [`pg66/F1b-ANALYSIS.txt`](pg66/F1b-ANALYSIS.txt), spec
[`pg66/CARD-F1-SPEC.md`](pg66/CARD-F1-SPEC.md)):

| window | control step, ms | `w8oproj` | `w8oproj` + `lowb_aux` (= Pg66) |
|---|--:|--:|--:|
| F1 | 7.841 (7.855 / 7.827) | 7.721 | 7.692 (−0.149) |
| F1b | 7.840 (7.835 / 7.846) | 7.725 | 7.642 (−0.198) |

Teacher-forced log-prob distance to control stayed inside the control-to-control spread (F1: both-components arm vs
control mean 0.0119, control vs control 0.0132–0.0143; top-1 agreement 1.0 on every pair), C8 and C32 did not regress,
and the 12-task agent gauntlet read 12/12 on every arm. The spec's GSM8K sub-gate was mis-specified and is void, as the
spec itself says. The KV pool is 35,828 tokens smaller than Pg65's.

## Pg67a: Card F (F1a)

Server-side prefill time by uncached prompt tokens, median of 11 reps per cell, arms S (Pg66) and F (Pg66 + F1a) booted
in one window ([`cards/F-w1/`](cards/F-w1/)):

| uncached tokens (target) | reps | Pg66, ms | + F1a, ms | saved, ms |
|--:|--:|--:|--:|--:|
| 40 | 11 | 44.5 | 44.8 | -0.3 |
| 92 | 11 | 47.1 | 47.0 | +0.1 |
| 120 | 11 | 49.0 | 49.3 | -0.3 |
| 264 | 11 | 133.7 | 128.7 | +5.0 |
| 392 | 11 | 133.0 | 128.2 | +4.8 |
| 648 | 11 | 132.4 | 127.5 | +4.9 |
| 1144 | 11 | 132.3 | 127.1 | +5.2 |
| 1416 | 11 | 136.2 | 130.3 | +5.9 |

Cells at 264 and above ran eager on this ladder (top rung 256 rows; a cell's row count is its uncached tokens plus the
128-row replay). Mean of the four eager cells' deltas (264–1144): **+4.98 ms**. The three captured cells did
not move. A12 memory receipt equal between arms; determinism 8/8 identical. The card's op-count arm lost its shard to a
script bug, so the launch-count reduction (12 → 3 per layer) is the design's claim, not a measurement here.

## Pg67b: Card PF (PF1)

The card's own table is [`cards/PF-w1/PF-VERDICT.md`](cards/PF-w1/PF-VERDICT.md): eager cells 264 / 392 / 648 / 1144
went 124.0 → 105.8, 123.4 → 105.5, 125.0 → 105.9 and 122.4 → 104.3 ms; pooled **+18.33 ms saved**, 95 % CI
[+16.68, +19.90]. Captured cells within ±0.5 ms. Decode C1 −0.2 %, C8 −1.0 %, C32 +0.9 % against the same-window
control. Determinism identical. Verdict HOLD on the KV-token gate described at the top.

## Pg67c: what the rung buys

Two probe runs on the live Pg67c lane ([`pg67c-rung-check.jsonl`](cards/pg67-live-gates/pg67c-rung-check.jsonl),
[`pg67c-gate-fix.jsonl`](cards/pg67-live-gates/pg67c-gate-fix.jsonl)), every rep classified by the uncached token count
the server reported:

| uncached tokens | path on Pg67c | reps | server prefill, ms: median (min–max) |
|---|---|--:|--:|
| ≤ 256 (rows ≤ 384) | captured | 19 | 44.8 (37.3–54.8) |
| > 256 | eager | 17 | 103.8 (99.2–115.1) |

On Pg66 a turn with 129–256 uncached tokens ran eager at about 133 ms (Card F's S arm); on Pg67c it runs captured.
The serving boot's memory receipt ([`raw/live/serving-boot-lines.txt`](raw/live/serving-boot-lines.txt), container started
2026-11-27 22:13 CST): KV pool 11,813,533 tokens (Pg66: 11,660,754), A12 guard PASS, 12.47 GiB free after capture
against a 12.0 GiB bar; `FUSE1 selftest OK` on layers 0, 20, 21, 39; `MEGA_PEER PF1 ACTIVE` on FlashInfer
0.7.0.post1. An earlier probe also showed a real limit: a prefill that followed a cold 52K-token prefix build waited
575 ms at 40 uncached tokens (`pg67c-probe.jsonl`, 264-cell rep 1; 583 ms in the same spot on Pg67b). The cause is not
established. The probe now waits 2 s after a cold build; the lane's behaviour is unchanged.

## Card F3 (2026-11-27 21:30–22:29 CST): one user, Pg67c vs Flare's current one-GPU core

Rule and bars were written before any cell ran: [`cards/F3/CARD-F3-SPEC.md`](cards/F3/CARD-F3-SPEC.md). Flare is
`local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE` at revision `gb300` (commit `b55be385`, core `d683cc1a`, the
2026-11-26 release), runtime `NVIDIA/hyperflare-inference` at `7db06e0c` built from the repo's Dockerfile, their README
serving flags (`--mtp-draft-tokens 7 --embedding-table-storage ram`), default settings, GB300 only. Order: Pg67c (P1),
Pg67c down, Flare (F26), Pg67c restored, Pg67c again (P3).

| one stream | Pg67c before (P1) | Flare (F26) | Pg67c after (P3) | Pg67c / Flare |
|---|--:|--:|--:|--:|
| llm-inference-bench 0.6.2, no context, 30 s windows, median of 3 runs | 389.4 (403.1, 389.4, 380.6) | 322.7 (333.9, 322.7, 320.5) | 380.7 | 1.21× |
| per-class C1, T=1, thinking off, 256 forced tokens, mean of 3 seeds × 5 classes | 498.9 | 366.3 | 500.4 | 1.36× |
| per-class C1, natural length (one run) | 541.1 | 392.5 | not run | |

Verdict by the pre-registered rule (a lead under 3 % is a tie): Pg67c on both instruments. Per class, tok/s:

| arm | prose | structured | code | shell_ops | tool_json |
|---|--:|--:|--:|--:|--:|
| Pg67c (P1) | 406.1 | 447.8 | 555.6 | 574.9 | 509.9 |
| Flare (F26) | 318.9 | 341.8 | 387.1 | 392.1 | 391.6 |
| Pg67c (P3) | 411.5 | 452.4 | 553.8 | 578.8 | 505.6 |

The full llm-inference-bench table, one run per arm, aggregate tok/s:

| arm | C1 | C8 | C32 | C1 @16K | C8 @16K | C32 @16K |
|---|--:|--:|--:|--:|--:|--:|
| Pg67c (P1) | 403 | 1,267 | 2,405 | 365 | 1,272 | 2,407 |
| Flare (F26) | 334 | 1,137 | 2,379 | 347 | 1,146 | 2,338 |
| Pg67c (P3) | 381 | 1,290 | 2,403 | 366 | 1,253 | 2,424 |

What this does and does not say:

- It is one boot per engine, on our box, at Flare's defaults. Flare's card lists 387.5 tok/s for one stream and
  3,858.5 at 32 in their own measurement; our run of their core read lower on both, and we have not ruled out our own
  setup (it ran in a container behind our launcher, untuned). Treat the Flare rows as our setup's reading, not their
  core's ceiling.
- **No many-user claim.** Flare's 2026-11-20 core, measured here on 2026-11-25 with 5 draft tokens, read
  1,529 at C8 and 4,931 at C32 on the same client, far above this run. We have not found what
  differs.
- Flare passed its smoke cells in this window: GSM8K-200 194/200, 3 needles at 200K, a tool call, the 12-task gauntlet
  12/12, and a ~114K cold prompt in 1.46 s ([`raw/f3/F26/bench.md`](raw/f3/F26/bench.md)).
  The two throughput windows of that bench read 0.0 on Flare because the bench takes them from vLLM's metrics; they
  are not results.
- The per-class harness reaches Flare through a key-renaming proxy. On 2026-11-25 Pg66 read 501.0 direct and
  495.5 through it on the same instrument ([`pg66/F2-analysis.json`](pg66/F2-analysis.json)); no correction is applied.
- Pg66 on the same client three days earlier: C1 380.7 (382.2, 377.1, 380.7),
  C8 1,256.7, C32 2,430.8 ([`pg66/llm-inference-bench/`](pg66/llm-inference-bench/)).
  Pg67c's decode is within run-to-run spread of it, as intended: the three Pg67 changes target eager prefill.

Flare is the work of Local Inference Lab (voipmonitor / Martin Vit, lukealonso, original-el8) on NVIDIA's Flare
engine, with the Parallax expert-residency technique by Jason Cook (original-el8). Nothing of theirs is in this recipe.

## Files

- `patches/`: `pg65-dsml-token-provenance.diff` (parser overlay, Pg64 → Pg65), `mega_peer_hook-f1a-on-pg66.diff`,
  `fuse1_route.py`, `mega_peer_hook-pf1-on-pg67a.diff`, `pf1_mm.py`, `serve-sh-rung384-pg67c.diff`,
  `recipe-env-util-pg67c.diff`, the `policy.env` diffs (paths scrubbed), `SHA256SUMS`, and `SERVING-SHA256SUMS.txt`.
  `fuse1_route.py` and `pf1_mm.py` are byte-identical to the files hashed inside the running container.
- `promote/`: every promote and rollback script and log for the five steps, and the lane-owner excerpt.
- `cards/`: F-w1, PF-w1, the Pg67 live-gate receipts, and F3 (spec, verdict, runner log, scripts).
- `raw/f3/`: the three arms' bench JSON and per-class tables. Per-request rows are withheld: the prompts are private
  Hermes transcripts (sha1 `618898aac432`). `raw/live/`: `docker inspect` of both containers, in-container hashes,
  boot lines, the memory receipt.
- `pg65/`: the port report and mount list. `pg66/`: Cards F1 / F1b and the 2026-11-25 same-client runs.
- `facts.json`: every number in this README and in the recipe's Pg67c text, computed from the files above.

One box detail a reader of `raw/live/ds41-inspect.json` will see: Pg67b and Pg67c mount the vLLM overlay from the PF
card arm's work directory rather than from their own. The files are the same bytes (hashed in the container); the path
is a leftover of the hand promotion.

# Pg62 "W1-exact" (2026-11-14): Pg61 + an exact FP8 decode-attention fast path — PROMOTED, experimental

**Verdict: PROMOTED 2026-11-14 23:11:55 CST** (James: "both together tonight"). Pg62 serves `:30006` on the reference
box. Status stays `experimental`: the 48 h soak started 2026-11-14 23:11:39 CST (`raw/promote/live/soak-start.json`) and
ends **Monday 2026-11-16 11:11 PM CST (05:11Z 11-17)**. No T2/T3 quality bench has run on Pg62.

Pg62 = Pg61 + two changes to decode attention. Nothing else changes: same image, same args except one env, same
1M context, same 32 seats, same speculation schedule, same KV pool (11,696,582 tokens).

1. **FP8 compressed-layer fast path.** Target layers 2–39 at 1–16 decode rows now run Garner McCloud's W1 bitwise
   kernel on `fp8_ds_mla` 528 B records instead of the stock FlashMLA op
   (`torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode`). The routing comes from the "W13" patch
   (`fastpath/w1attn/{integrate,ops,formats}.py` plus a second extension, `w1_attn_fp8_ext.so`), enabled by
   `DS41_W1_COMPRESSED_FP8=1`. At 17+ rows the stock op still runs. Pg61 already ran W1 on window-only layers.
2. **Exact rounding in W1, both extensions.** W1 computed the softmax sink term `sink*log2e - m` as one fused
   multiply-add; the stock op multiplies, then subtracts. That one float step sometimes lands a pre-quantization value on
   the other side of an FP8 rounding midpoint, and ONE output data byte flips. Both served extensions are rebuilt with
   `-DW1_VARIANT_STOCK_ALL`: sink unfused + ex2 without FTZ + `osc` divide without FTZ + output-scale divide `div.rn`.

**What "exact" means here:** every W1 decode-attention call that the shadow compared wrote the same bytes as the stock
FlashMLA op on the same inputs, on the workload tested. The claim covers only W1 decode attention. It says nothing about
the rest of Garner's engine against bare vLLM, and it does not cover the cluster (cx) kernel or FP4 records, which this
lane cannot exercise.

**Consequence:** Pg61 was never byte-equal to the stock op on window-only layers (70 of 17,797 calls differed,
shadow-h1). Pg62 is byte-equal on the tested workload. Pg62's greedy output therefore differs from Pg61's on some
prompts by design: tokens/step on the C1 sets moved 3.125 / 3.091 (Pg61) → 3.123 / 3.089 (arm F2).

## Arms

Performance cells: `gt_mix_drive`, greedy; C1 sets A and B, 20 prompts × 256 tokens each; C8 and C32 aggregates;
3 measured reps after warm-ups. Stock controls are booted in the same runner window around each candidate
(`raw/cells/runner.log`, `arms*.tsv`, one JSON per cell). Every arm bound with KV 11,696,582 tokens except S1b
(11,686,684).

| arm | tree | what changes vs Pg61 (shipped incumbent `w1_attn_ext.so` 7201643d… unless stated) |
|---|---|---|
| A1–A7, P-A1, P-A2 | `serve-spec` | none: stock Pg61 controls (W1 FP8 off) |
| S1b-k2-9to32 | `serve-spec` | draft length 2 at 9–32 running requests (`[[1,4,5],[5,8,3],[9,32,2]]`) |
| B-divfix | `serve-bisect` | FP8 fast path, W13 + exact output divide only (`bf6ccf96`) |
| B-stockall, P-B | `serve-bisect` | FP8 fast path, STOCK_ALL build `de874506` |
| G32, G76 | `serve-bisect` | FP8 also on the cx kernel up to 32 / 76 rows (build `cbdcea86`) |
| **F2, P-F2** | `serve-bisect` | **both fixes = Pg62's two binaries**: FP8 `de874506` + incumbent rebuilt STOCK_ALL `3bca6cac` |

The B, F2, G and P-B arms ran on the test tree (`serve-bisect`, with the bisect copy of `integrate.py`,
`scripts/integrate_bisect.py`, mask = all layers, all widths). The promoted tree got the oracle, one C1 pass and the
live caller and per-class probes.

## Byte equality: the shadow compare

`scripts/integrate_shadow_H.py` (a test instrument; never the served file) ran both W1 and the stock op on every
decode-attention call inside the CUDA graphs. The model consumed the stock op's output, and the two output buffers were
compared byte for byte on the device. Counts below are `dump-1` of each run (`raw/shadow/<run>/dump-1/stats.json.gz`,
`devices.cuda:0.sites`). Workload: a 16-prompt prose pass (`prose16`) + the full 8-prompt oracle pass (`w0_drive.py`)
+ C1 sets A and B + one C8 cell. shadow-e1 ran only prose16 + the oracle pass; lc1 and lc2 replace the gt cells with
the long-context drive.

| run | build (FP8 / incumbent) | FP8 path: calls with any mismatching byte | incumbent window-only path |
|---|---|---|---|
| shadow-e1 | W13 unfixed / not shadowed | **3 of 19,152** (one data byte each, 0 scale bytes) | — |
| shadow-e2 | STOCK_ALL / not shadowed | 0 of 133,304 | — |
| shadow-h1 | STOCK_ALL / **shipped 7201643d (= Pg61)** | 0 of 132,278 | **70 of 17,797** (one data byte each, 0 scale bytes) |
| shadow-i-stockall | STOCK_ALL / rebuilt STOCK_ALL 3bca6cac | 0 of 132,240 | 0 of 17,779 |
| shadow-i-sinkonly | STOCK_ALL / rebuilt sink-unfused-only 6ff853ae | 0 of 132,354 | 0 of 17,733 |
| shadow-lc1 (long context) | STOCK_ALL / not shadowed | 0 of 44,270 | — |
| shadow-lc2 (long context, cutover gate) | STOCK_ALL / 3bca6cac | 0 of 46,702 | 0 of 6,913 |
| shadow-g1 (cx up to 32 rows; not in the recipe) | cx build cbdcea86 / not shadowed | 0 of 152,342 | — |

The shipped incumbent's 70 mismatches by layer (shadow-h1): target L0 1 / 3,569, L1 17 / 3,569; drafter L40 7 / 3,553,
L41 15 / 3,553, L42 30 / 3,553. shadow-e1's counter RESET failed, so its dump-1 includes the prose16 pass. Its dump-0
(228 calls) shows 1 mismatch that may be one of the 3 (`LEDGER.md` 16:03).

Long context (lc1, lc2, via `scripts/lc_drive.py`): prompts of 7.6K / 61K / 238K / 571K tokens × 2 at C1 (up to 256
new tokens) plus 16 × 28.6K at 8 concurrent. lc2: `LC errors 0`, largest prompt 571,462 tokens (`raw/shadow/lc2/lc.log`).
The cutover gate (`raw/promote/gate.log`) armed the cutover only on 0/0 mismatches: `GATE PASS -> cutover` at 23:05:13.

## Greedy oracle

8 prompts (prose, code, math, json, shell, tool, long, story), up to 256 tokens each (the json, tool and story
references stop at 18, 21 and 157), with top-20 logprobs. The reference is `serve-quality-a2` (arm A2 = Pg61 config,
W1 FP8 off, shipped incumbent). Only the **first pass after boot** is an oracle: on the control arm `ctl-none` the
second pass is 7/8 against the reference. `raw/oracle/arm-*/verdict.json`, `raw/oracle/runner.log`; comparer
`scripts/compare.py`.

| arm (FP8 mask: layers × widths) | r0 text / logprob-exact | r0 mismatching | reading |
|---|---|---|---|
| ctl-none (no FP8 layers) | 8/8 · 8/8 | — | control |
| full (W13 unfixed, all) | 5/8 · 5/8 | code, prose, shell | the failure |
| L2-7 / L2-19 / L20-39 | 7/8 / 6/8 / 7/8 | shell / code, shell / prose | spread over layer groups |
| w6only / w8only / w4-12-16 / w6-8-12 | 7/8 / 6/8 / 8/8 / 5/8 | shell / code, prose / — / code, prose, shell | not one width |
| nopdl / seq / nold / divfix | 5/8 each | code, prose, shell | PDL, plan, early trigger, exact divide alone: no effect |
| stock-before (stock, then W1 overwrites) | 5/8 · 5/8 | code, prose, shell | W1's output differs on live inputs |
| stock-after (W1, then stock overwrites) | 8/8 · 8/8 | — | W1's presence does not perturb |
| **stockall (STOCK_ALL FP8 build)** | **8/8 · 8/8** | — | **fixed** |
| **Pg62 live, after cutover** (`raw/promote/cutover-20261114-2305/verdict.json`) | **8/8 · 8/8** | — | promoted tree |

The shadow-h1 / i / lc2 boots also logged an r0 8/8 verdict against the same reference (`raw/shadow/*/run.log`).

## Speed

Test tree, one window: A6 (22:14) → F2 (22:24) → A7 (22:33); TTFT P-A1 / P-B / P-A2 (19:33–19:55) and P-F2 (22:43).
All values from the cell JSONs (`throughput.csv`).

| cell | stock Pg61 (A6, A7) | **Pg62 binaries (F2)** | |
|---|---|---|---|
| C1 decode, set A (tok/s) | 413.88–414.72 (mean 414.22) | **439.36–439.57** (mean 439.47) | **+6.1 %** |
| C1 decode, set B (tok/s) | 409.88–410.80 (mean 410.25) | **435.28–435.42** (mean 435.33) | **+6.1 %** |
| C1 step (set A / B) | 7.639–7.654 / 7.628–7.645 ms | 7.205–7.209 / 7.191–7.194 ms | |
| tokens per step (A / B) | 3.125 / 3.091 | 3.123 / 3.089 | outputs now follow stock values |
| C8 aggregate (tok/s) | 1,085.1–1,108.6 | 1,087.6–1,094.9 | no change |
| C32 aggregate (tok/s) | 1,626.1–1,833.2 | 1,794.1–1,809.9 | no change (control cells spread 13 % in this window) |
| TTFT 32 / 1K / 8K / 64K tokens (s, median of 3) | 0.020 / 0.121 / 0.183 / 1.466 (P-A1) · 0.020 / 0.125 / 0.187 / 1.462 (P-A2) | 0.020 / 0.122 / 0.182 / 1.458 (P-F2) | no change |
| KV capacity (tokens) | 11,696,582 | 11,696,582 | |

TTFT prompt lengths are 30 / 953 / 7,620 / 60,953 tokens. The 8K first rep is a cold outlier on every arm (1.191 s
on P-F2); the median drops it. P-F2 has no control booted in its own runner start; P-A1 / P-A2 bracket P-B
(19:33–19:55).

FP8 fix alone (B-stockall vs its control A4): C1-A 436.43–440.87 (A4 397.67–414.19), C1-B 436.70–436.84
(A4 410.19–410.37); tokens per step 3.125 / 3.091, the same as stock, because the incumbent was still the shipped one.
P-B (FP8 fix alone) TTFT 0.020 / 0.119 / 0.184 / 1.462 s.

**Live Pg62 lane, after the cutover** (`raw/promote/cutover-20261114-2305/c1-A.json`, `cutover-pg62-20261114-2305.log`):
C1-A 439.33 tok/s, 7.214 ms/step, 3.126 tokens/step, 0 errors. KV 11,696,582. p3 watchdog ENFORCE (ppid 1), trip null.

### On the live lane, 2026-11-15 (light C1 load; soak not restarted)

**Caller probe** (`raw/caller/caller-pg62-20261115.json`, 05:11:51–05:12:35 CST; the same script as
`../2026-10-29-pg61-garner-engine/caller-2026-10-30/caller_probe.py`). The Pg61 column is that bundle's 2026-10-30
run: context, not a paired ratio.

| measure | Pg62, 2026-11-15 | Pg61, 2026-10-30 |
|---|---:|---:|
| single stream, 400-token reply (median of 3) | 330 tok/s (305 / 330 / 408) | 343 tok/s (303 / 343 / 346) |
| 40-token turn, complete (median of 5) | 85 ms | 91 ms |
| 8 concurrent streams × 400 tokens, aggregate (median of 3) | 1,094 tok/s (847 / 1,094 / 1,097) | 1,088 tok/s (858 / 1,088 / 1,107) |
| first token, idle (median of 5) | 32 ms | 35 ms |
| quiet single-stream step (median inter-chunk gap) | 7.0 ms (p90 7.3) | 7.5 ms (p90 7.9) |
| worst-10 % gap while a 33K prompt is read (median window p90) | 149 ms | 132 ms |
| that 33K prompt's first token (median) | 0.89 s | 0.87 s |

The single-stream median is the middle of a wide spread (305–408); the step time is the steadier signal.

**Per-class C1, temperature 1.0** (`raw/perclass/`; the Card G5 instrument `perclass_decode_t.py`, the same private
prompt set, sha1 `618898aac432`, 40 prompts × 256 forced tokens per class, top_p 1.0, seeds 42/43/44, 05:12:35–05:18:34
CST). Rows withheld: private Hermes transcripts. The Pg61 column is Card G5 (2026-10-29 21:12 CDT, one boot): **context,
not a paired ratio.** Different window, different day, and here the lane was live.

| class | Pg62 tok/s (3 seeds) | Pg61 G5 tok/s (3 seeds) | acceptance_len Pg62 / Pg61 |
|---|--:|--:|---|
| prose | 405.3 (402–413) | 367.2 (356–377) | 3.05 / 3.02 |
| structured | 441.1 (437–445) | 396.9 (393–404) | 3.35 / 3.27 |
| code | 556.2 (555–557) | 494.9 (480–504) | 4.52 / 4.30 |
| shell_ops | 561.9 (543–578) | 530.4 (525–536) | 4.58 / 4.65 |
| tool_json | 489.6 (478–511) | 463.8 (456–471) | 3.87 / 3.97 |
| **class mean** | **490.8** | 450.6 | 3.778 / 3.744 (pooled) |

0 errors, 0 foreign tokens, max running 1 in every seed. acceptance_len = 1 + Σaccepted / Σdraft steps from the per-run
JSONs (the same method reproduces G5's published 3.744). The gt_mix bracket above (+6.1 % same window, greedy) is
the paired speed number. The per-class gap (+9 % class mean) mixes the change with the window and a sampled workload.

## Tried and not in the recipe

| arm | result | why not |
|---|---|---|
| draft length 2 at 9–32 running (S1b) | C32 1,596.8 / 1,817.5 / 1,826.0 vs controls A1–A3 1,597.4–1,838.4 | no C32 gain; KV 11,686,684 (−0.085 %) |
| FP8 on the cx kernel, 17–76 rows (G32, G76) | byte-exact in shadow-g1 (0 of 152,342 calls); C8 G32 1,082.9–1,102.8, G76 1,089.9–1,097.1 vs A5 1,087.1–1,098.4; C32 within the control spread | no throughput gain |
| exact output divide alone (`div.rn`, divfix) | oracle 5/8; C1 441.25–441.40 / 437.40–437.46 (B-divfix) | does not fix fidelity; kept inside STOCK_ALL |
| PDL / early `launch_dependents` removed (nopdl, nold) | oracle 5/8 each | no effect |

## The diagnosis, briefly (dead ends included)

The unfixed W13 FP8 path failed the greedy oracle 5/8. Bisecting by layers, widths, kernel plan, PDL and divide pointed
at value-dependent inexactness spread over layer groups, not one untested layer or width. Two replays of captured
real inputs (76 fixtures each, prose layers 2–39 at the failing steps) passed 76/76 in eager and graph modes while the
served lane still failed. Those replays were a dead end. The in-graph shadow compare found the actual difference: 3
calls in 19,152, one byte each. Rebuilding one call offline (L14, 8 rows) gave a fixture. On that fixture the sink switch
alone flipped the byte back to stock (110), and the other three switches did nothing
(`raw/builds/fp8/*/F-probe.json`). A second shadow of the incumbent path found that Pg61's shipped window-only kernel
had the same flaw (70 of 17,797). Timeline: `raw/LEDGER.md`.

## What each gate does not exercise

- **Shadow:** the widths that occurred. FP8 calls ran at 6, 8, 12 and 16 rows. Incumbent calls ran at 6–24 rows. No
  call at 1–5 rows was compared on either path in these workloads (zero calls at those widths in `stats.json.gz`).
  Not exercised at all on this lane: the incumbent's FP4 compressed path and its 2-CTA cluster (cx) kernel (17–76
  rows on FP4 records, per the `integrate.py` docstring). Compressed records here are FP8. The incumbent-variants
  patch changes those kernels too, in source only (untested).
- **Oracle:** 8 prompts, greedy, first pass only. Its reference A2 is the Pg61 config with the FP8 path off, so it
  carries the shipped incumbent: it checks the FP8 change against Pg61, not against bare vLLM.
- **Quality:** no T2/T3 bench on Pg62 (no needle, GSM8K, BFCL, MMLU-Pro, gauntlet, TF bracket). No temperature > 0
  instrument beyond the per-class speed run.
- **Tree:** C8, C32 and TTFT were measured on the test tree (`serve-bisect`, the same two binaries, bisect
  `integrate.py`), not on the promoted tree.
- **Soak:** the 48 h clock runs to Monday 2026-11-16 11:11 PM CST. The Pg61 soak never completed.
- **Rollback:** `scripts/rollback-pg62-to-pg61.sh` has never been drilled. No `--dry-run` log is in this bundle.
  The round ledger (22:58) records `pg61-work` as left untouched. The dry run checks `pg61-work`'s
  `BUILD-SHA256SUMS`.
- **Launcher:** Pg62's tree was assembled by hand on the box (`LEDGER.md` 22:58). The published
  `DS41_PG62=1 launch-pg61.sh build` path was desk-checked (`patch-desk-check.txt`) and then run on the box 2026-11-15 from a fresh
  clone at the pin: both `.so` byte-identical to the served files, `tree/` identical, `verify` OK on the live lane
  ([`raw/launcher-build/`](raw/launcher-build/)). No lane has been booted from the launcher-built tree.

## Files

| path | what |
|---|---|
| `throughput.csv` | every speed cell (355 rows), one JSON value per row |
| `patch-desk-check.txt` | desk checks of `06`, the W13 patch, the `DS41_PG62` build and the unchanged default path |
| `BUILD-SHA256SUMS.expected` | the box's `pg62-BUILD-SHA256SUMS` with the published watchdog's hash (`c97d149d…` for the lane's `ba8373e9…`; two default log paths differ). Check a `DS41_PG62=1` build against it: expect FAILED on the three `.so` and, unless `WORK` is the box path, `policy.env` |
| `raw/LEDGER.md` | dated round log (box copy) |
| `raw/RECIPE-DRAFT.md` | the round's draft write-up (box copy, superseded by this README) |
| `raw/cells/` | perf runner log, `arms*.tsv`, one directory per arm (`args.json`, one JSON per cell) |
| `raw/oracle/` | oracle runner log, per-arm `fp8_mask.json` + `verdict.json` |
| `raw/shadow/<run>/` | shadow `run.log`, `engage.txt`, `fp8_mask.json`, `dump-*/stats.json.gz`, gt cells, `lc.log` |
| `raw/builds/` | FP8 and incumbent variant builds: `FLAGS`, `so.sha256`, compile logs, fixture `F-probe.json` |
| `raw/promote/` | cutover gate log, cutover receipts (oracle verdict, C1), live `docker inspect` args/env/mounts, in-container sha256, `pg61-`/`pg62-BUILD-SHA256SUMS`, `pg62-policy.env`, `serve.sh` diff, autostart, soak start |
| `raw/caller/` | live caller probe JSON + log |
| `raw/perclass/` | live per-class tables (`*.md`), JSONs (rows withheld), logs, `/metrics` before/after |
| `scripts/` | instruments, scrubbed (box home path → `$HOME`, GPU UUID → `GB300_UUID`): `compare.py` (oracle), `integrate_shadow_H.py` + `integrate_bisect.py` (test instruments, never served), `lc_drive.py`, `run_shadow*.sh`, `run_perf_i.sh`, `spec_runner2.sh`, `prefill_runner.sh`, `bisect_runner.sh`, `f_probe.py`, `run_f*.sh`, `incumbent_probe.py`, `run_i1.sh`, `build_variant.sh`, `build_incumbent.sh`, `gate_then_cutover.sh` |

The scripts are the box copies with paths scrubbed. They name box-only dependencies (`$HOME/dsv41/allround-20261112`
trees, `w0_drive.py` / `gt_mix_drive.py` from `jmeadlock/gb300-dsv41-decode`, the `results/G` drafts) and were not re-run
from this directory. `*.jsonl` prompt and output files are not published.

Upstream: the rounding difference is in Garner's W1 kernels. It has not yet been reported to him.

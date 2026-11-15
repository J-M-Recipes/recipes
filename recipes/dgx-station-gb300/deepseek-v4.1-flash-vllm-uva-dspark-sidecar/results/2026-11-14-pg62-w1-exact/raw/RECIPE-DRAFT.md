# DRAFT — DSV4.1-Flash on one GB300: FP8 decode-attention fast path, exact (2026-11-14)

Status: draft for James. Not promoted, not published. `:30006` is dark; Pg61 production tree untouched.

## What changes vs Pg61
One thing: compressed-layer (2..39) decode attention at 1..16 rows runs the W1 bitwise kernel on `fp8_ds_mla`
528 B records instead of the stock FlashMLA op. Launch difference: the candidate fast-path mount and
`DS41_W1_COMPRESSED_FP8=1`. Everything else is Pg61 (1M context, 32 seats, same speculation schedule).

Binary: W13 hardening tree + `results/F/variants.DRAFT.diff`, built `-DW1_VARIANT_STOCK_ALL`
(`w1_attn_fp8_ext.so` sha256 de87450648e96214…, on the box as `serve-bisect/tree/fastpath/w1attn/*.so.stockall`).
The fix that matters is the sink term: W1 computed `sink*log2e - m` as one fused multiply-add, stock multiplies
then subtracts. One float step of difference flipped one output byte in roughly 1 call in 5,000, which greedy
decoding amplified into different text on 3 of 8 prompts. `-DW1_VARIANT_SINK_UNFUSED` alone fixes the isolated
failing call; STOCK_ALL (that plus exact output-scale divide and two non-ftz forms) is the build that was served.

## Measured (same day, stock controls bracketing)
| | stock Pg61 | candidate |
|---|---|---|
| C1 decode, set A (tok/s) | 414.0-414.5 | 440.5-440.9 (+6.4 %) |
| C1 decode, set B | 409.8-410.4 | 436.7-436.8 (+6.5 %) |
| C1 step time | 7.64-7.65 ms | 7.17-7.19 ms |
| tokens per step (A / B) | 3.125 / 3.091 | 3.125 / 3.091 |
| C8 aggregate | 1,073-1,111 | 1,089-1,091 (no change) |
| C32 aggregate | ~1,600-1,840 | 1,821-1,836 (no change; single cells vary ~12 %) |
| TTFT 32 / 1k / 8k / 64k tokens (s) | 0.020 / 0.121 / 0.183 / 1.466 | 0.020 / 0.119 / 0.184 / 1.462 |
| KV capacity (tokens) | 11,696,582 | 11,696,582 |

## Fidelity
- Greedy oracle (8 prompts x 256 tokens, top-20 logprobs, first pass after boot): 8/8 text and logprob-exact vs
  stock. Stock vs stock is 8/8; the unfixed W13 build was 5/8.
- In-graph shadow compare (both kernels on every call, model on stock values): 0 mismatching bytes in 133,304 FP8
  calls (oracle prompts, C1 sets A and B, one C8 cell). Unfixed W13: 3-4 mismatching calls in 19,152.
- Not covered: long contexts (compressed top-k near 512 entries), sampling workloads beyond these cells, soak.
- The second oracle pass in a boot differs from the first on stock itself (7/8), so only first-pass is an oracle.

## Tried and not in the recipe
- Draft length 2 for 9-32 running requests: no gain at C32, KV -0.085 %.
- FP8 on the cluster (cx) kernel for 17..76 rows: byte-exact in shadow (0 of 152,342 calls, ~18.7k at 18-32
  rows) but C8 and C32 throughput unchanged. Kept as `results/G/` drafts.
- Prefill chunk threshold 4096 (Astra): lost.

## Before promotion (needs James)
1. Serve the sink-unfused-only build once (oracle + shadow) if the smaller patch is preferred over STOCK_ALL.
2. A/B/A in the production tree with the watchdog, then promote/rollback via the existing Pg61 scripts.
3. Tell Garner: the fused sink term is in the upstream W1 kernels (all record formats); FP4 lanes may carry the
   same one-byte rounding difference vs stock.
Evidence: `milo-takeover/LEDGER.md`; box `~/dsv41/allround-20261112/{milo-takeover,shadow-e1,shadow-e2,shadow-g1,perf-g1}`.

## Addendum 2026-11-14 late: incumbent window-only path (James: fix it)
Pg61's shipped `w1_attn_ext.so` (window-only `ss` kernel: target layers 0-1, drafter layers 40-42) differs from
the stock FlashMLA op on 70 of 17,797 live calls, one data byte each (same fused sink term). A rebuild of the
Pg61 incumbent source with `results/I/incumbent_variants.DRAFT.diff`:
- `-DW1_VARIANT_STOCK_ALL` (sha 3bca6cac…): 0 of 17,779 incumbent calls, 0 of 132,240 FP8 calls differ.
- `-DW1_VARIANT_SINK_UNFUSED` only (sha 6ff853ae…): 0 of 17,733 and 0 of 132,354.
With both fixes every decode-attention call on this lane wrote the stock op's bytes on the tested workload
(oracle prompts, C1 sets A/B, one C8 cell; incumbent path not yet run on long contexts).
Speed with both fixes: C1 439.5 / 435.3 tok/s (+6.1 % vs stock 414 / 410; FP8 fix alone 440.7 / 436.7),
tokens per step 3.123 / 3.089 (Pg61 3.125 / 3.091, because outputs now follow stock values, not Pg61's),
C8 1,088-1,095 (stock 1,085-1,109), C32 1,794-1,810, TTFT unchanged, KV 11,696,582.
Greedy output of the two-fix build will differ from Pg61 on some prompts by design; the 8-prompt oracle is
unaffected (8/8 either way).

## Status 2026-11-14 23:12: PROMOTED as Pg62 (James: both together tonight)
Tree `~/dsv41/garner/pg62-work` = pg61-work + `fastpath/w1attn/{integrate,ops,formats}.py` (W13),
`w1_attn_fp8_ext.so` de874506, `w1_attn_ext.so` 3bca6cac, two lines in `serve/serve.sh`, and
`DS41_W1_COMPRESSED_FP8=1` in policy.env. Launch unchanged: `launch-pg61.sh up|down` with `WORK=…/pg62-work`.
Gate before cutover: long-context shadow with both fixes, 0 of 6,913 incumbent and 0 of 46,702 FP8 calls differ.
On the live lane: KV 11,696,582, oracle 8/8, C1 439.3 tok/s (7.214 ms/step), watchdog ENFORCE.
Rollback: `~/dsv41/garner/promote/rollback-pg62-to-pg61.sh` (pg61-work untouched, hashes verified), then point
`promote/autostart-pg61.sh` and the Mac `dsv41_pg61_lane_check.sh` back (backups kept beside each).

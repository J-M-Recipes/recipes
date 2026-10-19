# 2026-10-18 — Card P1: prefill CUDA graphs to 8,192 tokens — **FAILED** (KV bar)

Question (idea from original-el8): capture CUDA graphs for prefill-sized batches so that 8K-token prefills stop running
eager, and cut the lane's 8K TTFT (0.65 s on the 0909 image vs 0.29 s on the M3G nightly lane). Lane under test: v22
(`dsv41-vllm-1017-v22-ksched1to8-BOUND-REF`, 0909 image, 1017 weights); window 04:53–05:41 CDT, CTL1 (live) → CG8K →
CG2K → CTL2 (restart). Nothing was promoted; the serving recommendation is unchanged.

| arm | `--cudagraph-capture-sizes` | graph capture | KV cache | boot |
|---|---|---|---|---|
| CTL (v22) | 1 … 128 (15 sizes) | 1.08 GiB actual / 1.29 GiB profiled | 5.26 GiB = **2,504,891 tokens** | hash hit `62426808…`, 316 s |
| CG8K | + 256 512 1024 2048 4096 8192 | 8.21 GiB profiled | **−1.6 GiB → `ValueError: No available memory for the cache blocks`** | died at 425 s, before autotune |
| CG2K | + 256 512 1024 2048 | 3.63 GiB profiled / 2.46 GiB actual | 2.98 GiB = **1,420,753 tokens (−1,084,138, −43%)** | **new autotune hash `1e1af343…`**, 15-min live tune, 1,380 s |

## Bars (pre-registered in `scripts/cardP1_runner.sh`; candidate vs mean of CTL1/CTL2)

| bar | threshold | result |
|---|---|---|
| WIN: TTFT 8K (RT C1 median, mean of 3) | ≤ 0.55 s (CTL 0.65) | not measured — benches skipped after the KV fail, by rule |
| WIN: KV loss | ≤ 0.3M tokens | **MISS** (CG2K −1.08M; CG8K no KV at all) |
| WIN: RT C1 / replay r2 | ≥ −2% | not measured |
| FAIL: KV loss | > 0.5M tokens | **HIT** |
| FAIL: decode −3% (RT C1/C4/C16) | | not measured |

**Verdict: FAILED** on the KV bar, both candidate lists. The CG2K boot is also PROVISIONAL (first boot on a new hash).

## Controls (v22, same window)

| arm | RT C1 (accept) | C4 | C16 | C32 out | TTFT 8K | cold ~53K TTFT | knee r2 C1/C8/C16 | replay r1/r2 | gauntlet |
|---|---|---|---|---|---|---|---|---|---|
| CTL1 (live) | 188.7 (2.97) | 326.8 | 537.4 | 511 | 0.64 | 2.17 s | 193/585/950 | 421/410 | 12/12 |
| CTL2 (restart) | 185.5 (2.95) | 321.5 | 519.4 | 491 | 0.66 | 3.10 s | 192/593/940 | 300/423 | 12/12 |

## Findings

1. **Graph memory grows super-linearly with the top capture size** (1.29 → 3.63 → 8.21 GiB at ≤128 / ≤2048 / ≤8192),
   and vLLM's CUDA-graph memory profiling (default since v0.21) charges it to the KV pool before allocation. On this
   lane 1 GiB ≈ 0.47M KV tokens, so the 5.26 GiB pool cannot pay for graphs much above ~512 tokens inside the 0.3M bar.
2. **The capture-size list is part of the FlashInfer autotune hash.** Any capture-list change is a new hash and a
   provisional first boot (~23 min here). The plan's "token-count sizes keep the hash" assumption was wrong.
3. The 8K TTFT gap to the nightly lanes is not explained by this card. Card P2f later measured 0.43 s on vLLM
   nightly-39983a5e with the same 15-size capture list (0.28 s with SWA bounded replay on), so the gap moves with the
   image at an unchanged capture list. Which part of the image is not isolated.
4. Reopen only bundled with an HBM source (`--max-model-len 524288` and/or a smaller KV dtype), capture cap 512–1024,
   bar "KV ≥ 2.0M and TTFT 8K ≤ 0.55 s", autotune hash seeded first.

## Also in this window: P5 long-generation cell (dry run)

`longgen_cell.py` (after original-el8's `long-gen.py`, Apache-2.0) on the v22 REF after the runner: `--max-tokens 2048
--target 0`, rc 0, 2,048 tokens, finish=length, TTFT 1.92 s, decode 219.2 tok/s, e2e 181.9, acceptance_len 3.52.
Receipts `receipts/longgen-dryrun-REF.txt`, `receipts/longgen-dry-REF.json`. The full 60K-token cell has not been run.

## Files

- `scripts/cardP1_runner.sh` (bars in the header), `scripts/cardP1_verdict.py`, `scripts/launch_CG8K.sh`, `scripts/launch_CG2K.sh`.
- `receipts/VERDICT.md`, `SUMMARY.txt`, `facts-CG8K.txt` / `facts-CG2K.txt` (graph-capture and KV lines from the boot logs), `facts-CTL*.txt`, `rt-CTL*/`, knee/knee6/replay JSON, `cold64k-*`, gauntlet summaries, `runner.log`.
- `throughput.csv` — control cells.
- Mirrored with a path scrub (box home → `~`/`$HOME`, LAN → host names). Boot/post logs stay on the box.

Runs: claude-opus-5-5 worker from Milo's brief. Write-up: Milo (Hermes, claude-fable-5-1). Idea: original-el8.

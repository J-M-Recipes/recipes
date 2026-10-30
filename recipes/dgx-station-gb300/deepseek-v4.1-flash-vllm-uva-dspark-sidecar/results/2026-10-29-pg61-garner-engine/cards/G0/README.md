# Card G0 — Garner's release as shipped, on our box (2026-10-29, measurement only)

**Verdict: REPRODUCES.** His published single-stream step rate reproduces on our GB300 + RTX PRO 6000 **Max-Q (300 W)**
within 0.4 %. The Max-Q is not a limit for his stack. Nothing adopted; `:30006` was dark 08:12–08:51 CDT and
h2-REF was restored with its watchdog (PPID 1, `trip: null`).

Same instrument on both arms: his `verify/bench_full.py` @ 8043a681, stages fixed (×10), gsm8k_c1, greedy_c1, streams 8/32.

| cell | A = h2-REF (ours) | G = Garner as shipped | G / A | his published |
|---|--:|--:|--:|--:|
| fixed coding prompt, tok/s | 435.4 (430–443) | **731.3** (702.5–762.0) | 1.68× | 733.6 |
| fixed, tokens/step | 4.57 | 5.39 | 1.18× | 5.43 |
| fixed, steps/s (→ ms/step) | 95.1 (10.5 ms) | **135.9** (7.36 ms) | **1.43×** | 135.4 (7.39 ms) |
| fixed, TTFT ms | 137 | 141 | | 138 |
| GSM8K T=1.0 thinking, 1 stream | 339.6 | **509.8** | 1.50× | 500.5 |
| GSM8K greedy thinking off, 1 stream | 403.9 | **591.4** | 1.46× | 588.6 |
| 8 streams aggregate | 1,147 | **1,404** | 1.22× | 1,373 |
| 32 streams aggregate | 1,561 | **2,382** | 1.53× | 2,524 |
| GPU KV cache tokens | 4,400,013 (8.5 GiB declared) | 11,372,409 | | 11,372,409 |

Boot gate on G: 27 fast-path components + `ds41_sampling`/`ds41_stop` ACTIVE, MEGA_PEER ACTIVE, LL_GEMM ACTIVE,
0 fallback lines. First boot with cold caches: sidecar + server ready in 12.7 min.

## Caveats that matter for reading the table

- **G is not under our policy.** `nvfp4_ds_mla` KV, MXFP4 indexer K, 512K context, forced T=1.0 / top_p 0.95 on
  thinking requests, `reasoning_effort: high` default, and three numerics-changing components. Card G1 measures his
  tree with those removed.
- The A arm's thinking-on cells ran at our server default effort; G ran at his default `high`. Tokens/step and step
  time are comparable; answer lengths are not.
- One instrument, one window, one rep of each timed window. The fixed-prompt row has 10 reps (IQR shown).

## Runner incidents (both fixed before G1)

1. `local arm=$1 … key=$R/$arm/key` in one `local` — `$arm` unset under `set -u`; runner died after ISOLATE with the
   lane up and guarded. Relaunched 38 s later. (`results-G0.fail1-unbound/`)
2. The runner's bind check called `/v1/models` without his API key (401) and would have failed a good boot. Runner
   stopped at 08:34:48, `g0_finish.sh` continued from "ready" with the key. G1 runs his server with no key.

## Files

`G0/A/bench.{json,md,log}`, `G0/G/bench.*`, `G0/G/boot.log`, `serve.log`, `sidecar.log`, `peer3_stats.json`,
`G0/runner.log`. Runner `g0_runner.sh` (sha1 at launch `1eb12acc`, then the `local` fix), `g0_finish.sh`.

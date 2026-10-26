# 2026-10-26-v27-t2-heldout: T0 + T2 + T1 (C1, C4) + BFCL held-out sibling on the live v27 lane

These are the standard-bench tiers from [`docs/benchmarking.md`](../../../../../docs/benchmarking.md), run against
the serving v27 lane (`dsv41-vllm-1017-v27-block-BOUND-REF` + `-rtx` on `:30006`, model `dsv41-flash-uva`). v27 is
v25m-kvtier with block verification added. The held-out BFCL sibling runs once at promotion, and this is that run.
The client was the M4 Max over the LAN. Nothing on the box was started, stopped or written; the lane kept serving
Forge guests throughout.

**Verdict, per the promotion bar in `docs/benchmarking.md`: PASS.** T2 shows no significant accuracy loss, the needle
stays EXACT 500/500, and T1 is not below the band. The held-out sibling also passes the recipe's own
`bfcl_delta_pt_min = −1.0` bar. **The recipe stays `experimental`:** the 48 h soak is still running, T3 has not run,
and no temperature > 0 quality instrument has run on v27 (see Caveats).

## Clock (CDT, `campaign.log`; UTC = CDT + 5 h)

| step | start | end |
|---|---|---|
| runner START / T0 | 17:08:15 | 17:08:16 |
| T2 needle-checksum | 17:08:16 | 17:09:50 |
| T2 GSM8K-200 | 17:09:50 | **17:10:52** (= 2026-10-26T22:10:52Z, the quality-gate stamp) |
| tool-eval (92 scenarios) | 17:10:52 | 17:12:37 |
| T1 C1/C4 × 0/32K/128K | 17:12:37 | **17:16:44** (= 2026-10-26T22:16:44Z, the performance-gate stamp) |
| BFCL held-out sibling | 17:16:44 | 17:20:41 |

Bench wall time was 12 min 26 s. The prep started at 17:07:46, and `RUN.md` was written at 17:25.

## Tools and commands

- llm-inference-bench v0.7.7 @ `c71ec1f2b34a4f1c8f702f1750ccd70da0e389e2`, with `LLM_BENCH_NO_UPDATE_CHECK=1
  --display-mode plain` and no upload. tool-eval-bench was at `570951a77e88` (the bench default pin). See
  `bench-version.txt`.
- `bfcl_gate.py` sha1 `839e69c6` (identical to the box copy). The BFCL data was copied read-only from the box, and
  every entry in `SHA256SUMS` checked OK (`bfcl-data-SHA256SUMS.txt`). The data files themselves are third-party and
  not republished.
- Exact commands: `command.txt`. Runner: `run_all.sh`. T0: `t0.py`. Host: `<gb300>`.

## Flags (deviations from the repo standard in bold)

| check | flags |
|---|---|
| needle | `--test-profile needle-checksum` **`--profile-concurrency 4`** (profile default C8); T=0, max_tokens 4096, 500 requests |
| GSM8K | `--test-profile gsm8k --profile-runs 200` **`--profile-concurrency 4`** (default C30) `--compare-baseline <baseline>`; T=0, max_tokens 131072, reasoning_effort = template default (not set) |
| tool-eval | `--tool-eval` **`--tool-eval-parallel 4`** (= leaderboard default); hardmode, seed 42, T=0; the full 92 scenarios rather than the "optional subset", so it compares to the v21 bundle |
| T1 | **`--concurrency 1,4`** (doc: 1,4,8; **C8 skipped** for guests) `--contexts 0,32768,131072 --duration 30` |
| BFCL held-out | `SUITE=heldout` **`CONC=4`** (card runners use 8) `MAXTOK=2048 THINKING=0` |

**Concurrency was lowered to 4 everywhere, to protect guests** on the live lane. At T=0 the scores do not depend on
concurrency in principle; the latencies do.

## Baseline

The GSM8K baseline is [`../2026-10-24-gv-v25m-gates/gsm8k1319-V.json.gz`](../2026-10-24-gv-v25m-gates/gsm8k1319-V.json.gz):
Card GV on the live v25m lane, 2026-10-24 05:44, the full 1,319 items at T=0 and C30, 1277/1319 = 96.82%. It is the
newest GSM8K json under either DSV4.1 recipe. A gunzipped local copy was passed to `--compare-baseline`, and the pairing
uses the 200 overlapping item ids. v25m(-kvtier) is the lane v27 replaced, and v27 is that lane plus block
verification. The BFCL held-out reference is the same card's `bfcl-V-heldout.jsonl.gz` (1018/1311 = 77.65%).

## Per-check table

| check | v27 result | baseline / bar | verdict | raw |
|---|---|---|---|---|
| T0 `/v1/models` | `dsv41-flash-uva` | id present | PASS | `t0.json` |
| T0 exact-word chat | `PINEAPPLE` | exact | PASS | `t0.json` |
| T0 tool call | 1 call `get_weather {city: Austin, unit: f}`, finish `tool_calls` | parsed call | PASS | `t0.json` |
| needle-checksum | **EXACT 500/500** (0 near-miss / wrong / error) | EXACT N/N | PASS | `t2-needle.json.gz` |
| GSM8K-200, paired | **192/200 = 96.0%** (Wilson 92.3–98.0); 0 unparseable, 0 hit max_tokens | baseline 192/200 on the same ids; flips 0 vs 0; **McNemar exact p = 1.0**; completion tokens mean 238.6 vs 256.9 (×0.93) | PASS (a p < 0.05 loss fails) | `t2-gsm8k200.json.gz` |
| tool-eval, hardmode, 92 scenarios | **89/100**, 163/184 points; pass 77 · partial 9 · fail 6; fails TC-43, 51, 68, 74, 89, 92; the safety gate did not pass (it did not on v21 either) | the doc sets no bar; the v21 GB300-only bundle scored 92/100, 169/184, failing TC-68/74/89/92 | informational: **TC-43 and TC-51 are new failures** | `t2-tooleval.json`, `t2-tooleval.tool-eval/` |
| T1 | see the table below; all 6 cells are above the v21 band | within the run-to-run band of the last promoted run | no regression | `t1-perf.json` |
| BFCL held-out sibling | **1013/1311 = 77.27%** (live_simple 202/258 = 78.29%, live_multiple 811/1053 = 77.02%), 0 errors, 238 s | v25m Card GV 1018/1311 = 77.65% → **−0.38 pt**; paired 17 lost / 12 gained, McNemar p = 0.458; bar `bfcl_delta_pt_min` −1.0 | **PASS** | `bfcl-v27-heldout-summary.json`, `bfcl-v27-heldout.jsonl.gz` |

### T1 (llm-inference-bench synthetic filler, `ignore_eos`, 30 s cells), aggregate tok/s

| ctx | C1 v27 | C1 v21 band | C4 v27 | C4 v21 band |
|---|--:|--:|--:|--:|
| 0 | **262.1** | 203.5–214.3 | **661.1** | 344.5–359.3 |
| 32K | **254.1** | 173.7–196.0 | **642.5** | 367.7–410.4 |
| 128K | **249.7** | 183.8–190.2 | **636.7** | 374.0–376.3 |

Every cell had 0 errors and no loop flags; `max_running_reqs` equalled the target concurrency. The server-side
acceptance length was 2.68–2.76 at C1 and 2.48–2.55 at C4. **C8 was skipped.** The v21 band is run 1
([`../../../deepseek-v4.1-flash-vllm-uva-dspark/results/2026-10-05-lib-t1-v21/`](../../../deepseek-v4.1-flash-vllm-uva-dspark/results/2026-10-05-lib-t1-v21/))
and run 2 (the T3 matrix in
[`../../../deepseek-v4.1-flash-vllm-uva-dspark/results/2026-10-06-lib-t2t3-v21/`](../../../deepseek-v4.1-flash-vllm-uva-dspark/results/2026-10-06-lib-t2t3-v21/)).

## Caveats

- **Everything here is at temperature 0. Block verification acts only at temperature > 0** (vLLM af7f9488 skips it
  on greedy rows). These checks cover the target path, not the block key itself: needle, GSM8K, BFCL and tool-eval
  measure what v25m-kvtier already served. A temperature > 0 quality instrument has not run on v27. That could be
  full-vocabulary KLD over forced tokens, as the BRRR card does, or seeded GSM8K at T=1, B vs A.
- **Concurrency was lowered to 4 for guests** (see Flags). The lane served Forge guests during the run. `engine-load.tsv`
  samples `/metrics` every 5 s (147 samples): running never exceeded 4, our own cap. Waiting was 0 except one
  sample of 1 at the 17:10:52 GSM8K → tool-eval handoff. /metrics cannot attribute requests, and vLLM `Running:` log
  lines were not captured (that needs `docker logs`, which was not used).
- **The T1 band is the v21 GB300-only lane** (0909 image, Grace offload). It is the only lib T1 on record; no lib T1
  existed for the sidecar lane until this run. **Above the band therefore means "no regression" only.** It is not a
  measured speedup over a sidecar baseline. This run becomes the sidecar lane's first T1 reference.
- The tool-eval comparison is also cross-lane (v21 GB300-only, a different image and effort mapping). The doc sets no
  tool-eval bar. TC-43 and TC-51 are recorded as new failures, not as a gate result.
- The T1 cells use synthetic filler with `ignore_eos`. They do not replace the recipe's real-text C1 headline.
- `analysis.txt` is the operator's scratch analysis, published as-is. Its `tool-eval scenarios 0 0` / `changed: []`
  lines are a parser miss: the tool-eval counts in this README come from `t2-tooleval.json`
  (`tool_eval_raw.scores.scenario_results`).
- One read-only `docker ps` ran on the box during discovery: the v27 pair had been up ~37–38 min at ~17:05 CDT.

## Files

`RUN.md` (the operator's run notes), `command.txt`, `run_all.sh`, `t0.py`, `campaign.log`, `bench-version.txt`,
`engine-load.tsv`, `analysis.txt`, `t0.json`, `t1-perf.json` + `.stdout.txt`, `t2-needle.json.gz` + `.stdout.txt`,
`t2-gsm8k200.json.gz` + `.stdout.txt`, `t2-tooleval.json` + `.stdout.txt` + `t2-tooleval.tool-eval/` (run.json,
progress.jsonl, the markdown report; the bench's sqlite cache is omitted, as in the v21 bundle),
`bfcl-v27-heldout-summary.json`, `bfcl-v27-heldout.jsonl.gz`, `bfcl-v27-heldout.log`, `bfcl-data-SHA256SUMS.txt`.

The raw json/jsonl files that carry model outputs are gzipped with `gzip -n`, like Card GV's, with bytes preserved
after the scrub. Per-item content matches what Card GV already publishes for these instruments. GSM8K and needle
carry the bench's own run records. BFCL keeps id / ok / call / content per item, and none of the dataset's prompt
text. The scrub was 43 replacements in this bundle: the Mac home directory → `~` (22) and the Station's LAN
address → `<gb300>` (21). The other `192.168.x.x` strings inside `bfcl-v27-heldout.jsonl.gz` are arguments the model generated
from BFCL's own prompts, not lab hosts.

# 2026-10-24 — v26 "Seeded Adaptive" (Card AR6 v2): PROMOTED ~21:30 CDT, 48 h soak waived

One DGX Station GB300, lane `:30006`, Grace-only (no sidecar GPU). v26 = the recipe's v23 launch (0909 image, unchanged
serve flags) + hook **v26ar.2** + the **seed boot map** `rowmap-mixed-ar-v1.json` + the shipped adaptive policy, adaptation
unfrozen in production. James promoted it on 2026-10-24 at ~21:30 CDT and **explicitly waived the 48 h soak**.

Card AR6 v2 verdict: **MIXED — `AR6-H1PASS-H2FAIL`** ([`cardAR6/VERDICT.txt`](cardAR6/VERDICT.txt),
[`cardAR6/verdict.json`](cardAR6/verdict.json)). H1 (seed map) passed. H2 (more aggressive swap queue) failed. The
promoted config is the H1 arm with the shipped policy.

Write-up: [Adaptive Residency, Part 3: The Seed Was the Prize](https://al-engr.com/gb300-adaptive-residency-seed-map.html).

## Provenance

| | pin |
|---|---|
| image | [`image.txt`](image.txt) — `vllm/vllm-openai:deepseekv41-flash-0909` @ `sha256:00d577a6…` (digest from 2026-10-18, not re-read in AR6; version line `0.1.dev20904+g179dd0fa9` re-read in `cardAR6/boot-A.log`) |
| model | [`model.txt`](model.txt) — `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| hook | [`hook.txt`](hook.txt) — v26ar.2 sha1 `69bf0777` hook / `7ea55cbd` e4_adaptive / `e65a93d7` sitecustomize (source not staged: v26ar.2 = the published [v26ar.1](../2026-10-23-ar-adaptive-residency/hook/) + env-gated telemetry); seed map [`hook/rowmap-mixed-ar-v1.json`](hook/rowmap-mixed-ar-v1.json) sha1 `52e2861c`, sha256 in [`hook/SHA256SUMS`](hook/SHA256SUMS) |
| exact launch | [`scripts/launch_ar6.sh`](scripts/launch_ar6.sh) with `ROWMAP=rowmap-mixed-ar-v1.json`; diff vs the published v26ar.1 launcher: [`launch_ar6-vs-launch_ar.diff`](launch_ar6-vs-launch_ar.diff) (adds `PIN_ROWMAP=/w/$ROWMAP`, `PIN_STEP_RING=65536`, `PIN_SWAP_CUDA_EVENTS=1`, hook dir `v26ar2`) |
| boot | hash hit (`Loaded 189 configs`, hash `62426808…`), boot→ready 474 s on both boots, `GPU KV cache size: 2,495,694 tokens` (2.38× at 1M), rehome `HBM_expert=206.61GiB pinned_expert=62.33GiB host_avail=197.2–197.3 GiB` |
| runner / verdict | [`scripts/cardAR6_runner.sh`](scripts/cardAR6_runner.sh), [`scripts/ar6_verdict.py`](scripts/ar6_verdict.py) (stdlib only); the pass/mixed/parity harness files (`ar6_pass.py`, `ar_shift_v2.py`, `ar_mixed_v2.py`, `e2c_parity.py`) were not staged — their sha1s are in [`cardAR6/instruments.txt`](cardAR6/instruments.txt) |

Production start sequence: launch with the seed map, `touch $RUN/GO`, then write `{"freeze": false, "baseline": true}`
to `$RUN/ctrl.json` (the hook boots frozen and only counts until then). Shipped policy = drain every 64 steps, ≤ 2 swaps
per layer per drain, EWMA half-life 500 steps, hysteresis 1.5×, min 8, ≤ 8 swaps per barrier.

## Design

Two boots, same window, same harness:
- **Boot 0 / Arm 0:** September static map `rowmap-static-v1.json`. Control.
- **Boot A / Arm A:** seed map `rowmap-mixed-ar-v1.json`. This is the top-295 experts per layer from one cumulative
  live-counter dump on 2026-10-23, with the same 295/89 geometry.

Each boot ran:
1. A **frozen held-out** mixed C1 run, 180 s, alternating JP/DE essays and agent-English prose. These prompts are
   held out from the map fit.
2. Then **shift passes**: agent 90 s → JP/DE 90 s → agent 90 s, adaptation unfrozen:
   - **B0** = shipped policy (control).
   - **B1** = drain every 32 steps (2× queue rate).
   - **B2** = drain every 16 steps (4×).

   B cells are interleaved over 3 rounds. Each pass starts from `restore_map`.

Instruments are defined in `ar6_verdict.py`. Primary is `t90_cold`: 2 s wall windows of the cold share of expert
picks, measuring time to clear 90% of the peak→converged excess.

## Results

H1 — frozen maps, held-out prompts, same window ([`throughput.csv`](throughput.csv)):

| C1, adaptation frozen on both | static map | seed map | Δ |
|---|---:|---:|---:|
| mixed aggregate (post-60 s) | 127.12 | **183.73** | **×1.445** |
| JP/DE per-request median | 85.45 | 177.23 | ×2.07 |
| agent per-request median | 202.74 | 201.36 | −0.7 % |
| first JP/DE request after boot | 84.96 | 177.89 | ×2.09 |

Adaptive steady state: converged C1 over the final 60 s, pooled reps:

| boot / cell | JP/DE | agent | churn |
|---|---:|---:|---:|
| static boot, B0 shipped | 189.94 | 219.13 | 107.2 /min |
| **seed boot, B0 shipped (= v26)** | **189.59** | **218.94** | 95.3 /min |
| seed boot, B1 2× queue | 189.62 | 218.21 | 102.8 /min |
| seed boot, B2 4× queue | 189.52 | 217.92 | 112.3 /min |

Both boots converge to the same steady state. The seed changes where you start, not where you end.

Convergence: cold-share T90 in 2 s windows, mean of reps:

| shift | static B0 | seed B0 | seed B1 | seed B2 | peak cold share, static → seed |
|---|---:|---:|---:|---:|---|
| agent → JP/DE | 11.0 s | **8.7 s** | 9.3 s | 10.0 s | 25.4 % → 5.7 % |
| JP/DE → agent | 14.0 s | **14.0 s** | 14.0 s | 14.0 s | 6.0 % → 3.3 % |

Converged cold share is 0.3–0.5 % per shift (0.4 % pooled) on every cell. On the seed boot, cold share is ≤ 1 % at
8.7 s and 14.0 s. **H2 failed:** neither aggressive cell halved T90 on both gated shifts. The T90 ratio vs same-boot B0
was ×1.00–1.15. The secondary `<2×conv` metric moved ×0.43–0.93 at a 3–6 % peak and is not counted. **Why:** the seed
map collapses the shift peak from 25 % to 3–6 %, so the swap queue the aggressive policy was built to drain never
saturates. The shipped policy stays.

Swap cost: first measurement, step ring + CUDA events. Gate G2 failed on every cell, including the control, so this is
a property of the mechanism:

| | static B0 | seed B0 | seed B1 | seed B2 |
|---|---:|---:|---:|---:|
| swap-bearing / swap-free host step, p95 ratio | 1.91 | 1.91 | 1.89 | 1.90 |
| excess ms per applied swap | 1.29 | 1.58 | 1.92 | 2.43 |
| CUDA row-copy ms (mean) | 0.27 | 0.26 | 0.26 | 0.27 |

Whole-boot host-step interval ([`p99-overall.txt`](p99-overall.txt), [`scripts/p99_overall.py`](scripts/p99_overall.py)).
The seed boot logged 267,226 consecutive host steps. **2.53 % were swap-bearing.** p99 of all steps was **22.17 ms**,
against 19.52 ms for swap-free steps alone, and p50 was 11.06 ms. These steps come from shift-heavy test traffic,
which is worst case. This is a host-step (verify-window) interval, not a client inter-token gap.

Quality:
- **Gauntlet 12/12** on the fixed seed map: [`cardAR6/gauntlet-ArmA/`](cardAR6/gauntlet-ArmA/).
- **Gauntlet 12/12 live under adaptation** on the promoted lane: [`gauntlet-soak1/`](gauntlet-soak1/), tag
  `v26seed-soak1`.
- **Greedy parity 49/50**, static Arm0-S1 vs seed ArmA-F1. The miss is `agent-tool-1`, the documented fixed-map noise
  prompt. It also misses S1 vs S2 and F1 vs F2 on a single map.
- **0 server errors** and **0 stale swaps** on every cell.

## Not in this bundle / not measured on v26

- **48 h soak: waived** at promotion (James, 2026-10-24). The AR5 soak ran ~19:57 → ~20:4x CDT before James cancelled
  it. The longest unattended window so far is about 1 h plus the battery traffic above.
- **Long context on v26:** no 128K or 1M request is in these receipts. Boot KV is 2.38× at 1M. The nearest 1M-class
  receipt is the sidecar lane's Card KV3 950K needle (954,028 prompt tokens, recalled, 32.8–33.0 s). It is a different
  lane: see the sidecar recipe.
- **Not re-run on v26:** BFCL, GSM8K, teacher-forced Δlogprob. Those rows stay on v23.

## Scrub notes

- Box home-directory paths → `~` in data and logs, `$HOME` in scripts. The GB300's LAN address → `<gb300>`.
- Other LAN addresses → `<lan-host>`. These were a Prometheus `/metrics` scraper in the vLLM access logs. No chat
  traffic came from it: `Running:` never exceeded 1 during cells.
- One parity prompt's model output contained a ping/ssh command with the GB300's LAN IP, which was scrubbed.
- All prompt pools are synthetic and kept: JP/DE/zh essays, math prose, agent-English prose, and gauntlet fixtures.
- These files are `gzip -n`, with bytes preserved: files ≥ 250 KB (`steps.bin`, `swaps.jsonl`, `telemetry.jsonl`, the
  shift jsons, the npz counter dump, one access log). `gunzip` before re-running `ar6_verdict.py`. `p99_overall.py`
  reads the `.gz` directly.

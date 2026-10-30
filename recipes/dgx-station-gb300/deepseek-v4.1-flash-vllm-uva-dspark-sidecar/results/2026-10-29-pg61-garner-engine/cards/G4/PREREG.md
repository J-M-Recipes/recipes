# PREREG — Card G4: T3 promotion bench for Pg61 (written 2026-10-29 before the first G4 boot)

Candidate **Pg61**, booted by `launcher/launch-pg61.sh up` from a fresh `build` (Garner 8043a681 + patches 01–05 +
overlay-vllm; device code == G3-tested, cubins 11b55da2/4d07c093). Control **A** = h2-REF (the serving lane),
restored in the same window. Both isolated; M4 (<m4>) allowed through. llm-inference-bench v0.7.7 @ c71ec1f2,
tool-eval-bench 570951a7, same flags on both arms, C ≤ 8.

Suite per arm (M4 over the LAN): T0 · needle-checksum 500 (C4) · GSM8K full 1319 (C4, `--compare-baseline`
v25m 1319) · MMLU-Pro 1000 stratified (C4) · T1 `--concurrency 1,4,8 --contexts 0,32768,131072 --duration 30` ·
tool-eval hardmode ×2 (parallel 4).

Bars (Pg61 vs A, paired by item id; a FAIL blocks promotion):
1. needle-checksum EXACT 500/500 on Pg61.
2. GSM8K-1319: no paired McNemar loss at p < 0.05, vs A (same day) and vs the v25m baseline.
3. MMLU-Pro-1000: no paired McNemar loss at p < 0.05 vs A.
4. T1: Pg61 ≥ A on every C1 and C4 cell (≥ 0.97× allowed on a single cell as band). C8 reported, no bar.
5. Launcher reproduces G3: `docker inspect ds41` Args identical to G3 Pg61 `args.json`; in-container hashes OK.
6. No p3 watchdog trip, server up after the suite.
Reported, no bar: tool-eval ×2 per arm (scores + fail TC ids), prefill (known −8 %, rowmap sweep after promotion).
Not exercised: temperature > 0 quality (all T = 0), > 128K decode (1M needle ran in G1f).

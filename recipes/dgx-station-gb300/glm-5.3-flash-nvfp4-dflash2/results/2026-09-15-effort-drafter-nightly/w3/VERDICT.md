# W3 verdict — `SGLANG_OPT_FUSED_KDA_VERIFY=1` on 8874c51a

Window: 2026-09-15T19:56:25Z → 20:07:03Z. Axis engaged. Stop+keep `glmf-w3-fusedverify` on `:30001`. Do not restore W2 unless told. Do not carry this env into later windows.

## Gates

| Gate | Bar | Result |
|---|---|---|
| Axis in boot log | `KDA fused chain-verify kernel enabled (topk==1 path).` | **yes** (19:58:50Z) |
| C1 recipe-method low | ≥ 212.8 (W2 202.7 × 1.05) | **fail** 200.5 (−1.1% vs W2) |
| Accept rate low | 0.40–0.44 (W2 0.42 ± 0.02) | **pass** 0.42 (len 3.52 vs W2 3.50) |
| Greedy vs W2 | identical ≥ W2-vs-W2 noise (20/20) | **fail** 1/20 |
| Tools ×2 | 10/10 | **pass** 10/10, 10/10 |

**Close the axis.** Divergence is not ~1 ulp reduction-order noise; 19/20 prompts first-diverge as early as char 4. Speed did not move. Leave fused-verify off for W4+.

## Numbers

C1 recipe-method history-essay 512, median of 3, effort=low:

- W2: 202.7 (201.6 / 202.7 / 202.8)
- W3: 200.5 (192.7 / 201.3 / 200.5)

C1 same, effort=max:

- W2: 265.3 (tight)
- W3: 244.0 (227.8 / 244.0 / 257.7) — noisier, not a win bar

Accept MEAN low: W2 0.42 / W3 0.42. Accept MEAN max: both 0.40.

C8 (void first pass, 107% spread, leaked 129): rerun after rewarm **635.1 agg / 79.39 per-stream** (654/626/625, spread 5%). W2 had no C8 row.

Greedy: W2-vs-W2 20/20; W2-vs-W3 1/20 (only prompt 6 identical). Receipt: `greedy-w2-vs-w3.txt`.

## Launch (one axis)

Same W2 knobs + `DOCKER_ENV='-e TORCHINDUCTOR_COMPILE_THREADS=1 -e SGLANG_OPT_FUSED_KDA_VERIFY=1'`. Image `lmsysorg/sglang:nightly-dev-cu13-20260915-8874c51a`. Drafter still 7d74 on `/models/GLM-5.3-Flash-DFlash2`. `max_running_requests` still capped at 9 by mamba 48.

## Follow-up (not done)

Handoff stop: file upstream with this receipt if we want #39219/#36821 authors to see DFlash2+NVFP4 greedy 1/20. Not filed from this run.

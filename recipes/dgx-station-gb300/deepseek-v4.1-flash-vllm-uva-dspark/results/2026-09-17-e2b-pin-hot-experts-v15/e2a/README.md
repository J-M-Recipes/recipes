# E2a — count-only routing profile

Worker: grok-4.6 / xai-oauth. Same-window pair on quiet GB300. GLM big stayed stopped-and-kept. `:30003` untouched. Autotune hash **hit** (`9ac7b387`, Loaded 210 configs, 0 new). E2A candidate left **up** on `:30006`. v14 REF stopped-and-kept.

CAT_ORDER (both boots): `prose structured code shell_ops tool_json`.

## Boots

| | container | bind | hash |
|---|---|---|---|
| control | `dsv41-vllm-v14-1M-ksched-agent-BOUND-REF` (`docker start`) | 277 s | 9ac7b387 / 210 / 0 new |
| candidate | `dsv41-vllm-E2A-count-EXP` (v14 args + `PIN_MODE=count` hook) | ~8 min (graphs slower with scatter_add) | 9ac7b387 / 210 / 0 new |

Hook: `TrtLlmMxfp4ExpertsModular._invoke_kernel`. Graph-safe `scatter_add_` into per-layer int64[384]; `-1` weighted 0. Decode = capturing **or** T≤16; else prefill. Dump eager/sentinel/atexit only (no sync in graph body). Layer index = first-seen `id(self)`: 40 main MoE (equal decode mass) + 3 DSpark nextn (lower mass).

## Knee (verdict instrument)

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| control | 91.65 | 320.7 | 435.9 |
| cand pass1 | 88.27 | 307.4 | 416.5 |
| cand pass2 | 88.40 | 309.1 | 420.9 |

Hook overhead C1 = **−3.69%** (88.27 vs 91.65). Bar was ≥ control − 1.5% (floor 90.28). **FAIL.** Pass2 confirms (88.40). Not day-drift: control runs 91.5/91.8, cand 88.1/88.5. Agent-fixture accept rates bit-identical at 59.6% — routing unchanged; the 3.7% is the count ops inside the graph.

## Other instruments

| | control | cand p1 | cand p2 |
|---|--:|--:|--:|
| agent prose tok/s | 100.6 | 96.1 | 96.8 |
| agent weighted accept | 59.6% | 59.6% | 59.9% |
| T5 prose fixture | 93.6 | 90.2 | 90.2 |
| replay n=4 r1/r2 | 197.7 / 239.8 | 190.8 / 224.1 | 226.1 / 231.7 |

Replay swing ~20%/run is the known instrument; direction matches the knee.

## Profile (decode, layers 0–39, dump `counts-20260917-183654.npz`)

- decode selections (40×384): 34.38M (143243 token-rows/layer ≡ ≫ 20k decode tokens)
- own-cold (89/384) share: **1.55%** (bar ≤ 3% in-domain) PASS
- LORO (pass1 cold → pass2 / swapped): 1.60% / 1.68%
- T5 overlap: **68.2% mean** (min 52.8 / max 78.7) — not < 50%, but own cold is tighter than T5's cold on this run (T5-cold captures 3.76% of this decode vs 1.55% own)
- Spearman vs T5 decode ranks: mean 0.807 (min 0.697)

**E2b must use this rowmap, not T5.** Do **not** leave `PIN_MODE=count` on for E2b — counting is not free. Strip the hook or gate it off the hot path (e.g. dump-only via `--enable-return-routed-experts` is worse; a no-op pass-through was not measured this window).

## Left running

`dsv41-vllm-E2A-count-EXP` on `:30006` (served `dsv41-flash-uva`, 1M). REF exited 0, kept.

## E2a VERDICT: hook overhead −3.69% · hash hit yes · own-cold share 1.55% · T5 overlap 68.2% · profile ready yes (rowmap); count-hook not free

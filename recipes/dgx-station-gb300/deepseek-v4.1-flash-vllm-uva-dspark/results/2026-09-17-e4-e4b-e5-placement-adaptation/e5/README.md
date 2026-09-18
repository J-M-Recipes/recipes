# E5 — offline adaptation (mixed-domain rowmap v2)

Worker: **grok-4.6 / xai-oauth**. GLM stayed stopped-and-kept. `:30003` untouched. Never `docker rm`. v15 left **up** on `:30006`.

One axis: `PIN_ROWMAP` only (`rowmap-mixed-v2.json` vs `rowmap-static-v1.json`). Same image, same launch args, 295 hot / 89 cold per layer. Autotune hash-hit `9ac7b387` Loaded 231 / 0 new on every boot.

## Why

v15 (agent-profiled static map) is +73% C1 over positional but a −1.5% wash on JP/DE. On-line adaptive is PARKED (E4b counter 1.68%). E5 = profile offline on mixed traffic, rebuild the map, hash-hit reboot.

## Profile (count mode on existing `dsv41-vllm-E2A-count-EXP`)

Stop-and-keep v15. `docker start` E2A. Hash-hit 9ac7b387 Loaded 231 / 0 new / 4.4 s. Zero dump `counts-20260918-010309.npz` (eager long-prefill; L=43 sel_dec=106887) **before** traffic. Final dump `counts-20260918-011301.npz` (sel_dec=32847552). Rowmap from decode, layers 0–39, final−zero.

Traffic (2 full passes of agent_fixture / T5 prose / replay n=4 / 10 new math / 20 new JP+DE+FR+ES, then extra agent+replay to rebalance). None of the 10 `e2c_heldout.py` prompts appear in the profile set (`scripts/e5_profile_prompts.json`).

| bucket | completion tokens | share |
|---|--:|--:|
| agent_fixture + replay | 27015 | 50.05% |
| T5 prose + math | 11214 | 20.78% |
| non-English | 15742 | 29.17% |
| **total** | **53971** | 100% |

Decode (E5 window): 31.84M selections, 132670 token-rows/layer.

| | |
|---|---|
| own-cold share (v2 map on E5 counts) | **2.11%** |
| v1-cold share on same E5 counts | **11.74%** |
| cold churn vs v1 | **24.5 / 89** mean (min 14, max 33) |
| hot ∩ hot | 270.5 / 295 |
| Spearman v1-profile vs E5-profile ranks | 0.742 mean (0.667–0.828) |

v1's agent cold set is a 12% miss on mixed traffic; v2's own-cold is 2%. Geometry unchanged (40 × 295/89).

## Boots

| | container | bind | autotune |
|---|---|---|---|
| E5 candidate | `dsv41-vllm-E5-pin-mixed-v2-EXP` (now `-C1-10PCT-FAIL`) | ~8 min first / ~308 s rebind | 9ac7b387 Loaded 231 / 0 new / ~4.6 s |
| v15 control | `dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` (`docker start`) | 308 s | 9ac7b387 Loaded 231 / 0 new / ~5 s |

Rehome: layers=40 HBM_expert=**206.61 GiB** pinned_expert=**62.33 GiB**. CAT_ORDER `prose structured code shell_ops tool_json`. Sequence **E5a → v15 → E5b**.

## Knee (verdict instrument)

C1/C8/C16 = mean of two `knee.sh` invocations per window.

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| E5a r1/r2 | 138.59 / 138.73 | 584.79 / 596.59 | 674.23 / 777.44 |
| v15 r1/r2 | 153.02 / 153.20 | 699.79 / 696.73 | 808.05 / 945.84 |
| E5b r1/r2 | 136.80 / 137.02 | 576.54 / 588.47 | 669.56 / 774.64 |
| two-window E5 mean vs v15 | **137.79 vs 153.11 (−10.01%)** | **586.60 vs 698.26 (−15.99%)** | **723.97 vs 876.95 (−17.44%)** |

Bar C1 ≥ −1.5%; C8/C16 ≥ −3%. **FAIL.**

## Agent fixture

| | E5a | E5b | E5 mean | v15 | Δ |
|---|--:|--:|--:|--:|--:|
| prose | 163.0 | 167.6 | 165.3 | 168.7 | **−2.02%** |
| structured | 200.9 | 199.3 | 200.1 | 203.7 | −1.77% |
| code | 216.3 | 213.6 | 214.9 | 211.1 | +1.82% |
| shell_ops | 240.3 | 240.7 | 240.5 | 247.8 | −2.95% |
| tool_json | 194.3 | 215.6 | 204.9 | 220.9 | **−7.22%** |
| weighted accept | 59.58% | 59.58% | | 59.58% | identical |

Bar: no category worse than −3%. tool_json **FAIL**. Accept rates bit-identical — tax is residency, not routing. tool_json is also the known fixture-order jitter class (E2c); same CAT_ORDER here.

## Held-out (streaming first→last, 5×400 tok; not in the profile set)

| | E5a | E5b | E5 mean | v15 | Δ |
|---|--:|--:|--:|--:|--:|
| math_proof | 201.34 | 199.45 | 200.39 | 181.48 | **+10.42%** |
| non_english JP/DE | 146.32 | 144.94 | 145.63 | 88.86 | **+63.89%** |

JP/DE bar ≥ +1.5% (target ≥ v14's 93.0). **PASS** — 145.6 vs v14 93.0 is +56% over positional-era held-out too. Math bar ≥ −1.5%. **PASS**.

## Parity (greedy T=0, seed 42, max_tokens 256, 18 prompts)

8 agent-fixture + 5 held-out math + 5 held-out JP/DE. Content + tool name/args.

| pair | exact |
|---|--|
| E5a vs v15 | **18/18** |
| E5b vs v15 | **18/18** |
| E5a vs E5b | **18/18** |

Bit-exact as contracted (both F4 split vs full-E).

## Host watch (cgroup `memory.current`)

| window | start GiB | end GiB | Δ GiB | MemAvailable |
|---|--:|--:|--:|--:|
| E5a | 348.689 | 348.707 | +0.018 | 125 |
| v15 | 351.911 | 351.957 | +0.046 | 125 |
| E5b | 351.871 | 351.917 | +0.046 | 124 |

No window grew > 2 GiB.

## Left running

`dsv41-vllm-v15-1M-pin-static-v1-BOUND-REF` on `:30006`. E5 stopped-and-kept as `dsv41-vllm-E5-pin-mixed-v2-EXP-C1-10PCT-FAIL`. GLM / `:30003` down.

## Read

The mixed map does what it was asked on the held-out domain (JP/DE −1.5% wash → +64%). It pays for that on the agent-shaped knee (C1 −10%, C8 −16%, C16 −17%). Offline adaptation is real and cheap (count-mode profile + hash-hit reboot); a 50/25/25 mix is **not** a free overlay on the agent lane. Next map, if any, needs a higher agent weight or a per-lane rowmap, not a thicker mix.

## E5 VERDICT: JP/DE Δ+63.89% · math Δ+10.42% · C1 Δ−10.01% · C16 Δ−17.44% · parity 18/18 · cold churn 24.5/89 mean · hash hit yes · left up: v15

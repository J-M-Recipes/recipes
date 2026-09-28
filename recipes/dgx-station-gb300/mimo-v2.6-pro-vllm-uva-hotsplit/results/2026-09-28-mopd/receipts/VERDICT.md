# MiMo-V2.6-Pro-MOPD run — verdict (bars from harness/protocol-v2.yaml)

| bar | result | detail |
|---|---|---|
| Q1 residency m-ctrl->m-hot: TF ppl within 0.5% rel | **PASS** | ppl 1.5062 -> 1.5062 (0.006%) |
| Q1 residency m-ctrl->m-hot: TF top-1 flips <= 2.0% | **PASS** | 951/80384 = 1.183% · mean|Δ| 0.02506 p99 0.4049 max 4.195 · docs 39 |
| Q1 residency BFCL dev mhot >= mctrl - 1 pt | **PASS** | mctrl 563/600 = 93.83% · mhot 559/600 = 93.17% · Δ -0.66 pt · no-call 4/4 · fail kinds {'value': 32, 'no_call': 4, 'name': 1} / {'value': 36, 'no_call': 4, 'name': 1} |
| errors mctrl dev <= 0.5% | **PASS** | 0/600 · wall 918 s |
| errors mhot dev <= 0.5% | **PASS** | 0/600 · wall 800 s |
| Q1 residency BFCL heldout mhot >= mctrl - 1 pt | **PASS** | mctrl 1028/1311 = 78.41% · mhot 1032/1311 = 78.72% · Δ +0.31 pt · no-call 45/39 · fail kinds {'value': 178, 'no_call': 45, 'missing': 29, 'name': 31} / {'value': 178, 'no_call': 39, 'missing': 30, 'name': 32} |
| errors mctrl heldout <= 0.5% | **PASS** | 0/1311 · wall 2049 s |
| errors mhot heldout <= 0.5% | **PASS** | 0/1311 · wall 1746 s |
| Q1 weights RL(09-24 ctrl)->m-ctrl: TF ppl within 0.5% rel | **PASS (reported)** | ppl 1.5081 -> 1.5062 (0.129%) |
| Q1 weights RL(09-24 ctrl)->m-ctrl: TF top-1 flips <= 2.0% | **PASS (reported)** | 1286/80384 = 1.600% · mean|Δ| 0.03346 p99 0.5294 max 4.206 · docs 39 |
| Q1 weights BFCL dev RL->MOPD (finding if < -2 pt) | **PASS (reported)** | RL 562/600 = 93.67% · MOPD 563/600 = 93.83% · Δ +0.16 pt |
| Q1 weights BFCL heldout RL->MOPD (finding if < -2 pt) | **PASS (reported)** | RL 1029/1311 = 78.49% · MOPD 1028/1311 = 78.41% · Δ -0.08 pt |
| Q1 rl-hot dev re-run (same-model repeat) | **info** | 564/600 = 94.00% (09-24 v23: 563/600 = 93.83%) |
| Q2 RL ranking on MOPD live >= RL ranking on RL live - 3 pt | **PASS** | MOPD 53.4% vs RL 45.9% (oracles 72.2% / 80.1%; Jaccard rank-vs-MOPD-oracle 0.367, RL-vs-RL 0.274; tokens/layer MOPD 104198 RL 137011) -> RE-RANK |
| Q3 flood rl-hot (RL) | **info** | solved 47/72 · within-rep 94.66% · cross-rep 94.66% · flood tasks 25 · dup tasks 25 · calls 5934 · max/turn 407 · tokens 113748 · unsolved {'no_answer': 0, 'truncated': 0, 'error': 0, 'wrong': 0, 'flooded': 25} |
| Q3 flood m-ctrl (MOPD stock) | **info** | solved 72/72 · within-rep 0.00% · cross-rep 0.00% · flood tasks 0 · dup tasks 0 · calls 339 · max/turn 13 · tokens 16707 · unsolved {'no_answer': 0, 'truncated': 0, 'error': 0, 'wrong': 0, 'flooded': 0} |
| Q3 flood m-hot (MOPD hotsplit) | **info** | solved 72/72 · within-rep 0.00% · cross-rep 0.00% · flood tasks 0 · dup tasks 0 · calls 341 · max/turn 12 · tokens 16828 · unsolved {'no_answer': 0, 'truncated': 0, 'error': 0, 'wrong': 0, 'flooded': 0} |
| Q3 residency sanity: |Δ flood tasks| <= 2 and |Δ solved| <= 3 | **PASS** | Δ flood tasks 0, Δ solved 0 |

greedy-compare-residency.log: GREEDY identical=1/12  (greedy-mctrl.json vs greedy-mhot.json) -> greedy-compare-mctrl-vs-mhot.json
  tool-shaped-1: first divergence at token 112 (tokens 259 vs 251)
  tool-shaped-2: identical (tokens 46 vs 46)
  code-1: first divergence at token 159 (tokens 512 vs 512)
  code-2: first divergence at token 0 (tokens 152 vs 136)
  code-3: first divergence at token 47 (tokens 512 vs 512)
  prose-1: first divergence at token 19 (tokens 512 vs 512)
  prose-2: first divergence at token 44 (tokens 370 vs 356)
  reason-1: first divergence at token 25 (tokens 512 vs 512)
  reason-2: first divergence at token 7 (tokens 512 vs 512)
  structured-1: first divergence at token 11 (tokens 163 vs 171)
  ops-1: first divergence at token 5 (tokens 512 vs 512)
  long-1: first divergence at token 61 (tokens 512 vs 512)

greedy-compare-weights.log: GREEDY identical=1/12  (greedy-rlctrl0924.json vs greedy-mctrl.json) -> greedy-compare-rlctrl0924-vs-mctrl.json
  tool-shaped-1: first divergence at token 0 (tokens 512 vs 259)
  tool-shaped-2: identical (tokens 46 vs 46)
  code-1: first divergence at token 0 (tokens 512 vs 512)
  code-2: first divergence at token 0 (tokens 512 vs 152)
  code-3: first divergence at token 5 (tokens 512 vs 512)
  prose-1: first divergence at token 19 (tokens 512 vs 512)
  prose-2: first divergence at token 0 (tokens 384 vs 370)
  reason-1: first divergence at token 24 (tokens 491 vs 512)
  reason-2: first divergence at token 7 (tokens 512 vs 512)
  structured-1: first divergence at token 2 (tokens 146 vs 163)
  ops-1: first divergence at token 0 (tokens 512 vs 512)
  long-1: first divergence at token 52 (tokens 512 vs 512)

tf-compare-residency.log: TF Δlogprob mctrl → mhot: positions=80384 docs=39 (token-mismatch docs=0) mean|Δ|=0.02506 p50=0.00005 p99=0.4049 max=4.195 top-1 flips=951 (1.183%) ppl 1.5062 → 1.5062; wrote tf-compare-mctrl-vs-mhot.json

tf-compare-weights.log: TF Δlogprob ctrl → mctrl: positions=80384 docs=39 (token-mismatch docs=0) mean|Δ|=0.03346 p50=0.00007 p99=0.5294 max=4.206 top-1 flips=1286 (1.600%) ppl 1.5081 → 1.5062; wrote tf-compare-ctrl-vs-mctrl.json

ttft-mhot.log: [mopd-mhot] prompt   2952 tok: TTFT   2.48s (  1192 tok/s prefill)  decode  37.2 tok/s  runs [(2.49, 37.2), (2.48, 37.2), (2.47, 37.0)]
[mopd-mhot] prompt  11522 tok: TTFT   8.38s (  1376 tok/s prefill)  decode  37.1 tok/s  runs [(8.38, 37.1), (8.38, 37.0), (8.37, 37.1)]
[mopd-mhot] prompt  45827 tok: TTFT  33.33s (  1375 tok/s prefill)  decode  36.6 tok/s  runs [(33.35, 36.6), (33.31, 36.6), (33.33, 36.6)]

fixture-mhot.log: [mopd-mhot-r2] tool_json     36.2 tok/s  accept= nan%  acc/step=0.00  tokens=282
[mopd-mhot-r2] code          32.2 tok/s  accept= nan%  acc/step=0.00  tokens=725
[mopd-mhot-r2] shell_ops     36.1 tok/s  accept= nan%  acc/step=0.00  tokens=148
[mopd-mhot-r2] structured    35.1 tok/s  accept= nan%  acc/step=0.00  tokens=372
[mopd-mhot-r2] prose         32.2 tok/s  accept= nan%  acc/step=0.00  tokens=354
[mopd-mhot-r2] weighted accept=nan%

portability.log: K                                             8692
layers                                        69
mopd_live_tokens_per_layer                    104198
hbm_served_share_rl_ranking_on_mopd           0.5340
hbm_served_share_oracle_mopd                  0.7221
jaccard_rlrank_vs_mopd_oracle                 0.3673
pearson_share_rlrank_vs_mopd                  0.4368
stock_layer_order_share_on_mopd               0.3043
rl_live_tokens_per_layer                      137011
hbm_served_share_rl_ranking_on_rl_live        0.4585
hbm_served_share_oracle_rl_live               0.8014
jaccard_rlrank_vs_rl_oracle                   0.2744
pearson_share_rlrank_vs_rl_live               0.2533
jaccard_mopd_oracle_vs_rl_oracle              0.4174
verdict                                       RE-RANK
power power-mctrl-dev.csv: mean 334.4 W, 183 samples
power power-mctrl-flood.csv: mean 338.7 W, 103 samples
power power-mctrl-heldout.csv: mean 345.7 W, 407 samples
power power-mhot-dev.csv: mean 355.2 W, 159 samples
power power-mhot-flood.csv: mean 371.4 W, 86 samples
power power-mhot-heldout.csv: mean 371.5 W, 347 samples
power power-rlhot-dev.csv: mean 355.4 W, 162 samples
power power-rlhot-flood.csv: mean 364.8 W, 451 samples

OVERALL (gated bars only): PASS — recipe verified on MOPD weights

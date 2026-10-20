# Lane C vLLM ladder, Round 2 (2026-10-19/20, sparks 5-8, TP4, DeepSeek-V4.1-Flash-1017)

Image vllm-ds41f-kkref:19f2c20ed4d6-r6x (c1c93733e852, vLLM tree 4d6851c8). Window 01:14-02:12 UTC. One change per boot; every boot warm (89-101 s to HTTP ready). Same-window control is C0ctl2 (unchanged FINAL). C1 = per-stream tok/s, mean of 2 runs; C4/C6 = aggregate tok/s; 93K = prefill_reps 3-rep cold median (93,335 tok); mixed = short-chat TTFT behind a cold 93K prefill (mean of 2). Identity is checked against C0ctl2: qeval content (75 tasks, conc 1), 40-prompt greedy raw-completion probe (T=0, 128 tok), bench C1 per-request token/char signature.

| arm | change | C1 tok/s | C4 | C6 | 93K prefill tok/s | 300K needle | KV tokens | qeval | mixed TTFT s | acceptance | NV_ERR_NO_MEMORY delta | outputs vs C0ctl2 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0ctl2 | none (FINAL, warm reboot) | 104.27 | 230.94 | 288.94 | 5998.4 | 3/3 (5462/5471/5453 tok/s) | 6,537,516 | 71/75 | 3.026 | 0.6119 | 0 new (all nodes) | reference; = FINAL qeval 75/75, = live FINAL greedy 40/40 | control |
| C7ced | Owen CED skip overlay (3 files, ro mounts) | 104.19 (-0.1%) | 231.21 (+0.1%) | 287.30 (-0.6%) | 6054.8 (+0.9%) | 3/3 (5545/5516/5580, +1.6%) | 6,537,516 | 71/75 | 2.872 (-5.1%) | 0.6119 | 0 new | identical: qeval 75/75, greedy 40/40, C1 9/9 | REVERT (no metric > +2%) |
| C8pp2 | --max-parallel-prefills 2 | 104.46 (+0.2%) | 211.68 (-8.3%) | 257.77 (-10.8%) | 6022.9 (+0.4%) | 1/1 (5449) | 6,537,516 | not run | 2.962 (-2.1%, noise) | 0.6119 | 0 new | identical: greedy 40/40, C1 9/9 | REVERT (C4/C6 regress; TTFT not improved, so C8pp2b skipped) |
| C8pp2b | + --long-prefill-token-threshold 6048 | - | - | - | - | - | - | - | - | - | - | - | NOT RUN (gate: C8pp2 TTFT did not improve) |
| C9nvfp4 | DRAFT_NVFP4_HEAD=1, MARKOV_NVFP4=1 | 106.00 (+1.7%) | 233.18 (+1.0%) | 280.77 (-2.8%) | 6049.6 (+0.9%) | 1/1 (5428) | 6,537,516 | 71/75 | 3.002 | 0.6103 | 0 new | identical: qeval 75/75, greedy 40/40, C1 9/9 | REVERT (C6 -2.8% > 2%; C1 below +2%) |
| FINALr2 | restore FINAL | - | - | - | - | - | 6,537,516 | gates PASS | - | - | 0 new | greedy 40/40 = C0ctl2 | SERVING |

Notes
- Every arm: functional gates PASS, 0 preemptions, max GPU clocks.gr 2190 MHz on all four nodes, MemAvailable >= 114.8 GiB after drop_caches before every boot.
- NV_ERR_NO_MEMORY: `dmesg | grep -c` counts only fell (ring buffer rolling old lines out: spark5 1200->1197, spark6 1472->1469, spark7 1458->1455, spark8 1466->1463 over the night); no new events in any arm.
- C7ced: overlay blobs are the end of Owen's patch chain (0186-0211 + candidate) and our image's ced.py / model_state.py are exactly that chain's pre-images (289561bee / 155b13ef0), so the overlay applies to our tree. In-container sha256 matched on all 4 nodes. On TP4 the gain is +0.9% at 93K, +1.6% at 300K, +4.1% at 131K needle prefill (single shot). Owen measured +10.6-11.1% on TP3. Mixed TTFT -5% (2 reps).
- C8pp2: the C4/C6 regression is outside the prior ladder's range (C4 227-231, C6 281-289 across 7 abbreviated arms).
- C9nvfp4: C6 280.8 is at the bottom of the prior ladder's C6 noise band (281-289), so this regression is marginal. Per-category acceptance fell slightly on json/format/coding.
- Previous FINAL row (Round 1, 07:38 UTC) for reference: C1 103.50, C4 229.54, C6 286.61, 93K 5977.1, mixed 2.962.

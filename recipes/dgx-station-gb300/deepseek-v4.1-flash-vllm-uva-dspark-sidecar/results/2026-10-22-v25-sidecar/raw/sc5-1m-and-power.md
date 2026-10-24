**Verdict: FAIL** — R_bench_C31 FAIL; E_speed FAIL; P_swcap FAIL; P_module FAIL. NOT promoted (James decides).
GATE_RESULT FAIL
SUFFIX SC5-FAIL

Cell groups: R FAIL · F PASS · E FAIL · P FAIL (map rowmap-static-v1.json; R cap 32768 tokens)

| # | cell | result | detail |
|---|---|---|---|
| 0 | valid_C | PASS | C (af7f9488): facts present OK; vLLM version OK; server image OK; sidecar image af7f9488 OK; peer mode 2, check 0 OK; rowmap rowmap-static-v1.json 40 layers OK; decoder-replay line (overlay applied) OK; no 'hook FAILED' OK | version=0.30.1rc1.dev245+gaf7f9488c |
| 1 | R_bench_C | PASS | lil-bench payload verbatim, max_tokens 32768: bench: C 0/32 vs A 0/32 OK — bar: runaways (finish=length at max_tokens) cand ≤ A |
| 2 | R_bench_C31 | FAIL | lil-bench payload verbatim, max_tokens 32768: bench: C31 1/32 vs A 0/32 FAIL — bar: runaways (finish=length at max_tokens) cand ≤ A |
| 3 | R_same_prompt_C | PASS | pinned chat_template_kwargs reasoning_effort (same rendered prompt on every image): e50: C 0/4 vs A 0/4 OK; e75: C 0/4 vs A 0/4 OK — bar: runaways (finish=length at max_tokens) cand ≤ A |
| 4 | R_same_prompt_C31 | PASS | pinned reasoning_effort: e50: C31 0/4 vs A 0/4 OK; e75: C31 0/4 vs A 0/4 OK — bar: runaways (finish=length at max_tokens) cand ≤ A |
| 5 | F_boot | PASS | C KV 2,537,081 tokens (bar ≥ 2,500,000); max concurrency for 1,048,576 tokens 2.42x (bar ≥ 1.0x); C31 KV 2,537,081 (2.42x, reported); A 2504891 |
| 6 | F_1m | PASS | C needle http 200 prompt_tokens 999303 (bar ≥ 950,000) TTFT 33.23 s (30069 tok/s) recall beta=True alpha=True finish=stop; post-probe True |
| 7 | F_after | PASS | 128K cold prefill ≥120,000 prompt tokens OK; C16 burst ok==np OK; C16 rc 0 OK; 0 engine-error lines during F OK; health (server+sidecar up, chat 200) OK; sidecar timeouts Δ 0 over F OK | 128K: 131104 tok in 2.61 s; C16 {'ok': 32, 'np': 32, 'decode_agg': 1565.92, 'median_ttft_s': 0.61} |
| 8 | F_decode | PASS | C kbench C1 decode_agg pre-1M 279.48 → post-1M 276.52 (bar post ≥ 0.97×pre OK); A after-1M 191.28 (pre 184.68) → C ≥ A OK |
| 9 | E_valid | PASS | C31 (v0.31.0): facts present OK; vLLM version OK; server image OK; sidecar image af7f9488 OK; peer mode 2, check 0 OK; rowmap rowmap-static-v1.json 40 layers OK; decoder-replay line (overlay applied) OK; no 'hook FAILED' OK | version=0.31.0 |
| 10 | E_tf | PASS | C31 vs C PASS W=2.652 (calib 2.845); C31 vs A PASS W=3.102; reported: C vs A W=2.967, C31 vs SC2-C W=2.621, D vs A W=1.648 |
| 11 | E_gsm8k | PASS | C31 96.97 (1279/1319) vs SC2 C 96.97 (+0.00 pt; McNemar p=1.0; ctok ratio 0.7416112368282705; hit_max 0) bar ≥ −1.0 |
| 12 | E_gauntlet | PASS | C31 12/12 |
| 13 | E_needle | PASS | C31 EXACT 500/500 (near_miss 0, wrong_facts 0, fail 0) — bar 500/500 |
| 14 | E_speed | FAIL | RT C1 decode_agg: C31 280.2 vs C 278.2 (+0.7%); knee6 C1 4-rep mean: C31 220.9 vs C 214.9 (+2.8%); knee6 C8 4-rep mean: C31 892.0 vs C 927.4 (-3.8%) — bar ≥ −3% each |
| 15 | timeouts | PASS | C: shm end 0 Δ 0 peer-log 0 hook 0 (watchdog trips: timeout 0, other 0; wait 0.0136 ms/wait); C31: shm end 0 Δ 0 peer-log 0 hook 0 (watchdog trips: timeout 0, other 0; wait 0.0061 ms/wait) — bar 0 everywhere |
| 16 | P_swcap | FAIL | idle 0.00%; c1_decode 0.00%; c32_decode 2.42% FAIL; prefill_64k 89.47% FAIL; c32_prefill_8k 85.71% FAIL — GB300 sw_power_cap active, % of 1 s samples, bar < 1.0% per phase |
| 17 | P_rtx | PASS | idle 106.22 W; c1_decode 135.38 W; c32_decode 193.02 W; prefill_64k 194.36 W; c32_prefill_8k 186.91 W — RTX power.draw p99, bar ≤ 300 W |
| 18 | P_module | FAIL | idle 385.36 W; c1_decode 860.68 W; c32_decode 1263.24 W FAIL; prefill_64k 1353.35 W FAIL; c32_prefill_8k 1342.23 W FAIL — GB300 module.power.draw.instant p99, bar ≤ 1190 W (PENDING if unreadable) |

Boot: C 527 s (autotune starts 25, loaded ['159'], saved ['159']); C31 547 s (autotune starts 25, loaded None, saved ['117']); KV C 2537081 / C31 2537081 / A 2504891.
Assumption (Milo decides): no cosine-gate (MEGA_PEER_CHECK) boot on C31; E_tf vs C stands in for it.

### R — runaway repro per item (runaways/done · completion tokens · correct) — cap 32768 tokens

| variant | run_index (item) | source | SC2 C ctoks / A ctoks | A | C | C31 |
|---|---|---|---|---|---|---|
| bench | 1304 (gsm8k-1303) | main | 131072 / 2245 | 0/8 · [4397, 4228, 6947, 6708, 4551, 3454, 2486, 6939] · 7✓ | 0/8 · [12062, 4963, 591, 6306, 6070, 9076, 8795, 2966] · 8✓ | 0/8 · [9886, 1949, 8211, 1274, 4366, 590, 11017, 10650] · 7✓ |
| bench | 101 (gsm8k-0100) | A>thr | 578 / 4467 | 0/2 · [1554, 4176] · 2✓ | 0/2 · [3071, 8913] · 0✓ | 0/2 · [1651, 2278] · 1✓ |
| bench | 185 (gsm8k-0184) | C>thr | 7908 / 394 | 0/2 · [343, 343] · 0✓ | 0/2 · [622, 8791] · 1✓ | 0/2 · [1038, 385] · 0✓ |
| bench | 381 (gsm8k-0380) | C>thr | 9400 / 338 | 0/2 · [467, 467] · 0✓ | 0/2 · [19403, 1770] · 1✓ | 1/2 · [32768, 6820] · 1✓ |
| bench | 407 (gsm8k-0406) | C>thr | 4753 / 5905 | 0/2 · [3147, 2465] · 0✓ | 0/2 · [3473, 1747] · 0✓ | 0/2 · [1661, 1659] · 1✓ |
| bench | 424 (gsm8k-0423) | C>thr | 7509 / 1156 | 0/2 · [2364, 2379] · 2✓ | 0/2 · [1361, 2815] · 2✓ | 0/2 · [4265, 1516] · 2✓ |
| bench | 783 (gsm8k-0782) | C>thr | 5662 / 847 | 0/2 · [558, 510] · 0✓ | 0/2 · [3563, 1767] · 0✓ | 0/2 · [3132, 2544] · 0✓ |
| bench | 952 (gsm8k-0951) | C>thr | 12101 / 1860 | 0/2 · [5694, 3282] · 0✓ | 0/2 · [538, 2454] · 1✓ | 0/2 · [8630, 3640] · 0✓ |
| bench | 1053 (gsm8k-1052) | A>thr | 1129 / 5009 | 0/2 · [1043, 891] · 0✓ | 0/2 · [996, 1228] · 2✓ | 0/2 · [1918, 814] · 2✓ |
| bench | 1120 (gsm8k-1119) | A>thr | 3170 / 4254 | 0/2 · [1250, 2543] · 2✓ | 0/2 · [8985, 3092] · 2✓ | 0/2 · [6951, 3668] · 2✓ |
| bench | 1147 (gsm8k-1146) | A>thr | 540 / 4859 | 0/2 · [623, 610] · 2✓ | 0/2 · [779, 314] · 2✓ | 0/2 · [370, 718] · 2✓ |
| bench | 1177 (gsm8k-1176) | C>thr | 10038 / 2353 | 0/2 · [9972, 8420] · 2✓ | 0/2 · [4983, 9726] · 2✓ | 0/2 · [4445, 1742] · 1✓ |
| bench | 1184 (gsm8k-1183) | C>thr | 5423 / 1953 | 0/2 · [2273, 4452] · 1✓ | 0/2 · [2559, 2432] · 0✓ | 0/2 · [2799, 3135] · 1✓ |
| e50 | 1304 (gsm8k-1303) | main | 131072 / 2245 | 0/4 · [3164, 6083, 6441, 4168] · 4✓ | 0/4 · [3516, 2823, 723, 4849] · 4✓ | 0/4 · [1569, 3574, 6651, 4674] · 4✓ |
| e75 | 1304 (gsm8k-1303) | main | 131072 / 2245 | 0/4 · [15413, 6648, 8051, 6227] · 4✓ | 0/4 · [4543, 6289, 3828, 8972] · 4✓ | 0/4 · [5322, 7763, 10069, 4395] · 4✓ |

- A bench: runaways 0/32 (requests 32, errors 0, skipped 0), max repeated-32-gram ratio (last 4000 chars) 0.2414
- A e50: runaways 0/4 (requests 4, errors 0, skipped 0), max repeated-32-gram ratio (last 4000 chars) 0.1295
- A e75: runaways 0/4 (requests 4, errors 0, skipped 0), max repeated-32-gram ratio (last 4000 chars) 0.0292
- C bench: runaways 0/32 (requests 32, errors 0, skipped 0), max repeated-32-gram ratio (last 4000 chars) 0.1484
- C e50: runaways 0/4 (requests 4, errors 0, skipped 0), max repeated-32-gram ratio (last 4000 chars) 0.103
- C e75: runaways 0/4 (requests 4, errors 0, skipped 0), max repeated-32-gram ratio (last 4000 chars) 0.0262
- C31 bench: runaways 1/32 (requests 32, errors 0, skipped 0), max repeated-32-gram ratio (last 4000 chars) 0.1325
- C31 e50: runaways 0/4 (requests 4, errors 0, skipped 0), max repeated-32-gram ratio (last 4000 chars) 0.0632
- C31 e75: runaways 0/4 (requests 4, errors 0, skipped 0), max repeated-32-gram ratio (last 4000 chars) 0.0436

### F — 1M fit (reported detail)

| arm | 1M prompt tok | TTFT s | prefill tok/s | recall | depth-1M decode tok/s (TTFT s) | C1 decode pre→post | 128K tok / s | C16 ok · agg |
|---|---|---|---|---|---|---|---|---|
| A | 998567 | 81.44 | 12262 | True | 172.6 (81.41) | 184.68 → 191.28 | 131179 / 5.8 | 32/32 · 526.76 |
| C | 999303 | 33.23 | 30069 | True | 537.3 (2.76) | 279.48 → 276.52 | 131104 / 2.61 | 32/32 · 1565.92 |

### P — power by phase (GB300 p50/p99/max W · module p99/max W · sw_power_cap % samples (counter %) · RTX p99 W · Grace socket p99 W · SM clock p5/p50/max MHz)

| arm | phase | n | GB300 W p50/p99/max | module W p99/max | swcap % (ctr %) | RTX p99 W | Grace p99 W | SM clk p5/p50/max | GB300 °C max |
|---|---|---|---|---|---|---|---|---|---|
| A | idle | 60 | 211.09/213.44/214.32 | 360.63/361.44 | 0.00 (0.00) | 22.59 | 71.745 | 2070.0/2070.0/2070.0 | 44.0 |
| A | c1_decode | 175 | 489.76/527.78/533.75 | 872.32/916.78 | 0.00 (0.00) | 23.84 | 101.058 | 2070.0/2070.0/2070.0 | 50.0 |
| A | c32_decode | 165 | 526.34/662.6/679.28 | 972.07/988.21 | 0.00 (0.00) | 23.35 | 102.308 | 2070.0/2070.0/2070.0 | 54.0 |
| A | prefill_64k | 40 | 951.53/974.65/974.65 | 1278.66/1278.66 | 65.00 (49.27) | 23.81 | 95.293 | 2010.0/2062.0/2070.0 | 61.0 |
| A | c32_prefill_8k | 29 | 955.32/970.38/970.38 | 1300.09/1300.09 | 51.72 (52.13) | 23.68 | 114.466 | 2002.0/2062.0/2070.0 | 60.0 |
| A | r_probe | 679 | 483.38/502.81/641.39 | 686.93/705.48 | 0.00 (0.00) | 23.55 | 88.985 | 2070.0/2070.0/2070.0 | 50.0 |
| A | f_fit | 280 | 788.86/988.68/1001.58 | 1260.08/1297.76 | 11.43 (10.24) | 23.65 | 99.152 | 2040.0/2070.0/2070.0 | 64.0 |
| A | whole_arm | 1561 | 489.14/974.54/1001.58 | 1240.37/1300.09 | 4.68 (4.22) | 23.67 | 101.49 | 2070.0/2070.0/2070.0 | 64.0 |
| C | idle | 60 | 213.37/220.32/638.59 | 385.36/389.05 | 0.00 (0.00) | 106.22 | 82.043 | 2070.0/2070.0/2070.0 | 52.0 |
| C | c1_decode | 175 | 588.47/604.9/608.29 | 860.68/908.72 | 0.00 (0.00) | 135.38 | 83.682 | 2070.0/2070.0/2070.0 | 51.0 |
| C | c32_decode | 165 | 849.51/988.35/990.23 | 1263.24/1317.98 | 2.42 (2.02) | 193.02 | 86.617 | 2070.0/2070.0/2070.0 | 61.0 |
| C | prefill_64k | 19 | 924.06/949.76/949.76 | 1353.35/1353.35 | 89.47 (76.01) | 194.36 | 88.754 | 1882.0/1980.0/2070.0 | 62.0 |
| C | c32_prefill_8k | 14 | 941.48/952.13/952.13 | 1342.23/1342.23 | 85.71 (76.91) | 186.91 | 108.016 | 1830.0/1980.0/2070.0 | 58.0 |
| C | r_probe | 574 | 600.09/614.32/680.3 | 801.27/932.07 | 0.00 (0.00) | 141.52 | 83.851 | 2070.0/2070.0/2070.0 | 54.0 |
| C | f_fit | 90 | 862.53/977.77/980.63 | 1365.96/1371.85 | 25.56 (31.03) | 202.57 | 87.496 | 1897.0/2070.0/2070.0 | 62.0 |
| C | whole_arm | 1301 | 599.49/977.77/990.23 | 1271.12/1371.85 | 4.61 (4.97) | 193.83 | 87.422 | 2070.0/2070.0/2070.0 | 62.0 |
| C31 | r_probe | 590 | 599.04/613.91/616.16 | 799.0/1082.74 | 0.00 (0.00) | 145.84 | 83.144 | 2070.0/2070.0/2070.0 | 53.0 |
| C31 | whole_arm | 978 | 602.49/918.22/970.8 | 1180.97/1193.05 | 0.41 (0.27) | 195.89 | 88.23 | 2070.0/2070.0/2070.0 | 61.0 |

Read-only power discovery (module field, hwmon power_meter channels, ipmitool) is in `power-discovery.txt`.

## Milo review notes (2026-10-22 02:07 CDT) — not part of the generated verdict
- **Runaway: not a sidecar effect.** C 0/32 (bench payload) and 0/8 at pinned efforts 50 and 75; item 1304 did not repeat
  (max 12,062 tokens). C31's single 32K hit (item 381, default effort = 75) sits in the effort-75 tail that A also shows when
  pinned to 75 (15,413 tokens on 1304). 1/32 vs 0/32 is not distinguishable; the consumer-visible cause is the image's effort
  mapping (default high = 75 vs v23's 50), which promotion would carry unless clients pin an integer effort.
- **v0.31.0 = af7f9488 on quality** (TF, GSM8K 1279/1319, needle 500/500, gauntlet 12/12, 0 timeouts). E_speed fails only on knee6
  C8 −3.8% (bar −3%; C1 +0.7/+2.8%); C8 is the bimodal cell (P2g notes) — one more bracket needed before calling it a regression.
- **Power: the incumbent already throttles in prefill.** v23 A: sw_power_cap 65% / 52% of samples in 64K / C32-8K prefill, module
  p99 1,279–1,300 W. The sidecar adds throttling in C32 decode (2.4%, module p99 1,263 W) and drops prefill SM clock p50 to 1,980 MHz
  (v23 2,062). The SC6 bars (<1% swcap, module ≤1,190 W) are failed by v23 too, so they were set too tight for this chassis;
  the speed numbers in SC1/SC2 were measured under this throttling. Firmware/vsloshd question stays open (W5).
- **1M fit passes** with headroom: 999,303-token needle recalled, TTFT 33 s (v23: 81 s), 128K prefill 2.6 s, C16 32/32 after.

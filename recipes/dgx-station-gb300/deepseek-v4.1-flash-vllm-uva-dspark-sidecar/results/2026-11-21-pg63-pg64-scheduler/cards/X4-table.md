| row | gb300-pg62-A6 | gb300-pg62-A7 | gb300-G1 | gb300-G2 |
|---|---:|---:|---:|---:|
| C1 per-stream | 541.25 | 546.13 | 543.76 | 551.15 |
| C16 agg | 2438.78 | 2405.96 | 2417.15 | 2397.11 |
| prefill 32K tok/s | 45290.1 | 45064.7 | 44891.6 | 45648.6 |
| prefill 131K tok/s | 43461.9 | 43351.5 | 43216.9 | 43607.7 |
| prefill 300K tok/s | 41121.1 | 41175.4 | 41070.2 | 40954.6 |
| prefill 1000K tok/s | 27497.7 | 28063.6 | 27880.7 | 28056.7 |
| needle | 3/3 36.1s | 3/3 36.1s | 3/3 36.1s | 3/3 35.9s |
| mixed_short (to answer) s | 1.298 | 1.461 | 0.669 | 0.707 |
| mixed_warm s | 0.523 | 0.608 | 0.318 | 0.462 |
| agent turn s | 0.259 | 0.268 | 0.261 | 0.256 |
| qeval | 71 | 71 | 71 | 71 |
| true TTFT behind 413K s | 0.487 | 0.493 | 0.329 | 0.362 |
| short decode tok/s (32t) | 18.5 | 15.9 | 37.55 | 34.1 |
| decode beside 413K tok/s | 44.5 | 44.1 | 50.0 | 43.8 |
| decode DURING prefill tok/s (400t) | 12.2 | 12.5 | 33.3 | 33.1 |
| decode DURING 1M prefill tok/s | 10.4 | 10.6 | 28.7 | 30.4 |
| 1M time while shared s | 37.469 | 37.531 | 37.044 | 37.414 |
| 413K time while shared s | 10.956 | 10.95 | 10.986 | 10.961 |
| C32 agg (mean of cats) | 3938.8 | 3879.4 | 3788.7 | 3913.4 |
| greedy same-boot | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 |
| JSON burst | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS |
| DSML live | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} |
| alive | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite |

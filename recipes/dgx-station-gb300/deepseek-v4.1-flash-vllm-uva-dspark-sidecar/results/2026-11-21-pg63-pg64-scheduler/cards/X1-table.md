| row | gb300-pg62-A1 | gb300-B | gb300-C | gb300-GR | gb300-pg62-A2 |
|---|---:|---:|---:|---:|---:|
| C1 per-stream | 547.32 | 559.46 | 562.48 | 547.51 | 540.42 |
| C16 agg | 2453.82 | 2403.07 | 2406.16 | 2421.88 | 2379.39 |
| prefill 32K tok/s | 45792.3 | 45148.3 | 45657.5 | 45325.0 | 45127.3 |
| prefill 131K tok/s | 43213.2 | 43335.0 | 43177.2 | 43257.8 | 43366.8 |
| prefill 300K tok/s | 41197.8 | 41020.2 | 41006.8 | 41243.5 | 40935.6 |
| prefill 1000K tok/s | 28129.2 | 28062.1 | 28068.0 | 28377.4 | 28035.7 |
| needle | 3/3 36.0s | 3/3 36.0s | 3/3 36.2s | 3/3 35.6s | 3/3 36.0s |
| mixed_short (to answer) s | 1.296 | 0.879 | 0.665 | 1.388 | 1.443 |
| mixed_warm s | 0.523 | 0.559 | 0.453 | 0.575 | 0.462 |
| agent turn s | 0.265 | 0.255 | 0.258 | 0.276 | 0.266 |
| qeval | 71 | 71 | 71 | 71 | 71 |
| true TTFT behind 413K s | 0.508 | 0.494 | 0.469 | 0.501 | 0.511 |
| short decode tok/s (32t) | 13.75 | 25.5 | 33.65 | 13.65 | 17.7 |
| decode beside 413K tok/s | 51.8 | 50.5 | 47.6 | 44.0 | 40.6 |
| decode DURING prefill tok/s (400t) | 12.6 | 23.2 | 34.6 | 12.8 | 12.8 |
| decode DURING 1M prefill tok/s | 10.2 | 14.4 | 30.2 | 10.0 | 11.1 |
| 1M time while shared s | 37.541 | 38.581 | 37.372 | 36.882 | 37.432 |
| 413K time while shared s | 10.875 | 10.948 | 10.955 | 10.816 | 10.94 |
| C32 agg (mean of cats) | 3895.2 | 3774.7 | 3659.5 | 3923.0 | 3887.0 |
| greedy same-boot | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 |
| JSON burst | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS |
| DSML live | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} |
| alive | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite |

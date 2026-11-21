| row | gb300-pg62-A1 | gb300-pg62-A2 | gb300-pg62-A3 | gb300-pg62-A4 | gb300-B | gb300-C | gb300-GR | gb300-D1 | gb300-D2 | gb300-E1 | gb300-E2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 per-stream | 547.32 | 540.42 | 545.54 | 540.37 | 559.46 | 562.48 | 547.51 | 557.51 | 549.23 | 554.96 | 551.72 |
| C16 agg | 2453.82 | 2379.39 | 2386.35 | 2374.74 | 2403.07 | 2406.16 | 2421.88 | 2404.49 | 2378.71 | 2407.04 | 2406.52 |
| prefill 32K tok/s | 45792.3 | 45127.3 | 44968.0 | 45102.8 | 45148.3 | 45657.5 | 45325.0 | 44791.2 | 44987.6 | 45301.2 | 44811.1 |
| prefill 131K tok/s | 43213.2 | 43366.8 | 43627.0 | 43193.1 | 43335.0 | 43177.2 | 43257.8 | 43176.9 | 43305.0 | 43380.3 | 43497.9 |
| prefill 300K tok/s | 41197.8 | 40935.6 | 40717.7 | 41145.0 | 41020.2 | 41006.8 | 41243.5 | 41121.7 | 40972.3 | 41244.4 | 41128.1 |
| prefill 1000K tok/s | 28129.2 | 28035.7 | 28164.3 | 28112.1 | 28062.1 | 28068.0 | 28377.4 | 28112.6 | 28146.7 | 28042.2 | 27936.3 |
| needle | 3/3 36.0s | 3/3 36.0s | 3/3 36.1s | 3/3 36.1s | 3/3 36.0s | 3/3 36.2s | 3/3 35.6s | 3/3 36.1s | 3/3 36.0s | 3/3 36.0s | 3/3 36.1s |
| mixed_short (to answer) s | 1.296 | 1.443 | 1.494 | 1.392 | 0.879 | 0.665 | 1.388 | 0.73 | 0.758 | 0.669 | 0.756 |
| mixed_warm s | 0.523 | 0.462 | 0.615 | 0.608 | 0.559 | 0.453 | 0.575 | 0.324 | 0.502 | 0.512 | 0.495 |
| agent turn s | 0.265 | 0.266 | 0.274 | 0.272 | 0.255 | 0.258 | 0.276 | 0.26 | 0.264 | 0.258 | 0.259 |
| qeval | 71 | 71 | 71 | 71 | 71 | 71 | 71 | 70 | 71 | 71 | 71 |
| true TTFT behind 413K s | 0.508 | 0.511 | 0.543 | 0.46 | 0.494 | 0.469 | 0.501 | 0.371 | 0.391 | 0.371 | 0.368 |
| short decode tok/s (32t) | 13.75 | 17.7 | 14.5 | 15.0 | 25.5 | 33.65 | 13.65 | 30.95 | 34.3 | 37.0 | 33.1 |
| decode beside 413K tok/s | 51.8 | 40.6 | 46.5 | 49.2 | 50.5 | 47.6 | 44.0 | 39.1 | 50.8 | 50.9 | 49.4 |
| decode DURING prefill tok/s (400t) | 12.6 | 12.8 | 13.2 | 13.7 | 23.2 | 34.6 | 12.8 | 34.6 | 32.2 | 32.6 | 33.8 |
| decode DURING 1M prefill tok/s | 10.2 | 11.1 | 10.4 | 10.4 | 14.4 | 30.2 | 10.0 | 28.6 | 31.8 | 28.2 | 28.6 |
| 1M time while shared s | 37.541 | 37.432 | 37.561 | 37.529 | 38.581 | 37.372 | 36.882 | 37.439 | 37.232 | 37.343 | 37.716 |
| 413K time while shared s | 10.875 | 10.94 | 10.929 | 10.954 | 10.948 | 10.955 | 10.816 | 11.03 | 11.053 | 10.924 | 10.983 |
| C32 agg (mean of cats) | 3895.2 | 3887.0 | 4001.5 | 3850.6 | 3774.7 | 3659.5 | 3923.0 | 3794.8 | 3692.7 | 3824.2 | 3766.2 |
| greedy same-boot | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 |
| JSON burst | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS |
| DSML live | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} |
| alive | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite |

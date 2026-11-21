| row | gb300-pg63-K1 | gb300-pg63-K2 | gb300-Y1 | gb300-Y2 |
|---|---:|---:|---:|---:|
| C1 per-stream | 547.63 | 552.16 | 547.07 | 536.3 |
| C16 agg | 2387.63 | 2399.86 | 2393.55 | 2454.55 |
| prefill 32K tok/s | 45373.4 | 45550.9 | 45278.1 | 45685.0 |
| prefill 131K tok/s | 42937.9 | 43122.3 | 43059.3 | 43249.1 |
| prefill 300K tok/s | 41090.3 | 41157.4 | 41180.3 | 41053.4 |
| prefill 1000K tok/s | 27959.9 | 28172.9 | 28117.7 | 27974.4 |
| needle | 3/3 36.0s | 3/3 36.1s | 3/3 36.0s | 3/3 36.1s |
| mixed_short (to answer) s | 0.583 | 0.81 | 0.572 | 0.562 |
| mixed_warm s | 0.467 | 0.383 | 0.53 | 0.406 |
| agent turn s | 0.263 | 0.259 | 0.268 | 0.258 |
| qeval | 71 | 71 | 71 | 71 |
| true TTFT behind 413K s | 0.418 | 0.352 | 0.211 | 0.223 |
| short decode tok/s (32t) | 43.25 | 37.2 | 43.65 | 37.2 |
| decode beside 413K tok/s | 44.3 | 46.3 | 44.7 | 43.0 |
| decode DURING prefill tok/s (400t) | 33.6 | 32.5 | 34.4 | 32.2 |
| decode DURING 1M prefill tok/s | 29.1 | 27.9 | 28.3 | 28.2 |
| 1M time while shared s | 37.837 | 37.968 | 37.56 | 37.558 |
| 413K time while shared s | 10.941 | 11.051 | 10.938 | 10.953 |
| C32 agg (mean of cats) | 3727.4 | 3793.2 | 3692.4 | 3788.5 |
| greedy same-boot | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 | greedy same-boot identical 40 / 40 |
| JSON burst | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 63, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS | 4, "exact_count": 64, "total": 64, "alive_after": true} PASS |
| DSML live | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} | h", "base": "http://gb300:30006/v1", "ALL_PASS": true} |
| alive | ALIVE after suite | ALIVE after suite | ALIVE after suite | ALIVE after suite |

# MiMo-V2.6-Pro finish run — verdict (bars from harness/protocol.yaml)

| bar | result | detail |
|---|---|---|
| TF ppl within 0.5% rel | **PASS** | ppl 1.5081 -> 1.5077 (0.025%) |
| TF top-1 flips <= 2.0% | **PASS** | 987/80384 = 1.228% · mean|Δ| 0.02630 p50 0.000067 p99 0.4272 max 4.342 · docs 39 (tok-mismatch 0) |
| BFCL dev v23 >= ctrl - 1 pt | **PASS** | ctrl 562/600 = 93.67% · v23 563/600 = 93.83% · Δ +0.16 pt · no-call 5/5 · fail kinds ctrl {'value': 31, 'no_call': 5, 'name': 1, 'missing': 1} v23 {'value': 31, 'no_call': 5, 'name': 1} |
| errors ctrl dev <= 0.5% | **PASS** | 0/600 · wall 929 s |
| errors v23 dev <= 0.5% | **PASS** | 0/600 · wall 794 s |
| BFCL heldout v23 >= ctrl - 1 pt | **PASS** | ctrl 1029/1311 = 78.49% · v23 1025/1311 = 78.18% · Δ -0.31 pt · no-call 46/52 · fail kinds ctrl {'value': 172, 'no_call': 46, 'missing': 34, 'name': 30} v23 {'value': 170, 'no_call': 52, 'missing': 34, 'name': 30} |
| errors ctrl heldout <= 0.5% | **PASS** | 0/1311 · wall 2390 s |
| errors v23 heldout <= 0.5% | **PASS** | 0/1311 · wall 1839 s |

Greedy parity (reported, not gated): GREEDY identical=1/12  (greedy-ctrl.json vs greedy-v23.json) -> greedy-compare-ctrl-vs-v23.json
1M row (v24): ["[finish-v24-1m] needle 1040000 d=0.1: prompt=1039679 TTFT=1101.63s prefill=944 tok/s pass=True got='743-BISON-8704'", "[finish-v24-1m] needle 1040000 d=0.5: prompt=1039704 TTFT=1101.18s prefill=944 tok/s pass=True got='176-LYNX-6788'", "[finish-v24-1m] needle 1040000 d=0.9: prompt=1039700 TTFT=1100.88s prefill=944 tok/s pass=True got='311-OTTER-7247'", '[finish-v24-1m] decode 1040000: prompt=1035393 TTFT=1096.47s decode=25.7 tok/s', '[finish-v24-1m] LONGCTX_DONE needles 3/3']
power power-ctrl-dev.csv: mean 333.7 W, 185 samples
power power-ctrl-heldout.csv: mean 344.4 W, 475 samples
power power-v23-dev.csv: mean 356.1 W, 158 samples
power power-v23-heldout.csv: mean 371.6 W, 364 samples

OVERALL: PROMOTE (experimental -> verified)

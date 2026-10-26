## per-class decode wall-clock — BR4-P-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 236.8 | 239.5 | 238.1 | 1.2% | 538.8 | 535.9 | 537.3 | 0.5% | 0.434/0.416 | 0.426/0.423 | 0.247 | 10240/10240 | 165 |
| structured | 40/40 | 256.2 | 251.9 | 254.1 | 1.7% | 540.3 | 566.8 | 553.5 | 4.8% | 0.460/0.448 | 0.414/0.432 | 0.468 | 10240/10240 | 193 |
| code | 40/40 | 328.6 | 328.7 | 328.7 | 0.0% | 689.8 | 716.1 | 703.0 | 3.8% | 0.700/0.698 | 0.646/0.678 | 0.784 | 10240/10240 | 118 |
| shell_ops | 40/40 | 356.7 | 337.5 | 347.1 | 5.5% | 829.5 | 783.7 | 806.6 | 5.7% | 0.712/0.671 | 0.678/0.636 | 0.770 | 10240/10240 | 117 |
| tool_json | 40/40 | 311.3 | 303.2 | 307.2 | 2.6% | 823.2 | 879.2 | 851.2 | 6.6% | 0.578/0.565 | 0.583/0.613 | 0.831 | 10240/10240 | 109 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

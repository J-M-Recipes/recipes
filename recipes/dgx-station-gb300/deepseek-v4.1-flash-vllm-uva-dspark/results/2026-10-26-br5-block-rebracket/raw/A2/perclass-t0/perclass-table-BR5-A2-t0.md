## per-class decode wall-clock — BR5-A2-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 226.1 | 246.3 | 236.2 | 8.5% | 528.4 | 534.7 | 531.5 | 1.2% | 0.415/0.434 | 0.415/0.419 | 0.247 | 10240/10240 | 167 |
| structured | 40/40 | 258.9 | 261.7 | 260.3 | 1.1% | 574.2 | 553.1 | 563.6 | 3.7% | 0.460/0.469 | 0.427/0.413 | 0.468 | 10240/10240 | 182 |
| code | 40/40 | 320.7 | 322.0 | 321.4 | 0.4% | 736.5 | 729.6 | 733.0 | 0.9% | 0.664/0.662 | 0.675/0.683 | 0.784 | 10240/10240 | 118 |
| shell_ops | 40/40 | 349.5 | 326.4 | 337.9 | 6.9% | 757.3 | 825.0 | 791.2 | 8.6% | 0.694/0.644 | 0.639/0.676 | 0.770 | 10240/10240 | 117 |
| tool_json | 40/40 | 302.2 | 319.0 | 310.6 | 5.4% | 931.3 | 877.7 | 904.5 | 5.9% | 0.555/0.600 | 0.625/0.592 | 0.831 | 10240/10240 | 104 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

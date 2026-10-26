## per-class decode wall-clock — BR4-A2-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 195.9 | 198.5 | 197.2 | 1.3% | 463.4 | 445.8 | 454.6 | 3.9% | 0.310/0.309 | 0.307/0.310 | 0.247 | 10240/10240 | 150 |
| structured | 40/40 | 217.1 | 217.2 | 217.2 | 0.1% | 557.3 | 542.6 | 550.0 | 2.7% | 0.350/0.355 | 0.364/0.351 | 0.468 | 10240/10240 | 187 |
| code | 40/40 | 260.4 | 261.4 | 260.9 | 0.4% | 572.3 | 547.0 | 559.6 | 4.5% | 0.515/0.519 | 0.537/0.495 | 0.784 | 10240/10240 | 126 |
| shell_ops | 40/40 | 280.6 | 284.6 | 282.6 | 1.4% | 645.2 | 626.5 | 635.8 | 2.9% | 0.526/0.533 | 0.515/0.525 | 0.770 | 10240/10240 | 136 |
| tool_json | 40/40 | 254.2 | 248.7 | 251.4 | 2.2% | 650.9 | 672.8 | 661.9 | 3.3% | 0.446/0.443 | 0.458/0.470 | 0.831 | 10240/10240 | 118 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

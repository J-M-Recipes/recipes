## per-class decode wall-clock — v23

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 232.7 | 232.0 | 232.3 | 0.3% | 538.7 | 525.3 | 532.0 | 2.5% | 0.419/0.417 | 0.431/0.412 | 0.247 | 10240/10240 | 161 |
| structured | 40/40 | 251.3 | 249.3 | 250.3 | 0.8% | 502.4 | 516.3 | 509.4 | 2.7% | 0.461/0.449 | 0.428/0.429 | 0.468 | 10240/10240 | 189 |
| code | 40/40 | 288.5 | 298.2 | 293.4 | 3.3% | 638.3 | 645.9 | 642.1 | 1.2% | 0.649/0.683 | 0.664/0.688 | 0.784 | 10240/10240 | 126 |
| shell_ops | 40/40 | 308.9 | 324.9 | 316.9 | 5.0% | 692.6 | 670.1 | 681.4 | 3.3% | 0.661/0.708 | 0.687/0.646 | 0.770 | 10240/10240 | 120 |
| tool_json | 40/40 | 302.3 | 297.7 | 300.0 | 1.6% | 694.7 | 673.4 | 684.0 | 3.1% | 0.603/0.589 | 0.603/0.545 | 0.831 | 10240/10240 | 100 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

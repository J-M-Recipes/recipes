## per-class decode wall-clock — BR6-A-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 217.1 | 226.1 | 221.6 | 4.0% | 480.5 | 486.4 | 483.4 | 1.2% | 0.372/0.379 | 0.367/0.368 | 0.247 | 10240/10240 | 164 |
| structured | 40/40 | 243.5 | 242.3 | 242.9 | 0.5% | 607.5 | 587.3 | 597.4 | 3.4% | 0.411/0.408 | 0.432/0.406 | 0.468 | 10240/10240 | 220 |
| code | 40/40 | 278.5 | 276.2 | 277.3 | 0.8% | 614.3 | 654.4 | 634.4 | 6.3% | 0.567/0.566 | 0.565/0.589 | 0.784 | 10240/10240 | 146 |
| shell_ops | 40/40 | 338.7 | 327.3 | 333.0 | 3.4% | 768.4 | 697.8 | 733.1 | 9.6% | 0.658/0.630 | 0.625/0.590 | 0.770 | 10240/10240 | 141 |
| tool_json | 40/40 | 289.5 | 281.9 | 285.7 | 2.6% | 727.2 | 798.2 | 762.7 | 9.3% | 0.542/0.520 | 0.530/0.539 | 0.831 | 10240/10240 | 84 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

## per-class decode wall-clock — BR3-A-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 233.3 | 241.8 | 237.6 | 3.5% | 492.6 | 514.0 | 503.3 | 4.2% | 0.425/0.429 | 0.396/0.421 | 0.247 | 10240/10240 | 164 |
| structured | 40/40 | 258.0 | 250.9 | 254.4 | 2.8% | 566.4 | 570.4 | 568.4 | 0.7% | 0.473/0.437 | 0.436/0.445 | 0.468 | 10240/10240 | 188 |
| code | 40/40 | 322.5 | 324.5 | 323.5 | 0.6% | 705.6 | 754.0 | 729.8 | 6.6% | 0.696/0.685 | 0.674/0.681 | 0.784 | 10240/10240 | 126 |
| shell_ops | 40/40 | 352.2 | 334.1 | 343.2 | 5.3% | 787.7 | 800.5 | 794.1 | 1.6% | 0.707/0.660 | 0.642/0.662 | 0.770 | 10240/10240 | 107 |
| tool_json | 40/40 | 308.1 | 309.8 | 309.0 | 0.5% | 837.2 | 795.0 | 816.1 | 5.2% | 0.580/0.586 | 0.570/0.565 | 0.831 | 10240/10240 | 104 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

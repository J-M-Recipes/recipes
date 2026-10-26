## per-class decode wall-clock — BR6-B-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 234.4 | 242.9 | 238.7 | 3.6% | 525.2 | 534.1 | 529.6 | 1.7% | 0.423/0.427 | 0.421/0.418 | 0.247 | 10240/10240 | 187 |
| structured | 40/40 | 259.4 | 259.7 | 259.6 | 0.1% | 583.1 | 591.1 | 587.1 | 1.4% | 0.458/0.458 | 0.432/0.452 | 0.468 | 10240/10240 | 196 |
| code | 40/40 | 319.7 | 324.8 | 322.2 | 1.6% | 717.0 | 748.0 | 732.5 | 4.2% | 0.665/0.677 | 0.657/0.670 | 0.784 | 10240/10240 | 116 |
| shell_ops | 40/40 | 334.5 | 343.1 | 338.8 | 2.5% | 831.9 | 821.3 | 826.6 | 1.3% | 0.654/0.675 | 0.663/0.658 | 0.770 | 10240/10240 | 117 |
| tool_json | 40/40 | 319.8 | 329.2 | 324.5 | 2.9% | 889.0 | 805.8 | 847.4 | 9.8% | 0.606/0.627 | 0.621/0.561 | 0.831 | 10240/10240 | 95 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

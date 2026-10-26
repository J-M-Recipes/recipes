## per-class decode wall-clock — BR3-B-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 225.7 | 236.6 | 231.2 | 4.7% | 512.1 | 521.0 | 516.6 | 1.7% | 0.408/0.414 | 0.410/0.427 | 0.247 | 10240/10240 | 160 |
| structured | 40/40 | 258.9 | 242.6 | 250.7 | 6.5% | 579.0 | 585.7 | 582.4 | 1.2% | 0.463/0.428 | 0.441/0.434 | 0.468 | 10240/10240 | 193 |
| code | 40/40 | 319.9 | 312.5 | 316.2 | 2.3% | 705.2 | 720.3 | 712.8 | 2.1% | 0.670/0.661 | 0.644/0.674 | 0.784 | 10240/10240 | 114 |
| shell_ops | 40/40 | 339.6 | 339.7 | 339.6 | 0.0% | 803.8 | 792.6 | 798.2 | 1.4% | 0.668/0.685 | 0.654/0.662 | 0.770 | 10240/10240 | 111 |
| tool_json | 40/40 | 306.6 | 298.3 | 302.4 | 2.7% | 801.9 | 840.8 | 821.4 | 4.7% | 0.569/0.561 | 0.593/0.585 | 0.831 | 10240/10240 | 94 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

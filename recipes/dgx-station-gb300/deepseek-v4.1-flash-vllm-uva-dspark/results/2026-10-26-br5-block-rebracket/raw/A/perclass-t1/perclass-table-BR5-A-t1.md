## per-class decode wall-clock — BR5-A-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 215.9 | 221.4 | 218.7 | 2.5% | 481.1 | 508.2 | 494.7 | 5.5% | 0.363/0.368 | 0.358/0.366 | 0.247 | 10240/10240 | 161 |
| structured | 40/40 | 235.5 | 243.2 | 239.4 | 3.2% | 598.9 | 610.9 | 604.9 | 2.0% | 0.398/0.413 | 0.413/0.413 | 0.468 | 10240/10240 | 212 |
| code | 40/40 | 276.1 | 266.6 | 271.4 | 3.5% | 606.1 | 636.3 | 621.2 | 4.9% | 0.565/0.535 | 0.571/0.574 | 0.784 | 10240/10240 | 124 |
| shell_ops | 40/40 | 320.7 | 321.8 | 321.2 | 0.4% | 730.5 | 717.9 | 724.2 | 1.7% | 0.624/0.617 | 0.613/0.620 | 0.770 | 10240/10240 | 121 |
| tool_json | 40/40 | 279.7 | 285.6 | 282.6 | 2.1% | 772.2 | 764.2 | 768.2 | 1.0% | 0.514/0.530 | 0.518/0.524 | 0.831 | 10240/10240 | 108 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

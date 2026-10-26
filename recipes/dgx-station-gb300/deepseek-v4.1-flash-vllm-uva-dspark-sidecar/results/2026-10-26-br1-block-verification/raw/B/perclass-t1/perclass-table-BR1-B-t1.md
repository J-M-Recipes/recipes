## per-class decode wall-clock — BR1-B-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 278.1 | 278.5 | 278.3 | 0.1% | 997.9 | 968.1 | 983.0 | 3.0% | 0.389/0.390 | 0.522/0.503 | 0.263 | 10240/10240 | 160 |
| structured | 40/40 | 306.7 | 303.8 | 305.3 | 0.9% | 1053.6 | 1082.8 | 1068.2 | 2.7% | 0.453/0.449 | 0.568/0.585 | 0.458 | 10240/10240 | 219 |
| code | 40/40 | 350.4 | 364.3 | 357.3 | 3.9% | 1258.9 | 1183.3 | 1221.1 | 6.2% | 0.556/0.586 | 0.723/0.681 | 0.749 | 10240/10240 | 116 |
| shell_ops | 40/40 | 378.3 | 378.0 | 378.1 | 0.1% | 1315.5 | 1325.2 | 1320.3 | 0.7% | 0.615/0.615 | 0.740/0.746 | 0.748 | 10240/10240 | 110 |
| tool_json | 40/40 | 341.4 | 353.8 | 347.6 | 3.6% | 1214.5 | 1190.5 | 1202.5 | 2.0% | 0.536/0.561 | 0.670/0.645 | 0.811 | 10240/10240 | 90 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

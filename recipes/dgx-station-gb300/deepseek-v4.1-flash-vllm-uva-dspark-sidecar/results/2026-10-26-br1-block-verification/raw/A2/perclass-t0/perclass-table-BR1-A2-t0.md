## per-class decode wall-clock — BR1-A2-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 294.5 | 289.1 | 291.8 | 1.8% | 1040.6 | 1012.5 | 1026.5 | 2.7% | 0.421/0.409 | 0.553/0.542 | 0.243 | 10240/10240 | 160 |
| structured | 40/40 | 298.6 | 297.9 | 298.2 | 0.2% | 985.2 | 1001.2 | 993.2 | 1.6% | 0.429/0.430 | 0.528/0.555 | 0.463 | 10240/10240 | 210 |
| code | 40/40 | 411.8 | 404.2 | 408.0 | 1.9% | 1277.2 | 1240.8 | 1259.0 | 2.9% | 0.685/0.672 | 0.753/0.743 | 0.772 | 10240/10240 | 126 |
| shell_ops | 40/40 | 393.9 | 400.3 | 397.1 | 1.6% | 1371.1 | 1319.7 | 1345.4 | 3.8% | 0.645/0.659 | 0.770/0.768 | 0.760 | 10240/10240 | 118 |
| tool_json | 40/40 | 370.2 | 363.9 | 367.1 | 1.7% | 1277.8 | 1243.2 | 1260.5 | 2.7% | 0.596/0.578 | 0.689/0.697 | 0.865 | 10240/10240 | 113 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

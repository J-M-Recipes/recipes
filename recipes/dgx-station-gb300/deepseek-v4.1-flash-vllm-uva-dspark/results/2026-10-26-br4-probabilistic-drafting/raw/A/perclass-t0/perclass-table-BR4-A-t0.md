## per-class decode wall-clock — BR4-A-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 235.1 | 231.6 | 233.3 | 1.5% | 513.3 | 492.0 | 502.6 | 4.2% | 0.435/0.406 | 0.428/0.406 | 0.247 | 10240/10240 | 167 |
| structured | 40/40 | 254.2 | 254.6 | 254.4 | 0.1% | 566.7 | 561.2 | 563.9 | 1.0% | 0.453/0.458 | 0.416/0.421 | 0.468 | 10240/10240 | 186 |
| code | 40/40 | 325.6 | 308.8 | 317.2 | 5.3% | 758.2 | 734.0 | 746.1 | 3.2% | 0.694/0.648 | 0.711/0.670 | 0.784 | 10240/10240 | 130 |
| shell_ops | 40/40 | 343.1 | 346.5 | 344.8 | 1.0% | 819.0 | 817.0 | 818.0 | 0.2% | 0.676/0.691 | 0.641/0.659 | 0.770 | 10240/10240 | 117 |
| tool_json | 40/40 | 305.3 | 301.1 | 303.2 | 1.4% | 850.2 | 857.2 | 853.7 | 0.8% | 0.576/0.565 | 0.604/0.586 | 0.831 | 10240/10240 | 109 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

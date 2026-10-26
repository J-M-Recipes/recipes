## per-class decode wall-clock — BR3-A-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 201.6 | 199.9 | 200.7 | 0.9% | 440.7 | 452.7 | 446.7 | 2.7% | 0.327/0.313 | 0.307/0.306 | 0.247 | 10240/10240 | 150 |
| structured | 40/40 | 215.9 | 213.5 | 214.7 | 1.1% | 560.4 | 526.1 | 543.2 | 6.3% | 0.347/0.339 | 0.376/0.353 | 0.468 | 10240/10240 | 195 |
| code | 40/40 | 262.9 | 247.3 | 255.1 | 6.1% | 520.4 | 524.4 | 522.4 | 0.8% | 0.523/0.479 | 0.489/0.478 | 0.784 | 10240/10240 | 118 |
| shell_ops | 40/40 | 281.3 | 292.9 | 287.1 | 4.0% | 653.2 | 627.1 | 640.2 | 4.1% | 0.529/0.552 | 0.546/0.539 | 0.770 | 10240/10240 | 156 |
| tool_json | 40/40 | 264.4 | 246.6 | 255.5 | 7.0% | 651.4 | 630.6 | 641.0 | 3.2% | 0.462/0.426 | 0.442/0.459 | 0.831 | 10240/10240 | 112 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

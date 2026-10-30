## per-class decode wall-clock — G1-Pg-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 372.3 | 366.5 | 369.4 | 1.6% | 1076.6 | 1113.0 | 1094.8 | 3.3% | 0.357/0.357 | 0.514/0.517 | 0.260 | 10240/10240 | 148 |
| structured | 40/40 | 412.3 | 420.3 | 416.3 | 1.9% | 1197.2 | 1183.0 | 1190.1 | 1.2% | 0.410/0.419 | 0.571/0.574 | 0.404 | 10240/10240 | 202 |
| code | 40/40 | 516.6 | 523.3 | 520.0 | 1.3% | 1372.5 | 1302.6 | 1337.6 | 5.2% | 0.478/0.483 | 0.702/0.667 | 0.687 | 10240/10240 | 132 |
| shell_ops | 40/40 | 514.3 | 548.0 | 531.2 | 6.3% | 1453.6 | 1468.0 | 1460.8 | 1.0% | 0.467/0.486 | 0.734/0.743 | 0.588 | 10240/10240 | 122 |
| tool_json | 40/40 | 457.2 | 431.2 | 444.2 | 5.9% | 1309.0 | 1317.2 | 1313.1 | 0.6% | 0.432/0.412 | 0.647/0.637 | 0.741 | 10240/10240 | 94 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

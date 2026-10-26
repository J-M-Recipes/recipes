## per-class decode wall-clock — BR3-A2-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 197.4 | 206.9 | 202.1 | 4.7% | 473.2 | 465.7 | 469.4 | 1.6% | 0.317/0.337 | 0.326/0.334 | 0.247 | 10240/10240 | 164 |
| structured | 40/40 | 218.2 | 217.3 | 217.8 | 0.4% | 544.2 | 535.0 | 539.6 | 1.7% | 0.351/0.347 | 0.339/0.334 | 0.468 | 10240/10240 | 224 |
| code | 40/40 | 267.9 | 252.4 | 260.2 | 5.9% | 569.8 | 520.1 | 545.0 | 9.1% | 0.538/0.508 | 0.489/0.486 | 0.784 | 10240/10240 | 117 |
| shell_ops | 40/40 | 282.3 | 288.7 | 285.5 | 2.3% | 647.5 | 654.3 | 650.9 | 1.0% | 0.524/0.544 | 0.526/0.545 | 0.770 | 10240/10240 | 146 |
| tool_json | 40/40 | 258.0 | 248.8 | 253.4 | 3.6% | 661.8 | 741.9 | 701.8 | 11.4% | 0.449/0.432 | 0.434/0.487 | 0.831 | 10240/10240 | 106 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

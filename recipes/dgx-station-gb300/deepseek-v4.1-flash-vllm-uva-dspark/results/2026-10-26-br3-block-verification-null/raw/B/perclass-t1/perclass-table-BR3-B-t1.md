## per-class decode wall-clock — BR3-B-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 196.6 | 206.3 | 201.4 | 4.8% | 454.9 | 451.2 | 453.0 | 0.8% | 0.314/0.333 | 0.328/0.312 | 0.247 | 10240/10240 | 180 |
| structured | 40/40 | 210.9 | 209.7 | 210.3 | 0.6% | 511.5 | 539.0 | 525.2 | 5.2% | 0.334/0.334 | 0.324/0.358 | 0.468 | 10240/10240 | 219 |
| code | 40/40 | 261.2 | 245.8 | 253.5 | 6.1% | 593.9 | 567.5 | 580.7 | 4.5% | 0.507/0.481 | 0.513/0.514 | 0.784 | 10240/10240 | 124 |
| shell_ops | 40/40 | 282.0 | 288.0 | 285.0 | 2.1% | 688.4 | 660.0 | 674.2 | 4.2% | 0.521/0.538 | 0.549/0.523 | 0.770 | 10240/10240 | 134 |
| tool_json | 40/40 | 263.9 | 252.5 | 258.2 | 4.4% | 700.9 | 757.7 | 729.3 | 7.8% | 0.473/0.452 | 0.460/0.503 | 0.831 | 10240/10240 | 105 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

## per-class decode wall-clock — BR6-A2-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 211.6 | 215.3 | 213.4 | 1.8% | 485.7 | 490.6 | 488.1 | 1.0% | 0.352/0.352 | 0.370/0.361 | 0.247 | 10240/10240 | 153 |
| structured | 40/40 | 243.8 | 245.0 | 244.4 | 0.5% | 604.4 | 597.2 | 600.8 | 1.2% | 0.413/0.412 | 0.418/0.403 | 0.468 | 10240/10240 | 223 |
| code | 40/40 | 278.9 | 278.9 | 278.9 | 0.0% | 639.4 | 634.3 | 636.9 | 0.8% | 0.564/0.574 | 0.554/0.561 | 0.784 | 10240/10240 | 138 |
| shell_ops | 40/40 | 310.4 | 324.2 | 317.3 | 4.3% | 747.2 | 724.2 | 735.7 | 3.1% | 0.606/0.636 | 0.618/0.602 | 0.770 | 10240/10240 | 134 |
| tool_json | 40/40 | 280.4 | 296.7 | 288.6 | 5.7% | 783.2 | 796.6 | 789.9 | 1.7% | 0.532/0.545 | 0.550/0.538 | 0.831 | 10240/10240 | 123 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

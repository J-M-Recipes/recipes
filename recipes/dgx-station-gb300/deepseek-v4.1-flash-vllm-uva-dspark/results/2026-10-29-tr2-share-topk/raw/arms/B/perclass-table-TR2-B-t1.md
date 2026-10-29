## per-class decode wall-clock — TR2-B-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 215.1 | 224.4 | 219.7 | 4.2% | 491.7 | 496.5 | 494.1 | 1.0% | 0.368/0.372 | 0.361/0.372 | 0.247 | 10240/10240 | 176 |
| structured | 40/40 | 240.7 | 250.1 | 245.4 | 3.8% | 591.6 | 583.8 | 587.7 | 1.3% | 0.406/0.427 | 0.430/0.406 | 0.468 | 10240/10240 | 222 |
| code | 40/40 | 271.7 | 264.8 | 268.2 | 2.6% | 604.2 | 650.3 | 627.2 | 7.3% | 0.545/0.536 | 0.536/0.550 | 0.784 | 10240/10240 | 122 |
| shell_ops | 40/40 | 325.1 | 308.5 | 316.8 | 5.2% | 688.3 | 683.8 | 686.1 | 0.7% | 0.615/0.587 | 0.608/0.616 | 0.770 | 10240/10240 | 162 |
| tool_json | 40/40 | 297.8 | 298.3 | 298.1 | 0.2% | 747.9 | 756.5 | 752.2 | 1.1% | 0.541/0.544 | 0.508/0.536 | 0.831 | 10240/10240 | 130 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

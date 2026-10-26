## per-class decode wall-clock — BR5-A2-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 220.8 | 219.2 | 220.0 | 0.8% | 486.0 | 499.3 | 492.6 | 2.7% | 0.371/0.356 | 0.360/0.366 | 0.247 | 10240/10240 | 194 |
| structured | 40/40 | 249.0 | 239.1 | 244.1 | 4.0% | 580.1 | 600.0 | 590.1 | 3.4% | 0.424/0.401 | 0.390/0.413 | 0.468 | 10240/10240 | 192 |
| code | 40/40 | 276.1 | 270.1 | 273.1 | 2.2% | 697.6 | 616.2 | 656.9 | 12.4% | 0.560/0.553 | 0.610/0.585 | 0.784 | 10240/10240 | 138 |
| shell_ops | 40/40 | 320.2 | 330.2 | 325.2 | 3.1% | 770.6 | 681.3 | 726.0 | 12.3% | 0.613/0.644 | 0.640/0.583 | 0.770 | 10240/10240 | 142 |
| tool_json | 40/40 | 293.2 | 292.3 | 292.7 | 0.3% | 771.5 | 753.4 | 762.4 | 2.4% | 0.544/0.541 | 0.519/0.516 | 0.831 | 10240/10240 | 118 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

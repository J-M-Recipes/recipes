## per-class decode wall-clock — BR4-PB-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 215.6 | 224.1 | 219.9 | 3.9% | 510.9 | 509.2 | 510.1 | 0.3% | 0.373/0.384 | 0.386/0.389 | 0.247 | 10240/10240 | 185 |
| structured | 40/40 | 250.5 | 246.2 | 248.4 | 1.7% | 603.6 | 619.1 | 611.4 | 2.5% | 0.438/0.424 | 0.417/0.417 | 0.468 | 10240/10240 | 214 |
| code | 40/40 | 276.9 | 279.7 | 278.3 | 1.0% | 623.3 | 666.0 | 644.6 | 6.6% | 0.583/0.556 | 0.550/0.604 | 0.784 | 10240/10240 | 126 |
| shell_ops | 40/40 | 321.9 | 325.0 | 323.4 | 1.0% | 722.6 | 755.7 | 739.1 | 4.5% | 0.618/0.622 | 0.580/0.624 | 0.770 | 10240/10240 | 124 |
| tool_json | 40/40 | 298.0 | 286.0 | 292.0 | 4.1% | 774.0 | 786.0 | 780.0 | 1.5% | 0.550/0.544 | 0.528/0.562 | 0.831 | 10240/10240 | 94 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

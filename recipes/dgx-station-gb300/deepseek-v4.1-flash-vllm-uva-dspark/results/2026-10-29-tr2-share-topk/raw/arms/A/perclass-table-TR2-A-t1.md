## per-class decode wall-clock — TR2-A-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 214.2 | 217.6 | 215.9 | 1.6% | 492.6 | 479.8 | 486.2 | 2.6% | 0.372/0.356 | 0.369/0.359 | 0.247 | 10240/10240 | 134 |
| structured | 40/40 | 246.3 | 241.5 | 243.9 | 2.0% | 595.8 | 599.0 | 597.4 | 0.5% | 0.422/0.408 | 0.409/0.424 | 0.468 | 10240/10240 | 224 |
| code | 40/40 | 278.5 | 263.9 | 271.2 | 5.4% | 656.5 | 648.0 | 652.2 | 1.3% | 0.570/0.543 | 0.582/0.582 | 0.784 | 10240/10240 | 136 |
| shell_ops | 40/40 | 309.7 | 311.5 | 310.6 | 0.6% | 735.8 | 745.8 | 740.8 | 1.3% | 0.607/0.607 | 0.597/0.619 | 0.770 | 10240/10240 | 142 |
| tool_json | 40/40 | 287.5 | 300.8 | 294.1 | 4.5% | 717.4 | 739.0 | 728.2 | 3.0% | 0.534/0.560 | 0.544/0.528 | 0.831 | 10240/10240 | 96 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

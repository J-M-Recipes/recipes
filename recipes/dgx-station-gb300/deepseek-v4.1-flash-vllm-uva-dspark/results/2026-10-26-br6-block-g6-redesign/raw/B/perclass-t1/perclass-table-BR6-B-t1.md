## per-class decode wall-clock — BR6-B-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 220.8 | 221.1 | 220.9 | 0.1% | 519.0 | 527.6 | 523.3 | 1.7% | 0.372/0.373 | 0.381/0.381 | 0.247 | 10240/10240 | 160 |
| structured | 40/40 | 240.0 | 248.1 | 244.1 | 3.3% | 602.8 | 628.0 | 615.4 | 4.1% | 0.414/0.436 | 0.414/0.437 | 0.468 | 10240/10240 | 190 |
| code | 40/40 | 294.7 | 291.9 | 293.3 | 0.9% | 608.6 | 685.0 | 646.8 | 11.8% | 0.599/0.599 | 0.566/0.622 | 0.784 | 10240/10240 | 152 |
| shell_ops | 40/40 | 295.1 | 322.4 | 308.8 | 8.8% | 746.0 | 725.5 | 735.8 | 2.8% | 0.579/0.638 | 0.626/0.601 | 0.770 | 10240/10240 | 142 |
| tool_json | 40/40 | 284.4 | 285.5 | 284.9 | 0.4% | 797.4 | 734.1 | 765.7 | 8.3% | 0.532/0.530 | 0.557/0.536 | 0.831 | 10240/10240 | 114 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

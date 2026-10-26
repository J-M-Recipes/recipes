## per-class decode wall-clock — BR4-A-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 197.3 | 197.6 | 197.4 | 0.1% | 441.1 | 462.4 | 451.8 | 4.7% | 0.326/0.305 | 0.322/0.316 | 0.247 | 10240/10240 | 153 |
| structured | 40/40 | 213.5 | 224.6 | 219.0 | 5.1% | 539.7 | 539.8 | 539.7 | 0.0% | 0.339/0.369 | 0.340/0.361 | 0.468 | 10240/10240 | 198 |
| code | 40/40 | 267.1 | 243.7 | 255.4 | 9.2% | 603.4 | 558.4 | 580.9 | 7.8% | 0.529/0.484 | 0.512/0.513 | 0.784 | 10240/10240 | 116 |
| shell_ops | 40/40 | 295.8 | 288.0 | 291.9 | 2.7% | 665.5 | 618.9 | 642.2 | 7.3% | 0.552/0.542 | 0.535/0.504 | 0.770 | 10240/10240 | 160 |
| tool_json | 40/40 | 251.4 | 257.7 | 254.6 | 2.4% | 663.1 | 680.7 | 671.9 | 2.6% | 0.448/0.447 | 0.440/0.438 | 0.831 | 10240/10240 | 118 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

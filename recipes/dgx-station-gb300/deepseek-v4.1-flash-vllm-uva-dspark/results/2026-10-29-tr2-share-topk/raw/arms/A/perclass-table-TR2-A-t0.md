## per-class decode wall-clock — TR2-A-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 233.4 | 243.5 | 238.4 | 4.2% | 518.3 | 523.2 | 520.8 | 0.9% | 0.419/0.431 | 0.410/0.425 | 0.247 | 10240/10240 | 166 |
| structured | 40/40 | 248.8 | 254.7 | 251.7 | 2.4% | 564.4 | 565.5 | 565.0 | 0.2% | 0.441/0.447 | 0.432/0.426 | 0.468 | 10240/10240 | 196 |
| code | 40/40 | 319.4 | 330.7 | 325.0 | 3.5% | 710.7 | 710.0 | 710.3 | 0.1% | 0.677/0.697 | 0.653/0.664 | 0.784 | 10240/10240 | 115 |
| shell_ops | 40/40 | 347.6 | 338.1 | 342.9 | 2.8% | 791.0 | 775.8 | 783.4 | 1.9% | 0.684/0.657 | 0.663/0.658 | 0.770 | 10240/10240 | 116 |
| tool_json | 40/40 | 313.7 | 313.9 | 313.8 | 0.1% | 829.1 | 838.6 | 833.9 | 1.1% | 0.583/0.587 | 0.577/0.561 | 0.831 | 10240/10240 | 106 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

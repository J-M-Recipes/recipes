## per-class decode wall-clock — BR3-A2-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 233.8 | 249.4 | 241.6 | 6.5% | 535.2 | 547.0 | 541.1 | 2.2% | 0.428/0.448 | 0.415/0.417 | 0.247 | 10240/10240 | 184 |
| structured | 40/40 | 247.2 | 250.5 | 248.8 | 1.3% | 552.8 | 575.6 | 564.2 | 4.0% | 0.442/0.452 | 0.425/0.424 | 0.468 | 10240/10240 | 200 |
| code | 40/40 | 320.1 | 302.8 | 311.4 | 5.5% | 738.7 | 723.8 | 731.2 | 2.0% | 0.686/0.640 | 0.676/0.668 | 0.784 | 10240/10240 | 110 |
| shell_ops | 40/40 | 334.2 | 350.7 | 342.4 | 4.8% | 816.6 | 794.0 | 805.3 | 2.8% | 0.675/0.710 | 0.671/0.671 | 0.770 | 10240/10240 | 117 |
| tool_json | 40/40 | 289.3 | 303.0 | 296.2 | 4.6% | 812.2 | 849.2 | 830.7 | 4.5% | 0.538/0.569 | 0.564/0.598 | 0.831 | 10240/10240 | 96 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

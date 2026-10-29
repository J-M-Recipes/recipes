## per-class decode wall-clock — TR2-A2-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 236.3 | 238.4 | 237.4 | 0.9% | 516.1 | 540.4 | 528.3 | 4.6% | 0.434/0.414 | 0.420/0.425 | 0.247 | 10240/10240 | 160 |
| structured | 40/40 | 258.6 | 259.9 | 259.3 | 0.5% | 596.4 | 553.4 | 574.9 | 7.5% | 0.456/0.471 | 0.438/0.440 | 0.468 | 10240/10240 | 178 |
| code | 40/40 | 326.7 | 320.8 | 323.8 | 1.8% | 742.3 | 720.9 | 731.6 | 2.9% | 0.681/0.679 | 0.688/0.675 | 0.784 | 10240/10240 | 120 |
| shell_ops | 40/40 | 347.6 | 353.3 | 350.4 | 1.6% | 806.2 | 776.3 | 791.2 | 3.8% | 0.701/0.700 | 0.650/0.664 | 0.770 | 10240/10240 | 117 |
| tool_json | 40/40 | 305.4 | 299.0 | 302.2 | 2.1% | 861.0 | 788.2 | 824.6 | 8.8% | 0.572/0.549 | 0.582/0.548 | 0.831 | 10240/10240 | 109 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

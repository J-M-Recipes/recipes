## per-class decode wall-clock — TR2-B-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 233.2 | 246.5 | 239.9 | 5.5% | 520.5 | 506.1 | 513.3 | 2.8% | 0.421/0.438 | 0.415/0.409 | 0.247 | 10240/10240 | 160 |
| structured | 40/40 | 255.8 | 253.9 | 254.8 | 0.8% | 551.6 | 556.2 | 553.9 | 0.8% | 0.455/0.456 | 0.431/0.421 | 0.468 | 10240/10240 | 188 |
| code | 40/40 | 326.6 | 326.1 | 326.3 | 0.1% | 716.1 | 750.3 | 733.2 | 4.7% | 0.684/0.689 | 0.678/0.679 | 0.784 | 10240/10240 | 122 |
| shell_ops | 40/40 | 355.4 | 351.4 | 353.4 | 1.1% | 795.5 | 816.3 | 805.9 | 2.6% | 0.697/0.685 | 0.663/0.664 | 0.770 | 10240/10240 | 118 |
| tool_json | 40/40 | 310.5 | 314.3 | 312.4 | 1.2% | 844.8 | 831.2 | 838.0 | 1.6% | 0.580/0.581 | 0.596/0.573 | 0.831 | 10240/10240 | 93 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

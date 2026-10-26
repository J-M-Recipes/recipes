## per-class decode wall-clock — BR4-A2-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 227.4 | 238.2 | 232.8 | 4.7% | 538.3 | 509.9 | 524.1 | 5.4% | 0.411/0.415 | 0.420/0.410 | 0.247 | 10240/10240 | 160 |
| structured | 40/40 | 264.1 | 248.4 | 256.3 | 6.1% | 575.4 | 558.9 | 567.1 | 2.9% | 0.476/0.445 | 0.427/0.430 | 0.468 | 10240/10240 | 200 |
| code | 40/40 | 323.0 | 323.9 | 323.5 | 0.3% | 706.3 | 757.3 | 731.8 | 7.0% | 0.673/0.689 | 0.664/0.693 | 0.784 | 10240/10240 | 116 |
| shell_ops | 40/40 | 344.7 | 350.7 | 347.7 | 1.7% | 816.8 | 778.1 | 797.5 | 4.9% | 0.688/0.701 | 0.660/0.642 | 0.770 | 10240/10240 | 118 |
| tool_json | 40/40 | 290.1 | 316.6 | 303.4 | 8.7% | 805.2 | 800.4 | 802.8 | 0.6% | 0.538/0.594 | 0.563/0.558 | 0.831 | 10240/10240 | 106 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

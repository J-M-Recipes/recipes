## per-class decode wall-clock — TR2-A2-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 217.1 | 214.0 | 215.5 | 1.4% | 507.8 | 492.8 | 500.3 | 3.0% | 0.370/0.355 | 0.361/0.351 | 0.247 | 10240/10240 | 167 |
| structured | 40/40 | 247.4 | 243.9 | 245.7 | 1.4% | 589.1 | 605.6 | 597.4 | 2.8% | 0.420/0.416 | 0.414/0.411 | 0.468 | 10240/10240 | 220 |
| code | 40/40 | 274.5 | 271.2 | 272.8 | 1.2% | 654.4 | 661.4 | 657.9 | 1.1% | 0.561/0.553 | 0.557/0.604 | 0.784 | 10240/10240 | 125 |
| shell_ops | 40/40 | 309.2 | 305.4 | 307.3 | 1.2% | 699.7 | 729.9 | 714.8 | 4.2% | 0.588/0.586 | 0.569/0.606 | 0.770 | 10240/10240 | 144 |
| tool_json | 40/40 | 284.5 | 295.8 | 290.2 | 3.9% | 812.1 | 807.1 | 809.6 | 0.6% | 0.523/0.539 | 0.538/0.564 | 0.831 | 10240/10240 | 130 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

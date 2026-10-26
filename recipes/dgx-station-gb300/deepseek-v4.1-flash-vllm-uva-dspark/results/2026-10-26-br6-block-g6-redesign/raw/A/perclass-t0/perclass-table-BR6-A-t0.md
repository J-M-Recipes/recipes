## per-class decode wall-clock — BR6-A-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 234.0 | 234.6 | 234.3 | 0.2% | 516.2 | 509.4 | 512.8 | 1.3% | 0.422/0.402 | 0.416/0.424 | 0.247 | 10240/10240 | 172 |
| structured | 40/40 | 252.0 | 253.2 | 252.6 | 0.5% | 570.1 | 557.2 | 563.6 | 2.3% | 0.447/0.446 | 0.427/0.442 | 0.468 | 10240/10240 | 200 |
| code | 40/40 | 324.7 | 316.6 | 320.7 | 2.5% | 733.9 | 741.4 | 737.7 | 1.0% | 0.686/0.666 | 0.675/0.697 | 0.784 | 10240/10240 | 114 |
| shell_ops | 40/40 | 347.6 | 349.5 | 348.5 | 0.6% | 790.2 | 828.7 | 809.5 | 4.8% | 0.679/0.686 | 0.668/0.674 | 0.770 | 10240/10240 | 116 |
| tool_json | 40/40 | 295.6 | 308.8 | 302.2 | 4.4% | 816.8 | 849.7 | 833.3 | 4.0% | 0.538/0.574 | 0.571/0.592 | 0.831 | 10240/10240 | 104 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

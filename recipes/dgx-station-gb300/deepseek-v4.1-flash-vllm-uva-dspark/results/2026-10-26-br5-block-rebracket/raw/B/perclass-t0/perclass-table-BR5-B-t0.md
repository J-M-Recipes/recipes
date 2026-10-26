## per-class decode wall-clock — BR5-B-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 227.5 | 239.1 | 233.3 | 5.0% | 510.4 | 540.6 | 525.5 | 5.7% | 0.407/0.421 | 0.419/0.416 | 0.247 | 10240/10240 | 164 |
| structured | 40/40 | 256.9 | 255.6 | 256.2 | 0.5% | 575.5 | 572.8 | 574.1 | 0.5% | 0.452/0.449 | 0.433/0.446 | 0.468 | 10240/10240 | 207 |
| code | 40/40 | 325.0 | 326.5 | 325.8 | 0.5% | 733.7 | 749.6 | 741.7 | 2.1% | 0.681/0.684 | 0.694/0.695 | 0.784 | 10240/10240 | 133 |
| shell_ops | 40/40 | 353.3 | 343.4 | 348.3 | 2.9% | 841.2 | 831.0 | 836.1 | 1.2% | 0.698/0.673 | 0.677/0.671 | 0.770 | 10240/10240 | 117 |
| tool_json | 40/40 | 310.2 | 308.7 | 309.4 | 0.5% | 805.6 | 842.6 | 824.1 | 4.5% | 0.582/0.580 | 0.586/0.580 | 0.831 | 10240/10240 | 104 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

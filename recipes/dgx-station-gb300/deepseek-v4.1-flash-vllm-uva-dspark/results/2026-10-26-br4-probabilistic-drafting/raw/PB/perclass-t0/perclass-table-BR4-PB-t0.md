## per-class decode wall-clock — BR4-PB-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 228.6 | 242.6 | 235.6 | 5.9% | 520.1 | 525.4 | 522.7 | 1.0% | 0.410/0.427 | 0.416/0.408 | 0.247 | 10240/10240 | 164 |
| structured | 40/40 | 250.8 | 255.3 | 253.1 | 1.8% | 583.2 | 561.8 | 572.5 | 3.7% | 0.452/0.447 | 0.435/0.446 | 0.468 | 10240/10240 | 217 |
| code | 40/40 | 308.4 | 330.5 | 319.4 | 6.9% | 743.4 | 706.2 | 724.8 | 5.1% | 0.654/0.691 | 0.675/0.658 | 0.784 | 10240/10240 | 128 |
| shell_ops | 40/40 | 357.9 | 359.2 | 358.6 | 0.4% | 840.8 | 845.8 | 843.2 | 0.6% | 0.708/0.708 | 0.674/0.683 | 0.770 | 10240/10240 | 107 |
| tool_json | 40/40 | 310.8 | 319.3 | 315.0 | 2.7% | 831.2 | 868.8 | 850.0 | 4.4% | 0.585/0.601 | 0.586/0.587 | 0.831 | 10240/10240 | 96 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

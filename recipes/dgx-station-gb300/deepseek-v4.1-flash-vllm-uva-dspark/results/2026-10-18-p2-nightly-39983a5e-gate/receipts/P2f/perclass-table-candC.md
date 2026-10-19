## per-class decode wall-clock — candC

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 240.3 | 243.0 | 241.6 | 1.1% | 575.8 | 553.7 | 564.8 | 3.9% | 0.404/0.406 | 0.438/0.424 | 0.249 | 10240/10240 | 158 |
| structured | 40/40 | 256.8 | 254.1 | 255.5 | 1.1% | 576.1 | 571.4 | 573.8 | 0.8% | 0.431/0.427 | 0.434/0.430 | 0.452 | 10240/10240 | 189 |
| code | 40/40 | 308.2 | 317.0 | 312.6 | 2.8% | 676.3 | 670.1 | 673.2 | 0.9% | 0.653/0.676 | 0.685/0.669 | 0.736 | 10240/10240 | 120 |
| shell_ops | 40/40 | 337.3 | 325.8 | 331.6 | 3.5% | 726.7 | 751.5 | 739.1 | 3.4% | 0.692/0.649 | 0.666/0.678 | 0.774 | 10240/10240 | 120 |
| tool_json | 40/40 | 320.5 | 313.3 | 316.9 | 2.3% | 757.7 | 776.5 | 767.1 | 2.4% | 0.604/0.589 | 0.582/0.609 | 0.848 | 10240/10240 | 104 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

## per-class decode wall-clock — BR6-A2-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 230.4 | 240.0 | 235.2 | 4.0% | 525.6 | 525.6 | 525.6 | 0.0% | 0.413/0.417 | 0.418/0.413 | 0.247 | 10240/10240 | 170 |
| structured | 40/40 | 250.8 | 259.7 | 255.2 | 3.5% | 593.7 | 564.5 | 579.1 | 5.0% | 0.435/0.456 | 0.453/0.417 | 0.468 | 10240/10240 | 203 |
| code | 40/40 | 325.3 | 326.0 | 325.6 | 0.2% | 723.5 | 735.7 | 729.6 | 1.7% | 0.685/0.682 | 0.663/0.703 | 0.784 | 10240/10240 | 118 |
| shell_ops | 40/40 | 350.6 | 353.9 | 352.2 | 0.9% | 769.4 | 784.3 | 776.8 | 1.9% | 0.690/0.695 | 0.659/0.651 | 0.770 | 10240/10240 | 115 |
| tool_json | 40/40 | 305.9 | 311.8 | 308.9 | 1.9% | 868.1 | 861.3 | 864.7 | 0.8% | 0.574/0.583 | 0.590/0.590 | 0.831 | 10240/10240 | 96 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

## per-class decode wall-clock — BR5-A-t0

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 234.5 | 240.8 | 237.6 | 2.6% | 526.4 | 519.9 | 523.1 | 1.2% | 0.432/0.420 | 0.417/0.425 | 0.247 | 10240/10240 | 165 |
| structured | 40/40 | 261.4 | 255.1 | 258.2 | 2.5% | 557.4 | 589.2 | 573.3 | 5.5% | 0.467/0.450 | 0.438/0.426 | 0.468 | 10240/10240 | 190 |
| code | 40/40 | 322.8 | 328.7 | 325.7 | 1.8% | 740.1 | 740.4 | 740.2 | 0.0% | 0.672/0.693 | 0.676/0.680 | 0.784 | 10240/10240 | 126 |
| shell_ops | 40/40 | 339.3 | 344.9 | 342.1 | 1.6% | 800.0 | 793.1 | 796.5 | 0.9% | 0.662/0.678 | 0.648/0.640 | 0.770 | 10240/10240 | 114 |
| tool_json | 40/40 | 315.5 | 323.6 | 319.6 | 2.5% | 836.0 | 805.8 | 820.9 | 3.7% | 0.595/0.611 | 0.566/0.568 | 0.831 | 10240/10240 | 98 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

## per-class decode wall-clock — BR5-B-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 218.5 | 225.7 | 222.1 | 3.2% | 512.1 | 515.7 | 513.9 | 0.7% | 0.387/0.383 | 0.385/0.388 | 0.247 | 10240/10240 | 176 |
| structured | 40/40 | 241.8 | 253.8 | 247.8 | 4.8% | 619.7 | 610.8 | 615.2 | 1.4% | 0.417/0.438 | 0.424/0.424 | 0.468 | 10240/10240 | 172 |
| code | 40/40 | 288.2 | 285.4 | 286.8 | 1.0% | 648.8 | 692.6 | 670.7 | 6.5% | 0.598/0.566 | 0.581/0.612 | 0.784 | 10240/10240 | 148 |
| shell_ops | 40/40 | 337.5 | 334.0 | 335.7 | 1.0% | 707.3 | 721.5 | 714.4 | 2.0% | 0.665/0.647 | 0.604/0.600 | 0.770 | 10240/10240 | 140 |
| tool_json | 40/40 | 283.6 | 301.3 | 292.4 | 6.0% | 766.2 | 776.3 | 771.2 | 1.3% | 0.526/0.548 | 0.552/0.541 | 0.831 | 10240/10240 | 106 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

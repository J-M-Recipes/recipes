## per-class decode wall-clock — G1-A-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 282.9 | 275.6 | 279.2 | 2.6% | 973.8 | 1018.2 | 996.0 | 4.5% | 0.381/0.367 | 0.497/0.531 | 0.279 | 10240/10240 | 162 |
| structured | 40/40 | 306.1 | 302.9 | 304.5 | 1.0% | 1073.2 | 1065.1 | 1069.2 | 0.8% | 0.434/0.426 | 0.566/0.568 | 0.465 | 10240/10240 | 204 |
| code | 40/40 | 379.4 | 365.3 | 372.3 | 3.8% | 1191.3 | 1230.1 | 1210.7 | 3.2% | 0.595/0.563 | 0.696/0.711 | 0.743 | 10240/10240 | 120 |
| shell_ops | 40/40 | 392.9 | 394.6 | 393.7 | 0.4% | 1280.1 | 1298.2 | 1289.1 | 1.4% | 0.626/0.628 | 0.730/0.743 | 0.745 | 10240/10240 | 122 |
| tool_json | 40/40 | 349.7 | 352.8 | 351.2 | 0.9% | 1154.5 | 1196.2 | 1175.3 | 3.5% | 0.529/0.530 | 0.635/0.673 | 0.811 | 10240/10240 | 118 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

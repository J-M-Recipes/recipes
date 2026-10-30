## per-class decode wall-clock — G1-A2-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 293.1 | 284.5 | 288.8 | 3.0% | 984.6 | 1018.6 | 1001.6 | 3.4% | 0.396/0.378 | 0.524/0.539 | 0.267 | 10240/10240 | 154 |
| structured | 40/40 | 313.0 | 309.7 | 311.4 | 1.1% | 1070.2 | 1074.8 | 1072.5 | 0.4% | 0.437/0.432 | 0.577/0.569 | 0.484 | 10240/10240 | 202 |
| code | 40/40 | 367.7 | 378.9 | 373.3 | 3.0% | 1182.5 | 1226.9 | 1204.7 | 3.7% | 0.561/0.586 | 0.680/0.684 | 0.780 | 10240/10240 | 141 |
| shell_ops | 40/40 | 388.6 | 390.3 | 389.4 | 0.4% | 1255.4 | 1323.3 | 1289.3 | 5.3% | 0.605/0.611 | 0.724/0.753 | 0.757 | 10240/10240 | 122 |
| tool_json | 40/40 | 360.4 | 368.8 | 364.6 | 2.3% | 1211.7 | 1016.6 | 1114.2 | 17.5% | 0.540/0.559 | 0.661/0.595 | 0.845 | 10240/10240 | 119 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

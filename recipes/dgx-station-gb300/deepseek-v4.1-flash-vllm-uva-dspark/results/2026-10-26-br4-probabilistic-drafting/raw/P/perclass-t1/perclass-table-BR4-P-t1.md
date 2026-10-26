## per-class decode wall-clock — BR4-P-t1

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 216.4 | 217.9 | 217.2 | 0.7% | 493.2 | 495.4 | 494.3 | 0.4% | 0.367/0.356 | 0.380/0.374 | 0.247 | 10240/10240 | 180 |
| structured | 40/40 | 245.7 | 238.2 | 241.9 | 3.1% | 584.5 | 594.7 | 589.6 | 1.7% | 0.422/0.397 | 0.411/0.418 | 0.468 | 10240/10240 | 226 |
| code | 40/40 | 266.8 | 273.0 | 269.9 | 2.3% | 642.3 | 611.3 | 626.8 | 5.0% | 0.538/0.553 | 0.583/0.556 | 0.784 | 10240/10240 | 128 |
| shell_ops | 40/40 | 318.6 | 320.9 | 319.8 | 0.7% | 746.9 | 699.7 | 723.3 | 6.5% | 0.611/0.631 | 0.612/0.601 | 0.770 | 10240/10240 | 131 |
| tool_json | 40/40 | 294.8 | 293.2 | 294.0 | 0.5% | 803.9 | 779.6 | 791.8 | 3.1% | 0.556/0.529 | 0.543/0.524 | 0.831 | 10240/10240 | 112 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

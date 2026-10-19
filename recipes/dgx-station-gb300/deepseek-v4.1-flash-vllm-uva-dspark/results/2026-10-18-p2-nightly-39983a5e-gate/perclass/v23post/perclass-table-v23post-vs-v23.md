## per-class decode wall-clock — v23post

FL (forced tokens/prompt): {'prose': 256, 'structured': 256, 'code': 256, 'shell_ops': 256, 'tool_json': 256}

| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | 40/40 | 230.9 | 231.8 | 231.4 | 0.4% | 524.1 | 514.5 | 519.3 | 1.9% | 0.413/0.416 | 0.418/0.401 | 0.247 | 10240/10240 | 160 |
| structured | 40/40 | 256.3 | 252.3 | 254.3 | 1.6% | 524.2 | 520.3 | 522.2 | 0.8% | 0.468/0.462 | 0.441/0.428 | 0.468 | 10240/10240 | 187 |
| code | 40/40 | 298.4 | 293.8 | 296.1 | 1.6% | 628.1 | 657.4 | 642.8 | 4.6% | 0.676/0.662 | 0.676/0.710 | 0.784 | 10240/10240 | 128 |
| shell_ops | 40/40 | 312.4 | 321.5 | 317.0 | 2.9% | 655.5 | 659.9 | 657.7 | 0.7% | 0.659/0.698 | 0.662/0.669 | 0.770 | 10240/10240 | 117 |
| tool_json | 40/40 | 286.3 | 286.1 | 286.2 | 0.1% | 728.1 | 715.6 | 721.8 | 1.7% | 0.565/0.561 | 0.606/0.577 | 0.831 | 10240/10240 | 106 |

Hygiene: no errors, no foreign tokens, running never above expected, all requests exact FL

### gate: v23post vs control v23 (bar = ctl mean x (1 - max(2%, 1.5 x ctl spread)))

| class | cell | ctl mean | ctl spread | bar | cand mean | Δ vs ctl | verdict |
|---|---|---|---|---|---|---|---|
| prose | c1 | 232.3 | 0.3% | 227.7 | 231.4 | -0.4% | PASS |
| prose | c8 | 532.0 | 2.5% | 511.9 | 519.3 | -2.4% | PASS |
| structured | c1 | 250.3 | 0.8% | 245.3 | 254.3 | +1.6% | PASS |
| structured | c8 | 509.4 | 2.7% | 488.6 | 522.2 | +2.5% | PASS |
| code | c1 | 293.4 | 3.3% | 278.8 | 296.1 | +0.9% | PASS |
| code | c8 | 642.1 | 1.2% | 629.2 | 642.8 | +0.1% | PASS |
| shell_ops | c1 | 316.9 | 5.0% | 292.9 | 317.0 | +0.0% | PASS |
| shell_ops | c8 | 681.4 | 3.3% | 647.5 | 657.7 | -3.5% | PASS |
| tool_json | c1 | 300.0 | 1.6% | 293.0 | 286.2 | -4.6% | FAIL |
| tool_json | c8 | 684.0 | 3.1% | 652.0 | 721.8 | +5.5% | PASS |

GATE: FAIL (9/10 cells)

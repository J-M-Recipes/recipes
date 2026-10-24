# Bring-up cosine gate, same judge (sc3/sc0_gate.py), checks with >= 32 cold rows

| map | gated checks | min cos | below 0.999 | p1 | worst check |
|---|---:|---:|---:|---:|---|
| rowmap-static-v1 (v25, SC0 arm A) | 1,341 | 0.99906 | 0 | 0.99970 | L39 T=128 rows=37 |
| rowmap-static-v1-h285 (285/99, SC3b, rejected) | 1,251 | 0.99885 | 2 | 0.99969 | L39 T=128 rows=45 |
| rowmap-mix-v1 (RonanLabsAI 285/99, SC3) | 1,179 | 0.99911 | 0 | 0.99960 | L22 T=128 rows=44 |
| **rowmap-mixed-ar-v1 (v25m, this run)** | 1,439 | **0.99811** | **1** | 0.99968 | L39 T=128 rows=49, cold share of output norm 0.079 |

Pre-registered bar: min cos >= 0.999 over gated checks, 40/40 layers covered, 0 timeouts. v25m FAILS on min cos
(coverage 40/40, 0 timeouts). The bulk of the distribution matches v25 static (p1 0.99968 vs 0.99970, median 0.99985);
the failure is one check at layer 39, the worst layer on every map tested. Not re-judged with a moved bar.

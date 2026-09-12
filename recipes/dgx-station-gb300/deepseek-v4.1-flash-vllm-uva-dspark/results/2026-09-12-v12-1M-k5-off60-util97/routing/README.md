# Unique routed experts per verify window vs speculation depth (2026-09-12)

From the T5 per-token routing capture (`--enable-return-routed-experts`; 23 requests, 3,545 decode tokens, 40 routed-MoE layers × 384 experts, top_k = 6). A DSpark step at depth k verifies k+1 positions; every expert any of them routes to must be streamed. Windows are k+1 consecutive accepted decode tokens (lower bound — rejected drafts route too). `t5_unique_vs_k.py` → `unique_experts_vs_k.json`.

| k | tokens | unique experts / layer | vs k=0 | Grace-touched share, positional offload (v12) | usage-chosen cold set of same size |
|---|---|---|---|---|---|
| 0 | 1 | 6.0 | 1.0× | 25.3% | 1.0% |
| 1 | 2 | 10.1 | 1.7× | 25.1% | 1.1% |
| 2 | 3 | 13.7 | 2.3× | 24.9% | 1.2% |
| 3 | 4 | 16.8 | 2.8× | 24.8% | 1.3% |
| **5** | 6 | **22.5** | **3.7×** | 24.6% | 1.4% |
| 7 | 8 | 27.4 | 4.6× | 24.6% | 1.5% |

Within ±3% across prose, shell, code, tool-JSON and structured — a router property, not a workload one. This is why k=5 is +100% on shell (91% accept) and −11% on prose (30% accept) against k=0 on this lane: the verify step streams 3.7× the expert bytes, a quarter over the 340 GB/s link, and prose does not accept enough drafts to pay for it.

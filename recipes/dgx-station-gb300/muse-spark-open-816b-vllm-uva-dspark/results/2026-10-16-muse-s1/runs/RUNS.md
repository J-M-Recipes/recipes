# Boot ledger (GB300, 2026-10-15/16)

- `s1-boot1` — eager, autotune on, offload 300; killed by the runner's own 1-h ready timeout mid-autotune (not a model failure)
- `s1-boot2-noat` — eager, autotune off; stopped by operator before ready (operator chose to keep autotune)
- `s1-boot3-at` — eager, autotune on, offload 300, no spec
- `s1-boot4-graphs` — CUDA graphs on, autotune on, offload 300, no spec
- `s1-boot5-dspark` — graphs, autotune on, offload 310, DSpark k=7
- `s1-boot6-graphs-noat` — graphs, autotune OFF, offload 300, no spec
- `s1-boot7-dspark-k3` — graphs, autotune off, offload 310, DSpark k=3
- `s1-boot8-dspark-k2` — graphs, autotune off, offload 310, DSpark k=2
- `s1-boot9-dspark-k7-adaptive` — graphs, autotune off, offload 310, DSpark k=7 + adaptive verification; FAILED at boot: KV 10.24 GiB < 12.5 GiB needed for 131072
- `q-k0` — quality A/B control: graphs, autotune off, offload 310, no spec (+ greedy/BFCL/GSM8K in quality/k0)
- `q-k2` — quality A/B candidate = RECIPE: graphs, autotune off, offload 310, DSpark k=2 (+ quality/k2)

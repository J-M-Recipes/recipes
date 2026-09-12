# Night two ledger (2026-09-11 19:45 → 2026-09-12 02:30 CDT) — how v12 was chosen

One axis per boot, `knee.sh` C1/C8/C16 as verdict, `agent_fixture.sh` as context, same-window v11 control because the reference drifted 82.1 → 79.5 on identical config.

| Boot | Axis | Offloaded GiB | KV GiB | knee C1 / C8 / C16 | Fixture prose / shell | Accept | Read |
|---|---|---|---|---|---|---|---|
| T1a | DSpark off (k=0) | 73.83 | 19.96 | 89.7 / 248.3 / 352.0 | 84.6 / 77.9 | n/a | per-step floor 11.1 ms; vs all-HBM 7.1 ms (catid 2×Station PP2, rev dba1be0a) → fetch tax ≈ 4 ms ≈ 36% |
| T1b | DSpark k=3 | 73.83 | 10.64 | 87.7 / 244.8 / 331.1 | 95.9 / 129.0 | 70.1% | prose +6.8% vs k=5, shell −20% — not adopted |
| T1c | v11 control, same day | 73.83 | 10.05 | 79.5 / 229.4 / 281.7 | 89.8 / 145.7 | 59.6% | −3.2% vs the day-before 82.1: the drift band |
| T2 | util 0.94 → 0.97 only | 73.83 | 17.54 | 79.5 / 228.7 / 277.9 | 90.3 / 145.2 | 59.6% | VOID: util grows KV, does not move experts |
| **T2b (v12)** | **OFFGB 70 → 60 + util 0.97** | **61.17** | 4.89 | **89.2 / 233.7 / 311.4** | 97.7 / 160.3 | 59.6% | **+12.2% C1 same-window, +10% C16, all five classes +10–12%, accept unchanged** |
| T5 | v11 + `--enable-return-routed-experts` | 73.83 | 10.05 | 74.9 / 219.4 / 269.3 | — | — | diagnostic: capture costs ~6%; decode top-73% expert share 0.987; usage-chosen cold 24% takes 2.7% of traffic (leave-one-out) vs 24% positional; agent↔prose rank ρ 0.15 |

Slope: ~0.76 tok/s per GiB of experts moved Grace→HBM (2× the linear-bytes model). Reference drift ~3%/day means the wash band (±1.5%) needs a same-window control boot.

## Confirm pair (2026-09-12 02:52–03:03 CDT)

Requested by James after promotion. v11 REF kneed live as control, then v12 booted (hash `9ac7b387` cached → 10 min) and kneed nine minutes later.

| | C1 | C8 | C16 | fixture prose / shell |
|---|---|---|---|---|
| v11 control 02:53 | 78.1 | 222.5 | 269.6 | — |
| v12 03:02 | **89.6** | **241.5** | **312.4** | 98.0 / 160.8 |
| Δ | +14.7% | +8.5% | +15.9% | |

Two v12 boots (89.2, 89.6) vs three v11 controls across two windows (79.5, 79.5, 78.1). `starve.py` on v12: shorts 0.81–1.09 s during a 480K prefill — the fairness flag is unaffected by the offload change. Cold prefill on v12: 6.5K 0.48 s, 52K 2.68 s, 207K 11.37 s (18.2k tok/s) — same as v11. Prefill is not offload-bound.

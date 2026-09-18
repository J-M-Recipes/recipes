# 2026-09-17 — E4 / E4b / E5: placement adaptation (negative results; nothing promoted)

Run family for the `limits` line "Placement adaptation CLOSED 2026-09-17". Same lane, same image (`vllm/vllm-openai:deepseekv41-flash-0909`), same launch flags as v15; every boot hash-hit FlashInfer autotune `9ac7b387` (Loaded 231 / 0 new). Control in every window is the live v15 reference (`docker start` of the stopped `-BOUND-REF`). Verdict instrument is `knee.sh` (C1/C8/C16, two runs per window); `agent_fixture_o.sh` with fixed `CAT_ORDER prose structured code shell_ops tool_json`; `e2c_heldout.py` (5 math + 5 JP/DE prompts, streaming first→last token); greedy parity (T=0, seed 42, 256 tokens). Workers: Grok 4.6 via xAI, from Milo's written briefs; every number below was read back from the JSON in this directory.

| Sub-run | Question | Result | Bar | Verdict |
|---|---|---|---:|---|
| `e4/` | On-line adaptive: in-graph `index_copy_` ring counter, swaps frozen | C1 138.3 vs v15 153.0 (**−9.63%**), C8 −8.1%, C16 −6.5%; accept 59.6% both | C1 ≥ −1% | FAIL (counter tax) |
| `e4b/` | Same, D1 fixed-slice snapshot counter (`snap[layer,:T].copy_`) | C1 150.3 vs 152.9 (**−1.68%**), C8 −1.26%, C16 −0.94%; D0 zero-copy not viable (`topk_indices_buffer` is Indexer scratch) | C1 ≥ −1% | FAIL (close) |
| `e5/` | Off-line adaptive: re-profile on mixed traffic, `rowmap-mixed-v2` (295/89), reboot | Held-out JP/DE 88.9→145.6 (**+64%**), math +10%; knee C1 153.1→137.8 (**−10%**), C8 −16%, C16 −17%; tool_json −7%; parity 18/18; accept 59.58% all windows | JP/DE ≥ +1.5% AND C1 ≥ −1.5% | FAIL on agent bars |

What the family settles: the hot/cold row-swap path is bit-exact (`e4/dryrun.json`: five ordered swaps 100% identical, a deliberately mis-ordered flip diverges), but observing the router from inside the CUDA graph costs more than 1% by every mechanism tried; and the 89-cold-slot budget is zero-sum across domains, so the row map is a per-workload artifact — build one per traffic class from that lane's real traffic and swap by hash-hit reboot, never blend. v15 (agent-profiled static map) remains the reference.

## Files

- `e4/` — `dryrun.json` (swap bit-identity), `overhead-e4f.json`, knee/agent/hostwatch pairs (`e4f-*` candidate, `v15e4-*` control), `README.md`
- `e4b/` — `d0.json` (why `topk_indices_buffer` cannot be used), `overhead-e4b.json`, knee/agentfix/hostwatch pairs, `swaps.jsonl`, `README.md`
- `e5/` — profile: `counts-{zero,mid,final}-*.npz`, `dump-*.json`, `profile-*.json|jsonl|log`, `profile-share.json`, `rowmap-churn.json`, `rowmap-mixed-v2.json`; windows: `knee-*`, `agentfix-*`, `heldout-*`, `parity-*`, `hostwatch-*`, `window-*.log`, `summary.json`, `README.md`
- `hook/` — `rowmap-mixed-v2.json` (E5 map; **not a serving map**), `e4_adaptive.py`, `e4_dryrun.py` (adaptive swap logic and its off-lane bit-identity test). The serving hook and `rowmap-static-v1.json` are in `../2026-09-17-e2b-pin-hot-experts-v15/`.

Box paths and LAN addresses are scrubbed (`<box-home>`, `<station-ip>`). Ledger of record: `pin-hot-experts/results/ledger.md`; closing note `pin-hot-experts/CLOSED-2026-09-17.md`; blog https://al-engr.com/vllm-pin-hot-experts.html#closed0917.

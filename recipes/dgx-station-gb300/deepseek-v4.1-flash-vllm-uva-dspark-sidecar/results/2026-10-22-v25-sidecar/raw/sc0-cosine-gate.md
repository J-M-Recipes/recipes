# SC0 result — arm A (ours, rowmap-static-v1, 295 hot / 89 cold) — PASS — 2026-10-21 20:16 CDT

Container `dsv41-vllm-SC0-armA-ours-EXP` + `-rtx`, image nightly-af7f9488 (sha256:3af6a45b…), 1017 weights,
replay overlay #58132 on, MEGA_PEER=2, MEGA_PEER_CHECK=100000, MEGA_COLD_TRT=1, b12x 2cc7f66a, el8 87af662 (+ path-only edit
local/peer_server2_local.py). Bound 20:12 CDT (~11 min boot, warm caches). Receipts: receipts/dsv41-vllm-SC0-armA-ours-EXP.log,
receipts/armA-peer_server2.log, receipts/armA-peer_stats.json, sc0-armA-send.log. Driver/judge: sc0_gate.py.

| Item | Value |
|---|---|
| Real prompts sent (post-startup) | 39 (16 docs ~2.7–4k tok, 18 code/scripts, 5 zh/ja/fr/ru/math) |
| Post-startup checks | 1,720; 1,341 with ≥32 cold rows |
| Layers with ≥1 check ≥32 cold rows | 40/40 |
| **Min cos (checks ≥32 rows) — gate ≥0.999** | **0.99906** (layer 39, T=128, 37 rows) — PASS |
| Max rel_diff (≥32 rows) | 0.0434 (same check); 11 checks in 0.025–0.043, all layers 22–39 at T=128 |
| Layers 0–20 (full prefill, 149–2,150 rows) | min cos 0.9997, max rel 0.0247 |
| Min cos, all rows (informational) | 0.99747 — layer 39, 5 rows, cold share of output norm 0.000 |
| Sidecar timeouts (peer log + hook) | 0 |
| RTX memory.used | 66,999 MiB (plan expected ≈65 GiB) |
| GB300 KV cache | 2,536,059 tokens (SC5 bar ≥2.50M: just clears) — vs 6.09M on arm B 285/99 |

Notes, honest:
- Layers 21–39 only see T=128 (bounded replay overlay), so their checks are small-row; the ≥32-row rule is Ronan's
  re-registered gate. The all-rows minimum is below 0.999 only on layer 39 small-row checks with ~0 cold share.
- The check path never exercises the fused T≤64 decode graphs (W1); decode correctness is covered by SC2 quality gates.
- Arm B (el8 285/99 positive control) was never gated on real prompts (see TAKEOVER.md). Not re-run: arm A passed, so the
  control is not needed to diagnose a failure. Can be re-run for the record if James wants it.
- Answers on all 39 prompts were coherent (first 70–90 chars logged).

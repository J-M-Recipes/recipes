# SC1 result — sidecar (295/89, production mode) vs v23 and M3G-ours — bracketed — 2026-10-21 20:16–21:21 CDT

Runner: box ~/dsv41/m3/cardSC1_runner.sh (copy here), results box ~/dsv41/m3/results-cardSC1/. Arms CTL1 v23 → CAND1 sidecar →
M3G (el8 hook Grace-only, same map/image) → CAND2 (fresh boot) → CTL2 v23 (left serving). Image af7f9488 for CAND/M3G, 0909 for v23.
Numbers below computed from the raw JSON (the runner's RT summary line printed None: parser read top level, data is under `summary`).

## Speed

| Cell | CTL1 | CAND1 | M3G | CAND2 | CTL2 | CAND mean vs CTL mean |
|---|---|---|---|---|---|---|
| Real-text C1 decode_agg (mean of 3, 8.2K in / 1,024 forced) | 187.1 | 276.8 | 176.7 | 282.2 | 187.9 | **279.5 vs 187.5 = +49%** |
| Real-text TTFT (8.2K) | 0.66 s | 0.19 | 0.28 | 0.19 | 0.66 | **3.5× lower** |
| Real-text C4 | 323.8 | 689.1 | 351.7 | 697.4 | 319.5 | +116% |
| Real-text C16 | 523.8 | 1,581.0 | 677.3 | 1,572.8 | 525.2 | **3.0×** (vs M3G 2.3×) |
| Real-text C32 | 926.1 | 2,401.3 | 949.3 | 2,399.3 | 947.7 | 2.56× |
| Knee C1 (agent text) | 192.0 | 220.7 | 173.4 | 224.7 | 192.2 | **+16%** (bar ≥198 PASS) |
| Knee C8 | 612.9 | 932.3 | 688.6 | 925.3 | 621.6 | +51% |
| Knee C16 | 941.5 | 1,420.4 | 955.9 | 1,394.5 | 941.0 | **+50%** (bar ≥974 PASS) |
| Agent replay r2 (4 workers, transcripts) | 423.4 | 519.0 | 409.3 | 498.5 | 414.2 | +21% |
| Gauntlet | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | — |

Pre-registered bars (REPORT Phase 2 #5): knee C1 ≥ 198 PASS (220.7/224.7); C16 ≥ 974 PASS; Ronan-harness C16 ≥ M3G +3% PASS
(1,577 vs 677); sidecar timeouts 0 PASS (peer=0 hook=0 both CAND arms); early-stop (C1 ≥ +10% knee) passed.

## Facts / flags

- KV cache: CAND 2,537,081 tokens (2.42× at 1M), v23 2,504,891, M3G 2,489,563.
- **Acceptance is lower on the af7f9488/el8 stack, not because of the sidecar:** agent-fixture weighted accept CAND 0.572/0.537,
  M3G 0.529, v23 0.583/0.583; replay acc/step 2.73–2.79 vs 2.89–3.07. M3G (same image+hook, no sidecar) shows the same drop, so it
  is an image/hook/overlay property. Trigger for SC2's TF-flip and per-class checks.
- **Power: GB300 `sw_power_cap` active in 6.0% (CAND1) / 7.96% (CAND2) of 1 s samples**, GPU p99 ≈ 1,000 W (v23/M3G: 0.00%,
  p99 ≈ 830/767 W). RTX p99 203 W (≤300 W). SC6's bar is < 1% of samples → **the 1,250 W module cap is binding during sidecar load
  today.** The speed numbers above were measured under that throttling (they are what the lane delivers), but SC6 must break it
  down by phase (prefill bursts vs C32 decode) before the recipe claims "no power action needed".
- Boot: sidecar prepared 40 layers in 46 s, 560 graphs, 68.4 GB on the RTX (66,213 MiB used); server bind 450 s warm (527 s total).

# T4 — E4b D1 adaptive counter, unfrozen, gated at C8/C16 — FAIL (operational bar), held-out NOT MEASURED

Runner: Miloh's overnight runner. `COUNTER=d1`, `PIN_ADAPTIVE_FREEZE=0`, defaults for half-life (500) / drain (64) / swaps-per-drain, GO file after bind, on v18 flags and the 0909 image. Hash hit `62426808…` (Loaded 189). Rehome 206.61 / 62.33, `ADAPT counter=d1 TMAX=96 K=6 layers=40 device_bytes=92320`. Graphs 1.30 + 1.09 GiB, KV 2,494,774 (v18 2,502,950; −0.3%). Sequence 06:24–06:59 CDT: T4 → v18ctl4 → T4b (restart of the same container) → v18 restore. Receipts: `receipts/boot-T4.txt`, `smoke-T4.txt`, `window-T4*.log`, `swaps-T4*.jsonl`, `heldout-*.json` (tracebacks).

## Two-window means (T4 + T4b) vs controls (v18ctl3 + v18ctl4)

| | C1 | C8 | C16 (mean) | C16 r2 paired | replay n=4 | prose | tool_json |
|---|---|---|---|---|---|---|---|
| v18ctl3/4 | 171.8 | 658.8 | 852.2 | 947 / 954 | 389 | 203 | 240 |
| T4/T4b | 166.7 (**−3.0%**) | 640.0 (**−2.9%**) | 934.6 (+9.7%) | 944 / 939 (**−1.0%**) | 431 (+10.8%) | 192 (−5.2%) | 234 (−2.6%) |

Bars: C8 ≥ −1.5% (measured tax was −1.26), C16 ≥ −1.0%, JP/DE held-out ≥ +20%, math ≥ −1.5%, parity 18/18, no fixture class < −3%.

- **C8 −2.9% fails the bar by 2×.** C1 −3.0%. The C16 "+9.7%" is the r1/r2 warm-cache artifact (controls' r1 was 751/756 vs candidates' 936/920 — the candidate windows ran after a warmer box); paired r2 is −1.0%, at the bar.
- **Swaps: 11 in T4, 12 in T4b, all applied in the first drain after GO; zero during the held-out prompts** (`swaps-T4*.jsonl`: `swaps after warm 1→11`, `after heldout p1/p2` unchanged). The EWMA never moved a row after warm-up on this traffic. So the ~3% is the counter's tax with no placement change to show for it — the same finding as E4/E4b on 2026-09-17, now with the counter unfrozen.
- **Held-out JP/DE + math never ran.** The runner called `e2c_heldout.py` with no `<tag>` argv → `IndexError: list index out of range` × 5 (T4 p1/p2, T4b p1/p2, v18ctl4). The one number T4 was supposed to earn is missing. Re-running it would measure a placement that never changed (12 swaps, none off-profile), so it was not re-run in daytime; the lever is closed on this hook regardless.
- Parity vs v18 not captured overnight (runner did not call it); T4 is the same image and rowmap as v18 so bit-exactness is not in question, only the swaps — which did not happen during the measured windows.
- Fixture prose −5.2% (< −3% bar) — consistent with the C1 tax.

VERDICT T4: FAIL — C1 −3.0% · C8 −2.9% (bar −1.5%) · C16 paired −1.0% · replay +10.8% (single-window instrument, ±20%) · 12 swaps total, 0 after warm · held-out unmeasured (runner argv bug) · hash hit · stopped-and-kept `dsv41-vllm-T4-adaptive-d1-unfrozen-C8gate-EXP`. Adaptive-on-this-hook closed: the D1 counter costs ~3% and the EWMA does not move rows on lane traffic.

# Card PF verdict  (base=S cand=PF root=~/dsv41/sidecar-resid/results-PF-w1)

## per-cell server prefill (median ms, n reps)
| target | n | base | cand | delta (base-cand) | 95% boot CI | client TTFT delta |
|---|---|---|---|---|---|---|
| 40 | 11/11 | 44.6 | 44.9 | -0.30 | [-3.8, +3.9] | +0.0 |
| 92 | 11/11 | 47.9 | 47.4 | +0.50 | [-1.1, +1.4] | +0.3 |
| 120 | 11/11 | 49.4 | 49.5 | -0.10 | [-56.4, +76.7] | +0.3 |
| 264 | 11/11 | 124.0 | 105.8 | +18.20 | [+16.8, +19.6] | +18.2 |
| 392 | 11/11 | 123.4 | 105.5 | +17.90 | [+16.0, +21.1] | +16.1 |
| 648 | 11/11 | 125.0 | 105.9 | +19.10 | [+15.7, +22.5] | +19.6 |
| 1144 | 11/11 | 122.4 | 104.3 | +18.10 | [+13.2, +22.6] | +16.5 |
| 1416 | 11/11 | 128.2 | 107.7 | +20.50 | [+6.3, +29.6] | +18.7 |

pooled eager delta (mean of the 4 per-cell medians' deltas) = **+18.33 ms**   95% CI [+16.68, +19.90]
informational 1416-row cell: +20.50 ms (expect 0..4: GPU-bound there)
PREDICTION pooled: lo 9.0 / mid 12.6 / hi 17.5 ms

## gates
- G1: PASS -- captured-cell server prefill cand-base 40:+0.3 92:-0.5 120:+0.1 (bar +-3.0; client TTFT is in the table, informational)
- G2: FAIL -- kv_bytes==; applied_graph_estimate_bytes==; post_capture_free_bytes==; profiled_peak_activation_bytes==; kv_tokens 11660754/11659893 capture_lines 45/45; quant_producer CAPTURE inventory lines identical (2/2)
- G3: PASS -- pooled +18.33 ms (bar 5.0), cells >= 3.0 ms: 4/4 (bar 3); matches lo..hi prediction
- G4: PASS -- 8 base rows vs 8 cand rows, identical=True; control vs its own 2nd pass identical=True
- G5: PASS -- decode tok/s cand vs base: c1 -0.2% c32 +0.9% c8 -1.0% (bar c1,c8 +-3 %; c32 reported only)
- G6: PASS -- cand ACTIVE lines=1, DISABLED=False, NOT installed=False; base PF1 lines=0

**VERDICT: PF-HOLD (G3 passes, another gate fails: do not ship)**

# CARD X6 — prefill yield on Pg63, third boot (tiebreak) — PRE-REGISTERED 2026-11-21 (before boot), James "2" at ~11:20

Arm Y3 = exactly X5's Y (ov/yieldcad + VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192 VLLM_PREFILL_YIELD=short
VLLM_PREFILL_YIELD_LIMIT_S=2), same battery + post + C32x5 + det_check + P1x5 + P2x2. Then restore Pg63.
Control = X5's K1+K2 (same day, same harness): P1 median 0.327 s; P2 median 1.07 s; long-prompt median 10.79 s; C1 mean 549.9;
C16 mean 2,393.7; C32x5 median 3,956.5; cold prefill means; decode during 413K mean 33.05; qeval 71.
Bars: identical to CARD-X5 Y (P1 ≤ 0.261 s; P2 ≤ 1.176 s; long ≤ +3 %; decode during 413K ≥ −5 %; C1 ≥ −2 %; C16 ≥ −3 %; C32x5 ≥ −3 %;
stalls ≤ 2; cold prefill ≥ −3 %; qeval 70–72, needle 3/3, DSML, JSON ≥ 63/64, greedy 40/40).
Promotion: Y3 passes all → Pg64 = Pg63 + yield (Y1 and Y3 = two passing boots; Y2 reported). Y3 fails any → yield stays built-but-off.

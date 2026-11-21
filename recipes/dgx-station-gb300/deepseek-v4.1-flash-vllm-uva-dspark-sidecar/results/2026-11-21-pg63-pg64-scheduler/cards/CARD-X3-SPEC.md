# CARD X3 — cadence gated on a long prefill in flight — PRE-REGISTERED 2026-11-21 06:47 CST (before any X3 boot)

## Why (X2 result, D1/D2 vs controls A1/A2; A3 pending at time of writing)
Long-only deferral kept every gain (decode during 413K 34.6 / 32.2 tok/s vs 12.7; during 1M 28.6 / 31.8 vs 10.65; long prompt time
unchanged; true TTFT behind 413K 0.371 / 0.391 vs 0.51 s; first answer token 0.73 / 0.76 vs 1.37 s; fidelity identical) but C32 still read
−2.5 % (D1) and −5.1 % (D2). Per-request C32 data: with NO long prompt anywhere, one request of a 32-burst gets its first token at the normal
0.18 s and the other 31 wait an extra 0.12 or ~0.47 s together (also in B and C; never in A1, A2, GR). So the cadence path perturbs
admission even when nothing is deferrable; root cause not yet isolated. X3 removes the question: the cadence tick only runs while a
running request still has > MIN uncomputed prompt tokens. With no long prompt in flight, `schedule()` is the stock code path.

## Change under test
`ov/cadence3` = cadence2 + gate (tick advances and throttle is set only when `any(r.is_prefill_chunk and r.num_tokens - r.num_computed_tokens
> MIN for r in running)`; else tick reset to 0, throttle untouched).
| arm | env |
|---|---|
| E1, E2 (two boots) | `VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192` |
| A4 (control after restore) | Pg62 as served |
Control reference = mean of A1, A2, A3, A4.

## Bars: identical to CARD-X2 (each of E1 and E2 must pass all), with one addition
7. C32 per-request: the 31-request extra wait pattern must be absent (C32 mean TTFT per category within +0.05 s of control mean).

## Promotion: as CARD-X2 (both boots pass → Pg63, Pg62 kept as scripted rollback).

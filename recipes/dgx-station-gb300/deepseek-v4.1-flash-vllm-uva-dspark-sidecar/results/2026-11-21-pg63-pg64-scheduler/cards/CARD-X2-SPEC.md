# CARD X2 — long-only prefill cadence on Pg62 — PRE-REGISTERED 2026-11-21 06:02 CST (before any X2 boot)

## Why (window 1 result, X1)
Cadence N=3 (arm C) lifted reply speed during a 413K prompt 12.7 → 34.6 tok/s (2.7×) and during a 1M prompt 10.65 → 30.2 (2.8×) with the
long prompt's own time unchanged (413K 10.96 s vs 10.91; 1M 37.37 vs 37.49), first answer token for a short chat behind 413K 1.37 → 0.665 s,
fidelity identical (qeval 71, needle 3/3, greedy 40/40, JSON 64/64, DSML pass). It FAILED the C32 bar: −6.0 % (3,659.5 vs 3,891.1 control mean),
concentrated in the `math` category (−26 % on both B and C), because X1 deferred *every* prefill, including the short prompts of a 32-user burst.
B (N=2) passed 413K (1.83×) but missed the 1M bar (1.35× < 1.4×) and sat at C32 −3.0 %.

## Change under test
`ov/cadence2` = X1 cadence + `VLLM_PREFILL_CADENCE_MIN_REMAINING`: only prefill work whose request still has > MIN uncomputed prompt tokens is
deferred (running chunks and new admissions). Short prompts and cached-prefix suffixes are never delayed. MIN=0 reproduces X1.

| arm | env |
|---|---|
| D1, D2 (two separate boots) | `VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192` |
| A3 (control after restore) | Pg62 as served |
Control reference = mean of A1, A2, A3 (A1/A2 from X1, same day, same harness).

## Bars (each of D1 and D2 must pass all)
1. decode DURING 413K prefill ≥ 1.5× control mean; 413K time while shared ≤ +6 %.
2. decode DURING 1M prefill ≥ 1.4× control mean; 1M time ≤ +5 %.
3. cold prefill 32K/131K/300K/1M ≥ −3 % vs control mean.
4. C1 ≥ −2 %, C16 ≥ −3 %, **C32 ≥ −3 %** vs control mean; crash-free.
5. qeval within control spread ±1; needle 3/3; DSML ALL_PASS; JSON 64/64; greedy same-boot 40/40.
6. true TTFT short-behind-413K ≤ +10 %.
Engagement: `xfer prefill cadence N=3: deferred` lines in the engine log (else INVALID).

## Promotion
If D1 and D2 both pass: promote as **Pg63** = pg62-work + `overlay-vllm/v1/core/sched/scheduler.py` + the two env vars, Pg62 kept as
rollback with a scripted rollback (James 11/20: "if it wins, it lives as a full-time resident"). Otherwise Pg62 stays, report.

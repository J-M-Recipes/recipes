# CARD X5 — on Pg63: (D) sidecar deterministic combine, (Y) prefill yield v2, phase-locked — PRE-REGISTERED 2026-11-21 10:00 CST (before any X5 run)

Base = production Pg63 (`garner/pg63-work`). Arms built by `xfer_arm2.sh` (copy of pg63-work; only the listed change differs; verified in-container).
Order: K1 (live Pg63, before) → D1 → Y1 → D2 → Y2 → restore → K2 (Pg63, after). Control = mean/median of K1, K2.

| arm | change |
|---|---|
| D1, D2 | sidecar env `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` (b12x combine: fixed-order ROUTE_BUFFER_TOPK_SUM instead of bf16 atomics for >64-row calls). Engine container unchanged. |
| Y1, Y2 | scheduler overlay `ov/yieldcad` (= cadence3 + 11/16 W1 port of Owen's 0545 yield, merged with no rejects) + env `VLLM_PREFILL_YIELD=short VLLM_PREFILL_YIELD_LIMIT_S=2` (cadence env kept) |

Every arm: battery + post suite (as X4) + C32×5 + det_check (32 concurrent greedy+logprob requests, run twice; 128-row MoE calls)
+ P1: phase-locked short chat (thinking off) sent at 0.8 × that rep's solo cold-413K time while a cold 413K runs, ×5
+ P2: same with an 8,192-token prompt as the "short" (must not be helped: not single-chunk under LPT 6144), ×2.

## Bars — D (each of D1, D2)
- Determinism (the point): det_check `identical_text_and_logprobs` = 32/32 AND `logprob_diffs` = 0. Control floor (K1/K2) reported.
- Cost: C32×5 median ≥ −3 % vs K median of 10; C1 ≥ −2 %; C16 ≥ −3 %; cold prefill 32K–1M ≥ −3 %; decode during 413K ≥ −5 %.
- Fidelity: qeval within ±1 of control, needle 3/3, DSML, JSON 64/64, greedy 40/40.
- If the determinism bar fails while K shows diffs too: D is not doing what we think → NOT promoted, report. If K already reads 32/32 with 0 diffs: the lever has nothing to fix at this cell → NOT promoted (no measured benefit), report.

## Bars — Y (each of Y1, Y2)
- Primary: P1 TTFT median ≤ 0.8 × control's P1 median (K1+K2 pooled reps).
- Mechanism: P2 (8K) TTFT median ≤ 1.10 × control; P1 long-prompt completion median ≤ +3 % vs control.
- No regression (as X4): decode during 413K ≥ −5 % vs control; C1 ≥ −2 %; C16 ≥ −3 %; C32×5 median ≥ −3 %; stall count ≤ control max + 2;
  cold prefill ≥ −3 %; fidelity as above.

## Promotion
D both pass → Pg64 = Pg63 + sidecar flag. Y both pass → yield added. If both pass, promote the combination only after one confirmation
boot of the combination (separate short card). Otherwise Pg63 stays. (James 11/20 standing rule; 11/21 "Go all".)

# CARD X1 — prefill cadence (W1) + prefill upstream backports #59131/#63540 (W4) on Pg62 — PRE-REGISTERED 2026-11-21 04:54 CST

Written before any arm boot. Window 1 = first boot of each arm; window 2 = second boots of anything that passes here.

## Arms (each = pg62-work copy + overlay files only; serve args identical; verified by sha256 in-container)
| arm | change | source |
|---|---|---|
| A1 / A2 | Pg62 as served (A1 live before the window, A2 after restore) | control |
| B | `VLLM_PREFILL_CADENCE=2` + scheduler overlay (`ov/cadence`, 22 lines: on non-cadence steps set stock `throttle_prefills` → stock `defer_prefills` path; stock guard: only when a decode request is running) | W1 §6.2 |
| C | `VLLM_PREFILL_CADENCE=3` (skipped if B's decode-during-413K < 1.5× A1) | W1 §6.2 |
| GR | #59131 (gather compressed KV once per KV-source group; 3 files + hit/miss counter) + #63540 (fused_inv_rope_fp8_quant 4 heads/program; 1 file) | W4 §3.3, §6 |

## Cells (battery.py phases smoke,c1,c16,prefill,needle,mixed_short,mixed_warm,agent_turn,qeval; post suite: mixed_trace short ×4, decode during 413K ×3, decode during 1M ×1, C32 stress, gates, greedy ×2, JSON burst, DSML live)

## Bars — cadence arms (B, C)
1. decode during 413K prefill (post `trace-d`, median): ≥ 1.5× A1/A2 mean; 413K time while shared ≤ +6 %.
2. decode during 1M prefill (single trace): ≥ 1.4× control; 1M time ≤ +5 % (single sample → report, confirm in window 2).
3. solo cold prefill 32K/131K/300K/1M within −3 % of control (deferral must not fire with no decode resident).
4. C1 ≥ −2 %, C16 ≥ −3 %, C32 ≥ −3 %, crash-free.
5. fidelity: qeval within control spread ±1; needle 3/3; DSML live ALL_PASS; JSON burst 64/64; greedy same-boot 40/40.
6. true TTFT short-behind-413K: no worse than +10 % vs control.
Engagement: engine log must show `xfer prefill cadence N=… deferred` lines; absent ⇒ INVALID, not null.

## Bars — prefill backport arm (GR)
1. cold prefill ≥ +5 % at 131K OR ≥ +8 % at 300K or 1M vs control mean (A1/A2), outside the A1–A2 spread.
2. no regression: C1 ≥ −2 %, C16 ≥ −3 %, 32K prefill ≥ −3 %; mixed rows not worse than +20 %.
3. fidelity as above; greedy agreement no worse than control-vs-control.
Engagement: `xfer #59131 retained span` log line with hit > 0; absent or hit = 0 ⇒ INVALID.

## Promotion rule
Two boots (window 1 + window 2) must each pass, and agree within the controls' spread. Then promote as Pg63 with Pg62 kept as rollback
(James 11/20: "if it wins, it lives as a full-time resident"). Anything in the James-decides band (gain with one bar missed by < half its margin) is reported, not promoted.

## Traps (from W1/W4)
`prefill_capacity_bound` disables throttling when blocked waiters exist; ITL mean vs p99 (cadence makes ITL bimodal); 1M is a different regime;
prefix-cache contamination in prefill cells (battery uses nonces); p3 watchdog `trip: null` start/end; workspace OOM at 1M with retained span.

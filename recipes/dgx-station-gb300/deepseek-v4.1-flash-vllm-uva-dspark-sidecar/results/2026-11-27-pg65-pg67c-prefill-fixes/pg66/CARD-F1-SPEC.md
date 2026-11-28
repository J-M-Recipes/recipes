# CARD F1 — Garner's numerics-class decode fusions on Pg65 (Flare-transfer round)

Pre-registered 2026-11-24 19:48 CST, before launch (runner sha1 05c81311, decode_lp sha1 2edf35c2). James 11/24: "yes; you have the gb300; let's test and see if we can
improve our recipe."

## Why this card (from the Flare read)
Flare's 11/23 DSV4.1 wins (fused Q/KV norm+RoPE, Q-RoPE in the MXFP8 GEMV, inverse-RoPE in WO-A, split-lagged mHC,
pre-quantized MoE input) were checked against our stack's kernel inventory (GTA2 nsys, Garner stack on GB300,
`~/dsv41/flarex/gta2-inventory.txt`): **all already present in Pg65** via Garner's components — MegaAttn/W1
(`fused_norm_rope_attn_rope_cast`, fp8 out), `preattn` (w3 QkvStage), `boundary` (w4 seam on main stream + w4 mix on
the mHC side stream = split mHC; mix exposed 0.021 ms/step) and the w4 router (stages MegaMoE inputs incl. MXFP8 =
prequantized MoE). Flare's remaining fusion class that we do NOT run is the WO-A + output-quant seam: our step still runs
DeepGEMM `fp8_einsum` wo_a (43/step, 0.507 ms summed / 0.396 ms alone) then a standalone 8192-wide MXFP8 quantize
(43/step, 0.086 / 0.061 ms) before wo_b. Garner ships exactly that fusion as `w8oproj` (wo_a + MXFP8 quantize in one
kernel, wo_a weights streamed in the prologue while attention runs; T 1..8 only). We dropped it on 10/29 as
"changes numerics": window layers bitwise, sparse + drafter layers a deterministic 2-part split-K (summation-order
class, no format change). Never measured on our lane in isolation. `lowb_aux` (cuBLAS split-K bf16 side GEMMs ->
deterministic LLBf16 at small T, summation-order class) rides along as arm D.

## Arms (one window)
A1 live Pg65 (isolated, no boot) -> C = Pg65 + `w8oproj` -> D = Pg65 + `w8oproj` + `lowb_aux` -> restore Pg65 -> A2.
Arms are copies of `pg65-work` differing ONLY in `DS41_FASTPATH_COMPONENTS_OVERRIDE` (and the overlay-dir path line);
runner asserts tree/overlay diff 0 and exactly 4 changed policy.env lines.

## Cells per arm (identical)
decode_lp x2 (40 prompts, greedy, top-5 logprobs, thinking off, 256 tok), perclass C1 T=1.0 nat + seeds 42/43/44,
perclass C1 T=0, gauntlet (12), Garner bench_full (fixed, gsm8k_c1, greedy_c1, C8 x3, C32 x3), C32 x2 more.

## Gates (frozen now)
1. **Engagement:** boot log `DS41_FASTPATH ACTIVE w8oproj` (C, D) and `ACTIVE lowb_aux` (D); w8oproj `kernels loaded
   for T in 1..8`; at least one engaged-T line for T=6 or 8 in serve.log. Else arm INVALID (not FAIL).
2. **Survival:** server up after all cells (5 C32 windows); no `illegal memory access` / EngineDead.
3. **Speed (primary, C1):** per-class pooled step ms = 1000 x (1 + acc/step) / tok/s, mean of the 3 forced seeds, 5
   classes averaged. Win bar vs control MEAN (A1, A2): <= -0.10 ms/step AND beyond 2x the A1-A2 half-spread.
   Report vs min control too. Also report T=0 cell and nat cell.
4. **No regression:** C8 / C32 aggregate (bench_full medians) >= control mean - max(3 %, control spread).
5. **Quality (numerics-changing arm):**
   - decode_lp arm-vs-A1: mean |dlp| and p99 |dlp| <= 1.5x the worse of (A1-vs-A2, A1 run1-vs-run2 same boot);
     top1_agree on common prefix >= control pair's - 0.002; first-divergence median >= 0.7x control pair's.
   - gauntlet: passes >= min(A1, A2) passes.
   - gsm8k_c1 (bench_full): within the control spread + 2 pts.
6. **Promotion:** only if 1-5 all pass on C or D. Promote the cheaper-to-explain arm when both pass and D's extra gain is
   inside noise. Promotion = new tree `pg66-work` via cloned promote script + rollback script; needs a second boot's
   confirmation cells (C1 per-class 3 seeds + C32x3) as part of promotion.

## Known traps
- prompt_logprobs (TF) is prefill-only (T >> 8): blind to w8oproj/lowb_aux. Hence decode_lp.
- Single post-boot C32 is noisy (first C32 after a fresh boot reads low) — bench_full warms 20 s per window; judge medians.
- Same-boot greedy repeat (pre-card live: 32 % positions bit-equal, mean |dlp| 0.012, first-div median 49) — the lane is
  not batch- or replay-deterministic; gate on spread, not equality.
- End state: Pg65 restored on every exit path (trap). MAINTENANCE written at start, archived at END.

## F1 result (21:15 CST) and F1b pre-registration (written before F1b launch)
F1: both arms pass every gate (`~/dsv41/flarex/results-F1/ANALYSIS.txt`). C1 step vs control mean 7.841 ms (A1 7.855 /
A2 7.827): C −0.120, D −0.149. C32/C8 flat, gauntlet 12/12 all arms, decode_lp inside control spread on every metric.
D−C = −0.029 ms ≈ 2× control half-spread (0.028): at the noise edge, so the rule "promote the cheaper arm when D's extra
gain is inside noise" cannot be decided from one boot. GSM8K accuracy was NOT measured (bench_full's `gsm8k_c1` is a
throughput cell; `quality` stage skipped) — gate 5's gsm8k sub-gate was mis-specified and is void, stated here.

F1b (second boot each, order A3 live → C2 → D2 → restore Pg65 → A4; runner `f1b_runner.sh`), cells: decode_lp ×1,
per-class C1 T=1 seeds 42/43/44, T=0, fixed greedy ×3 + C32 ×3 windows (bench_full) + 2×C32 more.
- Confirm (per arm): 3-seed C1 step Δ vs mean(A3, A4) ≤ −0.08 ms; C32 mean ≥ control mean − max(3 %, control spread);
  decode_lp vs A3 inside F1's control envelope rules; gauntlet not repeated.
- D over C: promote D only if D beats C in BOTH boots AND by ≥ 0.05 ms pooled over the two boots; else promote C.
- Promotion = `pg66-work` (copy of pg65-work + the winning component list), promote + rollback scripts cloned from
  pg64→pg65, smoke + watchdog + C1 spot check after promote.

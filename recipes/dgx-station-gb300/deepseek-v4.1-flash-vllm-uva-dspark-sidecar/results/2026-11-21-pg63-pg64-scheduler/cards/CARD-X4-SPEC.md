# CARD X4 — long-prefill cadence (cadence3), better 32-user instrument — PRE-REGISTERED 2026-11-21 08:05 CST (before any X4 run)

## Why a new card (X3b diagnostic, 07:37–08:04, no promotion possible from it)
Fresh boots of E3 (cadence on), F (cadence code, switched off) and Pg62 (A5), each C32 ×3 cold → 1M trace → C32 ×2:
- The first C32 after ANY fresh boot is bad on all three, including Pg62: means 2,670 / 2,859 / 2,696, with 0.65–1.6 s stalls (warm-up).
- 0.3–0.74 s single-category stalls recur at random on all three arms in later reps (Pg62 A5: codin 0.31 s r2, forma 0.74 s r3).
- After warm-up, C32 means: E3 4,004 / 3,981; F 3,932 / 3,966; Pg62 3,943 / 4,007. No math stall in any of the six warm reps.
- F's decode during 1M = 8.9 tok/s (stock behaviour); E3's = 28.2 (cadence).
So the single-rep C32 inside the post suite (the instrument behind the X1–X3 C32 failures) has a stall noise floor on Pg62 itself
of the same size as the failures. The pre-registered X3 verdict stands (not promoted). X4 keeps every bar and replaces only the
32-user instrument with a repeated, warmed measurement on all arms, controls included.

## Arms (same window, control bracketing)
A6 (live Pg62, before) → G1 (cadence3 boot 1) → G2 (cadence3 boot 2) → restore → A7 (Pg62, after).
Env for G: `VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192`, overlay `ov/cadence3` (unchanged since X3).
Each arm: full battery + post suite exactly as X1–X3, then **C32 ×5** (v41bench levels 32, back to back).

## Bars (each of G1 and G2 must pass all)
1–6: identical to CARD-X2 (decode during 413K ≥ 1.5× control mean; 413K time ≤ +6 %; decode during 1M ≥ 1.4×; 1M time ≤ +5 %;
cold prefill ≥ −3 %; C1 ≥ −2 %; C16 ≥ −3 %; qeval within ±1; needle 3/3; DSML; JSON 64/64; greedy 40/40; true TTFT ≤ +10 %),
control mean = A6/A7 mean.
C32 bar (replaces the single post-suite rep): median of the arm's 5 extra C32 means ≥ −3 % vs median of the 10 control reps (A6+A7).
Stall bar (replaces X3 bar 7): count of category cells with mean TTFT > 0.5 s across the arm's 5 extra reps ≤ control's per-arm
count (max of A6, A7) + 2.
The single post-suite C32 is reported, not gated.

## Promotion
Both G boots pass → promote **Pg63** = pg62-work + `overlay-vllm/v1/core/sched/scheduler.py` (cadence3) + the two env vars; Pg62 kept
with a scripted rollback (James 11/20: "if it wins, it lives as a full-time resident"). Otherwise Pg62 stays and this lever is written up as
measured-not-promoted.

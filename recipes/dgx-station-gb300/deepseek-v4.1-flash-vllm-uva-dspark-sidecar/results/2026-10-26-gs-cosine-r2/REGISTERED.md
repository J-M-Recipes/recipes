# Cosine gate R2 — registered 2026-10-24 ~08:15 CDT, before any log was re-judged under it

**Why.** R1 (min cos >= 0.999 over post-startup checks with >= 32 cold rows) fails on rounding noise. The cosine is taken
between the sidecar's and the GB300's *cold-expert* contributions, and when those carry ~0 of the layer output (Card GV2:
layer 39 on v25's exact split, min 0.99831, cold share 0.000) the cosine of two near-zero vectors says nothing about tokens.

**R2.** Same scope as R1. For each check, from the fields `mega_peer_hook._log_check` already prints:
`e = rel_diff × cold_share = ||cold_peer − cold_trt|| / ||y_trt||`, the sidecar's error as a fraction of the whole layer output.
- coverage: all 40 layers have a >= 32-row check (unchanged)
- 0 peer + 0 hook timeouts (unchanged)
- **bar B = 1.5 × the largest per-boot max e over the reference boots**, all on rowmap-static-v1 (v25 static):
  SC0 arm A (2026-10-21) + Card GS K1 and K2 (2026-10-26, after the v25m 48 h soak)
- candidate PASS iff coverage, timeouts, and max e <= B
- reported, not gated: R1 verdict, min cos over checks with cold share >= 1 %, p99 e

**Re-judged under R2 (Card GS, `cardGS_runner.sh`):** 285/99 h285 (SC3b), mix-v1 (SC3), v25m (GV), v25m2 (GV2).
The table is published whatever it says, including any boot R1 failed that R2 passes.
What was already known when R2 was written: v25m's worst-*cosine* check has e = 0.062 × 0.079 ≈ 0.0049. Not known: any
boot's max e, and every reference boot's e.
`judge_r2.py` sha1 b2b9d66fbbe0 (box copy; paths scrubbed in this one).

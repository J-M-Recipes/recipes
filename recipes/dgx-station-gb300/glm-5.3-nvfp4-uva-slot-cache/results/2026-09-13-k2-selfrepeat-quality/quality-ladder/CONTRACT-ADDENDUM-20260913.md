# Addendum 2026-09-13: K2 vs K1 on the 256K slot profile
Same frozen harness, fixtures (seed 20260906), scorer, and acceptance rules as campaign v1 (CONTRACT.md). Only the lane table changed:
- control  k1-256k: glm53-big-sc13g-mtp-ctx256k-DAILY-20260913 (7360 slots, ctx 262144, MTP K=1)
- candidate k2-256k: glm53-big-sc13g-mtp2-ctx256k-K2-selfrepeat-20260913 (identical, MTP K=2)
No V1 lane (descriptive control dropped; sole inferential comparator is k1-256k). Same PASS rule: both >=85% overall / >=80% per category; Clopper-Pearson one-sided 95% upper bound on loss proportion <0.05.
Diff vs v1 harness: image digest and CONTAINERS in run_lane.py only. frozen-files.json regenerated for that one file.

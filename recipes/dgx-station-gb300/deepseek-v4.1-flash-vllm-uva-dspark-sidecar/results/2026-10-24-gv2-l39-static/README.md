# 2026-10-24 — Card GV2: does the cosine miss come from the map at layer 39? No.

Card GV's only failing gate was the bring-up cosine check (min 0.99811, 1 of 1,439 checks, layer 39). Hypothesis: the
mixed map's layer-39 cold set is the cause. Test: `rowmap-mixed-ar-v2-l39s.json` = v25m's map with layer 39 replaced by
rowmap-static-v1's layer 39 verbatim (48 of 295 hot experts change on that layer; layers 0–38 untouched). Same check-mode
boot, same frozen prompt file, same judge (`sc0_gate.py`), bar not moved. Pre-registered: v25m2 takes the soak only if the
cosine gate passes AND the full Card GV battery is 10/10; otherwise fall back to v25m v1.

**Result: FAIL, min cos 0.99831** (1 of 1,428 gated checks below 0.999). The fallback ran: v25m v1 re-booted, watchdog
re-armed, 48 h soak restarted 2026-10-24 07:12:57 → 2026-10-26 07:12:57 CDT. No V battery was run on v25m2.

| boot | map at layer 39 | gated checks | min cos | worst check |
|---|---|---:|---:|---|
| SC0 arm A (v25 static) | static | 1,341 | 0.99906 | L39 T=128 rows=37, cold share **0.000** |
| GV K (v25m) | mixed | 1,439 | 0.99811 | L39 T=128 rows=49, cold share 0.079 |
| **GV2 K (v25m2)** | **static (identical to v25)** | 1,428 | **0.99831** | L39 T=128 rows=44, cold share **0.000** |

Reading: with layer 39's split byte-identical to v25 static, layer 39 is still the worst check and still misses. The
worst checks on both static-L39 boots have the cold experts contributing ~0% of that layer's output norm, so the cosine
there compares two near-zero vectors and is dominated by rounding noise. So layer 39's checks are noisy, and
v25 static's single passing boot (0.99906) is one sample.

**Correction (2026-10-24 08:15):** an earlier version of this file concluded "the miss is not attributable to the map".
That overreached. v25m's own failing check (Card GV) had cold share 0.079 (rel_diff 0.062), not ~0, so GV2 does not
clear v25m. Resolution goes to cosine gate R2, registered before any re-judging: `../2026-10-26-gs-cosine-r2/`.

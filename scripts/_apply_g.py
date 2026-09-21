#!/usr/bin/env python3
import pathlib
# ---- DSV4.1: resolve the "unexplained C8 -6%" and record the cgsizes trade
R = pathlib.Path("recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark")
p = R / "recipe.yaml"; s = p.read_text()
old = "Cost of the 24-slot profiles vs v15 (16 slots): knee C8 -6% (698 -> 655, two same-afternoon controls; unexplained, new autotune hash suspected), C16 flat, C1 +12%."
new = ("Cost of the 24-slot profiles vs v15 (16 slots): knee C8 -6% (698 -> 655, two same-afternoon controls; unexplained, new autotune hash suspected), C16 flat, C1 +12%. "
       "RESOLVED 2026-09-21 PM (Card G, results/2026-09-21-v20-promotion/receipts/cardG): it is the capture-size list, not the autotune. Same v20 image + hook, --cudagraph-capture-sizes omitted (engine default) vs the 15-size list, same window: "
       "C1 173 vs 180 (-4%), C4 443 vs 400 (+11%), C8 720 vs 657 (+9.6%), C16 1007 vs 969 (+4%); fund C16 warm agent p95 1.73 vs 1.68 s (flat) but "
       "fund C24 warm agent p95 9.5 vs 2.4 s (mean 2.7 vs 0.98 s) and C24 aggregate 249 vs 283 - because default graphs cost 0.7M KV tokens (pool 1.86M vs 2.55M) and at 24 seats the slot queue comes back. "
       "The list is a stated trade: -10% on the 4-8 stream knee for +4% C1, +0.7M KV and a 24-seat lane that stays warm. Kept. A 4-8 stream deployment that never runs 24 seats should drop the flag.")
assert s.count(old) == 1; s = s.replace(old, new); p.write_text(s)
# README round note
p = R / "README.md"; s = p.read_text()
old = "## Round 11b — the bisect finishes on one PR, and v20 is promoted (2026-09-21, 07:38–12:30)"
new = '''## Round 11c — the "unexplained" C8 loss was the capture-size list, and it is a trade worth keeping (2026-09-21, 14:04–15:30)

One pair on the v20 image: `--cudagraph-capture-sizes` omitted (engine default) vs the 15-size list, same hook, same window. Receipts: [`receipts/cardG/`](results/2026-09-21-v20-promotion/receipts/cardG/).

| | default capture | 15-size list (v20) |
|---|---|---|
| C1 / C4 / C8 / C16 | 173 / **443** / **720** / **1007** | **180** / 400 / 657 / 969 |
| KV pool | 1.86M | **2.55M** |
| fund C16 warm agent p95 | 1.73 s | 1.68 s |
| fund C24 warm agent p95 / mean | 9.5 s / 2.7 s | **2.4 s / 0.98 s** |
| fund C24 aggregate | 249 | **283** |

So the Sept-18 note ("24-slot profiles cost ~6% at C8, unexplained") is closed: the list costs ~10% on the 4–8 stream knee and buys +4% C1, +0.7M KV tokens and a 24-seat lane that stays warm. That is the right trade for this lane's job (the 35-seat fund harness). A deployment that lives at 4–8 streams and never seats 24 should drop the flag and take the knee. Also this window: the GLM Flash draft `bf582e4e` A/B on `:30001` — see that recipe; no change, pin kept.

## Round 11b — the bisect finishes on one PR, and v20 is promoted (2026-09-21, 07:38–12:30)'''
assert s.count(old) == 1; s = s.replace(old, new); p.write_text(s)

# ---- GLM Flash: draft pin stays, tested
G = pathlib.Path("recipes/dgx-station-gb300/glm-5.3-flash-nvfp4-dflash2")
p = G / "recipe.yaml"; s = p.read_text()
i = s.index("the newer bf582e4e draft"); j = s.index('"', i)
tail = s[i:j]
new_tail = tail + (" TESTED 2026-09-21 PM (results/2026-09-21-v0520-rebase/receipts/bf58df): draft bf582e4e on the same v0.5.20 + DFlash2 b7 launch - greedy 20/20 identical to the 7d74cdd8 draft, "
                   "accept len 3.42 / rate 0.40 vs 3.37 / 0.39, C1 246 vs 248, C8 692 vs 699, tools 10/10 x2, TF repeat-identical. No change; pin stays on 7d74cdd8")
s = s[:i] + new_tail + s[j:]; p.write_text(s)
p = G / "README.md"; s = p.read_text()
old = "**Status: verified** (rebased to **SGLang v0.5.20** September 21, 2026;"
new = "**Status: verified** (rebased to **SGLang v0.5.20** September 21, 2026; draft `bf582e4e` tested the same afternoon — greedy 20/20 identical, acceptance and speed within noise, pin stays on `7d74cdd8`;"
assert s.count(old) == 1; s = s.replace(old, new); p.write_text(s)
print("ok")

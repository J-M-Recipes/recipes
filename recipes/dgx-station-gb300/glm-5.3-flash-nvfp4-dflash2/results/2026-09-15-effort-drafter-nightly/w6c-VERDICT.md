# W6c verdict — DFlash2 block-size ladder, hook-free (2026-09-15)

Window: 21:55Z → 22:24Z, three back-to-back boots on `nightly-dev-cu13-20260915-8874c51a`, drafter 7d74, env `TORCHINDUCTOR_COMPILE_THREADS=1` only, no accept-hist hook. `glmf-w6c-b7` is the same-session block-7 control; W2 (2 h older) is the reference the bars were written against. All three containers stopped+kept, exit 0. `:30001` is down.

## Absolute numbers (C1, effort=low, median of 3)

| Block | recipe-method | prose | code | shell | accept len / rate | tools | greedy vs W2 |
|---|---:|---:|---:|---:|---|---|---|
| **7** (control) | 202.6 | 151.3 | 289.0 | 162.4 | 3.46 / 0.41 | 10/10 ×2 | 20/20 |
| **6** | 207.0 (+2.2%) | 158.0 (+4.4%) | 286.7 (−0.8%) | 165.8 (+2.1%) | 3.26 / 0.45 | 10/10 ×2 | 20/20 |
| **5** | 212.3 (+4.8%) | 156.7 (+3.6%) | 271.1 (−6.2%) | 162.9 (+0.3%) | 3.03 / 0.51 | 10/10 ×2 | 20/20 |

W2 reference: recipe 202.7 / prose 151.4 / code 289.3 / shell 162.4. The b7 control reproduces W2 within 0.1% on every class — boot-to-boot noise on this image is negligible for C1.

Per-run spreads: every C1 class ≤ 2.5% except b5 shell (2.8%). Tight.

## Gates

**b6 bar:** recipe ≥ 208.8 AND code ≥ 280.6 AND tools AND greedy.
- recipe 207.0 — **miss by 0.9%** (bar was W2 × 1.03)
- code 286.7 — pass
- tools, greedy — pass
- **Verdict: not a win on the bar.** Real but small: +2–4% on prose-like, flat on code. Mirrors the hooked W6 (+3.2% / +4.9% / −0.8%).

**b5 bar:** recipe ≥ 208.8 AND prose ≥ 156, code regression tolerated to −6%.
- recipe 212.3 — pass
- prose 156.7 — pass
- code 271.1 — **−6.2%, just outside the −6% tolerance**
- tools, greedy — pass
- **Verdict: passes the bars it was set, misses the tolerance by 0.2 pt.** Not a clean win either.

## C8 — two of three rows are void

| Block | C8 agg | spread | status |
|---|---:|---:|---|
| 7 | 506.6 | 113% (706/681/**133**) | **void** — leaked pass in rep 3 |
| 6 | 526.1 | 116% (749/**138**/691) | **void** — leaked pass in rep 2 |
| 5 | **775.9** | 3% (792/765/770) | valid |

Rule: spread > 10% at C≥8 = leaked warm pass, rerun. The leaked rep in both b7 and b6 is a ~130 tok/s outlier; the other two reps sit at 680–750, i.e. in the same band as b5. **Nothing here says b5 is faster at C8.** The 775.9 vs "635 (W3 rewarm)" comparison in the run header is apples-to-oranges until b7/b6 C8 is rerun clean. Also `max_running_requests` is capped at 9 by mamba 48 on all three, so C8 is the last shape before the cap; C16/C24 were not run.

## What the ladder actually says

Block size trades draft depth for acceptance: b7 commits 3.46 at 0.41, b5 commits 3.03 at 0.51. Net C1 is within ±5% across the ladder on every class; the direction flips by class (prose favours short blocks, code favours long). There is no block size that wins everywhere, and the largest single move is b5's −6% on code.

**Recommendation: keep block 7 as the recipe default.** It is the code-best setting, it is what the 09-11 recipe published, and the prose-side gain from b5/b6 (+4%) is smaller than the code-side loss. If a concurrency win for b5 exists it has to be shown with a clean C8/C16 rerun of all three — not from this data.

## Not done / open
- C8 rerun of b7 and b6 (void rows). ~15 min if the window is reopened. Only worth it if we care about the b5 concurrency story.
- No `mamba` pool per-slot comparison was extracted from boot-excerpt across blocks (W6 hooked run has it; W6c boot excerpts exist in each dir).
- `reasoning_tokens=None` on every row: the split still isn't reported as a usage field on this image (#39227 fix expected in 09-16 nightly). `reasoning_len` shows the split itself works at effort=low.

## Files
`w6c/w6c-b{5,6,7}/` — c1-low/max, accept-low, tool-1/2, knee-c8, greedy-*.json, boot-excerpt, meta. `w6c/w6c_run.log` — full runner log.

# 2026-10-25 — Card BR2: a DSpark k ladder (k=9 with block verification) + the BRRR peer harness on every arm

Two questions on the production sidecar lane (v25m map + kv-host-tier flags, image
`vllm/vllm-openai@sha256:3af6a45b…`, nightly af7f9488):

1. **Does a longer draft pay?** The K9 arm = block verification (`"rejection_sample_method":"block"`) +
   `num_speculative_tokens 9` + `num_speculative_tokens_per_batch_size [[1,4,9],[5,8,3],[9,32,3]]`. The controls A and
   A2 are the **production spec config** (sequential verification, k=5, `[[1,4,5],[5,8,3],[9,32,3]]`). The K9 ratios
   therefore measure block + k=9 against production. They do not isolate k. Card BR1 measured block verification
   alone at −0.4% to +5.4% (C1) at temperature 1.0.
2. **How does this lane read on someone else's bench?** Each arm also ran a stdlib port of
   [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR)'s
   `src/brrr/bench.py` (Apache-2.0, credit header kept): [`scripts/peerbench.py`](scripts/peerbench.py). Every change
   from upstream is listed in [`scripts/DIFF-br2-peerbench-vs-upstream.txt`](scripts/DIFF-br2-peerbench-vs-upstream.txt).

Bracketed **A → K9 → A2**, 2026-10-25 21:36 (lane dark; first boot 21:38) – 23:24 CDT. Each arm got a fresh
sidecar + server boot, the watchdog was armed in enforce mode on each, and the lane was firewalled to loopback while
dark. KV was 4,400,013 tokens on all arms. The K arm is **exploratory** and was never gated on speed.

## Boot: k=9 was a hash hit

| arm | kspec | schedule | boot s | autotune loaded / saved / new | same cache dir | CUDA-graph tokens 32×(k+1) | max under the schedule |
|---|---|---|--:|---|---|--:|--:|
| A | 5 | `[[1,4,5],[5,8,3],[9,32,3]]` | 529 | 117 / 117 / 0 | yes | 192 (> 128) | 128 |
| K9 | 9 | `[[1,4,9],[5,8,3],[9,32,3]]` | 529 | 117 / 117 / 0 | yes | 320 (> 128) | 128 |
| A2 | 5 | `[[1,4,5],[5,8,3],[9,32,3]]` | 529 | 117 / 117 / 0 | yes | 192 (> 128) | 128 |

The card's brief predicted that k=9 would force a fresh autotune (25–35 min, with a new hash dir). It did not:
`num_speculative_tokens` is not in the autotune hash on af7f9488 ([`raw/K9/facts.json`](raw/K9/facts.json),
[`raw/K9/autotune-dirs.txt`](raw/K9/autotune-dirs.txt)). The checkpoint's `dspark_block_size` is 5. The DSpark
speculator emits N = `num_speculative_tokens` query positions (the anchor plus N−1 noise positions) in one pass,
and the engine does not cap N. The 32×(k+1) figure is an annotation only: the batch schedule caps k at 3 above
batch 4, so on every arm the largest real decode batch is 128 tokens, which fits the 128-entry capture list.

## Verdicts — both on the record

- [`VERDICT.txt`](VERDICT.txt) / [`verdict.json`](verdict.json): **INVALID-CONTROL-A-A2** (pre-registered). This is
  the same G0 defect as Card BR1: the container env was compared as an ordered list, and Docker returns the same
  44 variables in varying order.
- [`VERDICT-rescore.txt`](VERDICT-rescore.txt) / [`verdict-rescore.json`](verdict-rescore.json): **PASS `BR2PASS`**
  (`scripts/br2_verdict_rescore.py`, which differs from `br2_verdict.py` by the one-line set compare). The controls
  agree at C1 within 0.0–3.7% per class, peerbench is 12/12 OK on every arm, and the watchdog read 0 timeouts and
  0 trips on every arm.

## Speed, temperature 1.0 (`br2_perclass_decode_t.py`, 40 prompts/class × 256 forced; mixed = `ar_mixed_t.py` C1)

| arm | prose C1 | structured C1 | code C1 | shell_ops C1 | tool_json C1 | mixed C1 |
|---|---|---|---|---|---|---|
| A | 271.1 | 286.5 | 356.0 | 371.4 | 339.6 | 212.3 |
| **K9** | 222.4 (**0.813**) | 250.9 (**0.860**) | 308.4 (**0.868**) | 340.9 (**0.918**) | 291.6 (**0.853**) | 168.4 (**0.793**) |
| A2 | 275.9 | 297.4 | 354.3 | 371.3 | 343.8 | 212.5 |

C8, K9 ÷ mean(A, A2): prose 0.896, structured 0.891, code 0.925, shell_ops 0.930, tool_json 0.882.

Tokens per verify step at C1 (1 + accepted/drafts; eff k = draft tokens / drafts):

| arm | prose | structured | code | shell_ops | tool_json | eff k |
|---|--:|--:|--:|--:|--:|--:|
| A | 2.85 | 3.03 | 3.83 | 3.98 | 3.63 | 5 |
| K9 | 2.60 | 2.94 | 3.66 | 4.04 | 3.45 | 9 |
| A2 | 2.91 | 3.15 | 3.80 | 3.97 | 3.66 | 5 |

## Per-position acceptance, C1, temperature 1.0 ([`raw/per-position-curves.json`](raw/per-position-curves.json))

curve[p] = accepted-at-p / drafts (`vllm:spec_decode_num_accepted_tokens_per_pos`); marginal[p] = curve[p] / curve[p−1].

| class · arm | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prose · A (k5) | .717 | .476 | .310 | .205 | .145 | | | | |
| prose · A2 (k5) | .724 | .489 | .322 | .217 | .157 | | | | |
| prose · K9 | .612 | .379 | .235 | .147 | .097 | .063 | .036 | .020 | .011 |
| prose · K9 marginal | .61 | .62 | .62 | .62 | .66 | .65 | .57 | .57 | .54 |
| structured · K9 | .655 | .449 | .303 | .204 | .145 | .094 | .052 | .027 | .015 |
| structured · K9 marginal | .66 | .69 | .67 | .67 | .71 | .65 | .55 | .53 | .55 |
| code · A (k5) | .823 | .653 | .536 | .445 | .371 | | | | |
| code · A2 (k5) | .819 | .652 | .529 | .434 | .363 | | | | |
| code · K9 | .719 | .548 | .428 | .322 | .255 | .179 | .108 | .064 | .037 |
| code · K9 marginal | .72 | .76 | .78 | .75 | .79 | .70 | .60 | .60 | .57 |
| shell_ops · K9 | .759 | .613 | .490 | .382 | .315 | .225 | .142 | .075 | .039 |
| shell_ops · K9 marginal | .76 | .81 | .80 | .78 | .82 | .72 | .63 | .53 | .51 |
| tool_json · K9 | .696 | .511 | .385 | .294 | .228 | .155 | .096 | .054 | .036 |
| tool_json · K9 marginal | .70 | .73 | .75 | .76 | .78 | .68 | .62 | .56 | .67 |

**Read:** on the K9 arm the marginal acceptance runs 0.61–0.82 through position 5, which is the drafter's trained
block. From position 7 it falls to 0.51–0.67. Position-1 acceptance also falls under the longer block (prose .72 →
.61, code .82 → .72). The extra positions cost more than they return, so **k=5 stays**. Mixed traffic is worse:
the K9 mixed-C1 curve is .466 / .210 / .089 … against A's .588 / .294 / .129. The per-position histogram is now a
standing instrument.

## Peer harness — arm A is ours, K9 exploratory

Method mirrored from their `bench.py`: a coding prompt padded to 2K/16K/64K through `/tokenize`, with a nonce per
stream. Each round makes two passes: `max_tokens 1` for TTFT/PP, then the same prompts from cache so all streams
decode together. TG counts only while every stream decodes, and the TG window is the median tokens per stream.
Generation stops at natural EOS (≤ 2048), thinking is off, and temperature 1.0 / top_p 1.0 are sent as their
bench sends them. Concurrency C1/4/8/16. Differences from their run: **2 rounds** (theirs 5); **no cool-down wait**
(the hottest GPU temperature is logged per round instead); our server omits `prompt_tokens_details`, so the
cache-bust check reads the `/metrics` prefix-cache counter instead.

Tables per arm: [`raw/A/peerbench-BR2-A.md`](raw/A/peerbench-BR2-A.md), [`raw/K9/`](raw/K9/),
[`raw/A2/`](raw/A2/) (`.md` + `.json`). A2 agrees with A within 0–2.5% on TG p50 and TG total in all 12 cells. PP
p50 and TTFT agree within 2% in 10 of 12 cells; the exceptions are 2K/C1 (PP 14,279 vs 15,498, −8%; TTFT 152 vs 139 ms)
and one more cell at 4.4%.

| prompt · conc | TG p50 theirs / ours | PP p50 theirs / ours | TTFT p50 theirs / ours | TG total theirs / ours | accept theirs / ours |
|---|---|---|---|---|---|
| 2K · C1 | 703 / 466 | 12,452 / 14,279 | 172 / 152 ms | 706 / 467 | 5.94 (71%) / 5.13 (83%) |
| 2K · C4 | 329 / 321 | 4,470 / 8,137 | 489 / 264 ms | 1,320 / 1,287 | 5.94 / 5.15 |
| 2K · C8 | 231 / 213 | 2,589 / 5,259 | 829 / 408 ms | 1,847 / 1,697 | 5.97 / 3.76 |
| 2K · C16 | 166 / 169 | 1,610 / 3,357 | 1,333 / 643 ms | 2,648 / 2,719 | 5.96 / 3.70 |
| 16K · C1 | 684 / 471 | 16,953 / 47,297 | 972 / 349 ms | 691 / 470 | 5.99 / 5.21 |
| 16K · C4 | 321 / 319 | 7,199 / 16,356 | 2,362 / 1,025 ms | 1,281 / 1,277 | 5.95 / 5.05 |
| 16K · C8 | 226 / 208 | 4,072 / 10,271 | 4,089 / 1,617 ms | 1,806 / 1,665 | 5.99 / 3.76 |
| 16K · C16 | 158 / 170 | 2,167 / 5,944 | 7,624 / 2,781 ms | 2,522 / 2,715 | 5.98 / 3.70 |
| 64K · C1 | 675 / 453 | 17,852 / 46,246 | 3,677 / 1,419 ms | 669 / 454 | 5.97 / 5.05 |
| 64K · C4 | 312 / 316 | 7,435 / 18,847 | 9,165 / 3,597 ms | 1,243 / 1,266 | 6.02 / 5.13 |
| 64K · C8 | 210 / 209 | 4,051 / 10,860 | 16,396 / 6,098 ms | 1,686 / 1,675 | 5.95 / 3.81 |
| 64K · C16 | 131 / 165 | 2,131 / 6,045 | 30,893 / 10,890 ms | 2,096 / 2,653 | 5.98 / 3.72 |

Provenance: **theirs** = the "Thinking off" table on the BRRR model card (4× RTX PRO 6000 Blackwell Max-Q 325 W,
Threadripper 9960X, 64 GB RAM, Samsung 990 PRO, adaptive verification off, DSpark 7 draft tokens, 5 rounds),
copied, not re-run by us. **Ours** = arm A of this card (one GB300 + one RTX PRO 6000 Max-Q sidecar, k=5 at C1–C4
and k=3 at C5+, 2 rounds). Our C8/C16 acceptance is lower by construction because the batch schedule drops k to 3
there.

**Decomposition at 2K/C1.** Their 703 tok/s = 5.94 tok/step × 118 steps/s (8.45 ms per step). Ours: 466 = 5.13 ×
91 (11.0 ms per step). The 1.51× gap factors as 1.16× tokens per step × 1.30× steps per second. Tokens per step
favour them because k=7 on a friendly codegen prompt beats our k=5. Step time is the larger factor. Two hypotheses
we did not test in this card: our sidecar hop grows with the expert union under high acceptance, and their
persistent fused `decoder.cu` removes small-kernel launch churn per step. In their accept column, `(71%)` is
accepted ÷ draft tokens = 4.94 / 7, not an independent per-position rate.

**K9 on the peer harness** ([`raw/K9/peerbench-BR2-K9.md`](raw/K9/peerbench-BR2-K9.md)): TG p50 ties A at 2K/C1 (468
vs 466, 5.70 tok/step), 2K/C4 (321 vs 321) and 16K/C1 (469 vs 471). It wins 64K/C1 (489 vs 453, 6.05 tok/step), is
slightly lower at 16K/C4 (316 vs 319, 5.95) and 64K/C4 (306 vs 316), and is lower at every C8/C16 cell (−4 to
−7%). Step time at 2K/C1 is 12.2 ms vs A's 11.0 ms. So k=9 can pay on padded single-stream codegen, and it loses
on the real-traffic classes above.

## Files

| path | what it is |
|---|---|
| `VERDICT*.txt`, `verdict*.json` | both scorings; every annotation above (boot shape, speed, tokens/step, per-position, peerbench tables) is in `VERDICT-rescore.txt` |
| `arms.json`, `instruments.txt` | arm table (spec extra, kspec, schedule), sha1 of every script/fixture as run on the box, image digests |
| `SUMMARY.txt`, `runner.log`, `phases.jsonl`, `launch-*.txt`, `health-*.json`, `errcount-*.txt`, `oom-*.txt`, `mem.tsv`, `autotune-dirs-*.txt` | runner timeline and per-arm operational receipts |
| `raw/<arm>/` | `facts.json` (kspec/ksched/block in args, cg tokens, autotune loaded/saved/new + dirs), boot logs, `args.json`/`env.json`, `perclass-t1/` (tables md+json, **rows stripped**), `mixed-*.json.gz`, `peerbench-BR2-<arm>.md/.json`, `specmetrics/`, `turns-*-summary.json`, `wd.json` |
| `raw/per-position-curves.json` | per-position curves, marginals, tokens/step and eff k for every arm and class (C1 and C8) and mixed C1 |
| `raw/gauntlet-<arm>/`, `raw/gauntlet-log-*.txt` | gauntlet 12/12 on A, K9 and A2 (temp 0) |
| `raw/production/` | production lane args/env before and after |
| `scripts/` | `cardBR2_runner.sh`, `launch_br2.sh`, `br2_verdict.py`, `br2_verdict_rescore.py`, `br2_perclass_decode_t.py`, `peerbench.py`, `perclass_run_t.sh`, `ar_mixed_t.py`, and `DIFF-*.txt` against their parents and upstream, as run, paths scrubbed |
| `throughput.csv` | per-class, mixed, tokens/step, and every peerbench TG/PP cell per arm |

Withheld as in Card BR1: the private per-class prompt file (sha1 `618898aac432`) and per-request `rows`; the replay
turns are reduced to per-pass counts; post-cell server logs stay on the box. Path scrub: box home → `~`, LAN → `<gb300>`.

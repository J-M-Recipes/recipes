# KDA snapshot cap probe — 2026-10-01 (Card M)

**Question.** HelixML (2026-09-26, "GLM-5.3-Flash Ran Out of Cache Snapshots, Not Cache Tokens") found that on hybrid KDA models a prefix-cache hit
needs a saved recurrent-state snapshot at the divergence point, and the snapshot pool (shared with running requests) is far scarcer than the KV
token pool: one ~300K prompt writes ~50 chunk-end snapshots and LRU-evicts every other session's. Their fix: `--mamba-max-states-per-path 2`
(sgl-project/sglang#31230) + a host tier. Does this bite our daily?

**Setup.** Promoted daily image `glmf-sglang:0922-582389ce-revert39688-tf5.16.1`, daily knobs (48 mamba slots → 9 running, DFlash2 b7, ReplaySSM, 1M
ctx). Three arms, one axis: cap −1 (stock) / 2 / 4. Probe `kda_snapshot_probe.py` (stdlib): prime 6×20K sessions → re-send → one 300K prompt →
re-send → branch at 70% → 12-session LRU loop. 1-token requests, thinking off, `/flush_cache` between phases. Two reps per arm. Then knee C8/C16,
greedy 20 vs daily, TF vs v0.5.20 ctrl, tools ×1.

**Result.**

| arm | warm re-send | after 300K prompt | 12-loop hits | branch @70% TTFT | C16 knee | greedy | TF | tools |
|---|---|---|---|---|---|---|---|---|
| cap −1 | 0.21 s | 0.21 s (all warm) | 12/12 | 0.52 s | 710 | 20/20 | 0.000 | 10/10 |
| cap 2 | 0.21 s | 0.21 s | 12/12 | **0.78 s** | 716 | 20/20 | 0.000 | 10/10 |
| cap 4 | 0.20 s | 0.20 s | 12/12 | 0.51 s | 701 | 20/20 | 0.000 | 10/10 |

Cold 20K prefill on this lane is 0.78 s; warm is 0.21 s. Every post-eviction re-send was warm on every arm. The cap-2 branch cost is the
expected one: the interior snapshot the branch would resume from has been pruned, so it resumes from the root.

**Verdict.** Not adopted. The mechanism is real (the flag and `_evict_excess_path_states` are in this image) but our pool is not under pressure at
this traffic shape. Worth revisiting only if the Flash lane serves many long concurrent sessions.

**Open.** By Helix's arithmetic (4 snapshots/running request, ~50 from a 300K prompt) 48 slots *should* have evicted. Either snapshot accounting
differs on this build, or an evicted KDA state is recomputed from the KV prefix cheaply enough that 0.21 s hides it. One follow-up card could
pin which; low priority.

**Instrument notes.** `usage.prompt_tokens_details.cached_tokens` is absent on this image unless `return_cached_tokens_details: true` is sent; the
probe read wall time only. C8 knee rows were unusable (one 700+ rep against two ~130 reps in every arm — the 9-running vs 8-stream queue
colliding with prefill), so only C16 is reported. Receipts: `results/2026-10-01-cardM-kda-snapshot-cap/`.

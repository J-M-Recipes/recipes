# Card G1 — Garner's tree under our policy, bracketed against h2-REF (2026-10-29)

**Verdict: INCOMPLETE — CRASH.** Both Garner arms (P and K) died with `CUDA error: an illegal memory access` in the
multi-stream stage, on a **32-request × 4-token pure-decode step** (`total_num_scheduled_tokens=128`,
`num_running_reqs=32`, KV usage 0.7 %). Every cell after the crash (per-class C1/C8, TF capture, gauntlet, 1M needle)
logged `SERVER DOWN` and has no data. Controls A and A2 ran every cell. Card G1b (crash attribution) follows.

Lane: dark 08:55–09:48 CDT for the Garner arms; h2-REF pair restored, smoke OK, watchdog armed, then A2 measured; isolate
off at 10:06. Runner `g1_runner.sh` sha1 `22710f2f`.

## Arms

| arm | what |
|---|---|
| A, A2 | h2-REF (production pair), isolated |
| P | `pol` tree (his 8043a681 + 5 no-op-unless-set hooks), `fp8_ds_mla`, default indexer, `--max-model-len 1048576`, 48 GB host KV, no sampling/effort override, no API key; components = his 29 minus `ds41_sampling, ds41_stop, w8oproj, lowb_aux, tail_drafthead_mxfp8` (24 ACTIVE, 0 fallback) |
| K | P + `nvfp4_ds_mla` + his `--attention-config` (MXFP4 indexer K, sparse logits) |

## Speed (his `bench_full.py`; single-stream cells completed before the crash)

| cell | A | A2 | P | K | P / ctrl | K / ctrl | K / P |
|---|--:|--:|--:|--:|--:|--:|--:|
| fixed coding prompt tok/s | 446.9 | 447.4 | 636.3 | 706.1 | 1.42× | 1.58× | 1.11× |
| tokens/step | 4.71 | 4.69 | 5.37 | 5.39 | 1.14× | 1.15× | 1.00× |
| **steps/s** (ms/step) | 94.7 (10.56) | 95.8 (10.44) | **119.6 (8.36)** | **131.0 (7.63)** | **1.26×** | **1.38×** | **1.095×** |
| GSM8K T=1.0 thinking, C1 | 334.2 | 335.7 | 451.2 | 484.9 | 1.35× | 1.45× | 1.075× |
| GSM8K greedy, C1 | 397.4 | 400.9 | 548.5 | 582.3 | 1.37× | 1.46× | 1.06× |
| 8 streams agg | 1,120 | 1,124 | — | — | | | |
| 32 streams agg | 1,490 | 1,463 | **crash** | **crash** | | | |
| GPU KV tokens at 1M | 4,400,013 | same | 11,696,582 | (see boot.log) | 2.66× | | |

Controls agree within 1.2 % on steps/s. Reference points: G0 (his release as shipped, 512K) = 135.9 steps/s; so of
G0's 1.43× over h2-REF, **our policy keeps 1.26×**, the KV ruling costs **~9.5 % in steps/s** (K vs P), and the last
~3.7 % (K → G0) is his three numerics-changing components + 512K + his sampling defaults.

Where the P/K speed comes from at C1 is step time, not just acceptance: tokens/step rises 14 % on the fixed prompt
(his copy/suffix drafting + drafter fusions), step rate 26 %.

## Controls' per-class decode (ours, temp 1.0, rep 1)

A/A2 C1 per class 283–393 tok/s; C8 974–1,280 agg. Full tables `G1/{A,A2}/perclass-t1/`. TF capture 39 docs / 79,544
positions, 0 errors on both; gauntlet 12/12 on both.

## The crash

```
EngineCore ... torch.AcceleratorError: CUDA error: an illegal memory access was encountered
  (reported at async_utils.get_output -> copy_event.synchronize; the real fault is earlier on the stream)
SchedulerOutput: 32 cached reqs, num_scheduled_tokens 4 each (k=3 band), total 128; KV usage 0.7 %
```

Same scheduler shape as the sidecar lane's Card AP split-KV deaths (32 × 4 = 128 decode tokens). G0 (his release at
512K, 120 GB host KV, all 29 components) ran C32 for a 60 s window without faulting. Candidate triggers, in order:
1M max-model-len (buffer widths: W3 candidate select logs `width 1048576`; block tables double), the five removed
components, 48 vs 120 GB host KV, sampling/effort defaults. Card G1b bisects the first against the rest.

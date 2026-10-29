# 2026-10-29 · Card TR2 — L2 attention/indexer top-k table sharing on the GB300-only lane

**Verdict: `PASS` on the pre-registered gates, `NOT ADOPTED` on the result.** The lever is real and numerically
exact — 30 of 38 compressed-layer calls per decode forward disappear, and 2,670 in-process re-computations returned
bit-identical tables — and it buys **+1.1 % at C1** on the primary cell. But C8 moved **−2.7 %** with four of five
classes below *both* controls, and that is not explained. It stays out of the launch line until a C8-focused cell
says whether that is noise or a real cost. Reference stays `v28-prob`.

Pre-registration, written before the first bracket boot and unedited since: [`PREREG-TR2.md`](PREREG-TR2.md).
Machine verdict: [`raw/verdict.json`](raw/verdict.json). Runner log: [`raw/runner.log`](raw/runner.log).

## The axis

One change: memoise `vllm.models.deepseek_v4_1.nvidia.flashmla.compute_global_topk_indices_and_lens` inside one
decode forward, keyed by `(kv_source_layer_id, index_source_layer_id, compress_ratio)`, for layers that are not their
group's index source ([`share_topk.py`](share_topk.py), unit tests [`test_share_topk.py`](test_share_topk.py)).

Everything else is the release the lane already documents as `v28-prob`: image `vllm/vllm-openai:deepseekv41-flash-0909`,
hook v26ar.2, seed map `rowmap-mixed-ar-v1.json`, `--cpu-offload-gb 60`, k-schedule `[[1,8,5],[9,24,1]]`, 24 seqs,
`draft_sample_method: probabilistic`, same autotune cache. The overlay is python-only, so the FlashInfer hash is
untouched: all three arms loaded **189 configs, 0 new** (hash `62426808`), and `GPU KV cache size` is
**2,480,774 tokens on A, B and A2**.

Arms: **A** (`L2_SHARE=0`, installed but report-only) → **B** (`L2_SHARE=2`) → **A2** (report-only), bracketed in one
window, production regime (adaptive residency unfrozen — GO + `{"freeze": false, "baseline": true}`), lane isolated
with a `:30006` non-loopback REJECT for the whole card.

## What the instrument saw

Per decode forward, on the live lane:

```
A  : {'src': 8, 'hit': 0, 'hit_noskip': 30, 'miss': 0}     # the lever is counted, every kernel still runs
B  : {'src': 8, 'hit': 30, 'hit_noskip': 0, 'miss': 0}     # 30 of 38 calls skipped
```

Totals over B's boot: `src 6,024 · hit 22,590 · miss 0`. A: `hit_noskip 22,320 · hit 0`. The ratio (3.75) is the
30/8 structure of this checkpoint's KV-source map, exactly as predicted from `attention.py:263-291`.

**One observation worth keeping:** during CUDA-graph capture the same stamps *missed* (30 misses per capture
forward) and at runtime they hit. The metadata tensors are fresh objects per capture forward and stable ones per
steady-state step. A card that only instrumented capture would have concluded the lever does not exist.

## Speed — the pre-registered cells

`perclass_run_t.sh`, temperature 1.0, 40 real Hermes transcripts per class, 256 forced tokens, two reps, C1 and C8.

| class | C1 A | C1 B | C1 A2 | **B ÷ mean(A,A2)** |
|---|--:|--:|--:|--:|
| prose | 215.53 | 219.75 | 219.40 | **1.0105** |
| structured | 243.63 | 244.75 | 245.67 | **1.0004** |
| code | 280.65 | 280.29 | 277.93 | **1.0036** |
| shell_ops | 328.14 | 336.37 | 322.45 | **1.0340** |
| tool_json | 304.28 | 306.87 | 304.97 | **1.0074** |
| **mean** | | | | **1.0112** |

Bar was: no class below 0.98, headline ≥ 1.005 for WORKS. Met (+1.12 %).

| class | C8 A | C8 B | C8 A2 | B ÷ mean(A,A2) |
|---|--:|--:|--:|--:|
| prose | 486.2 | 494.1 | 500.3 | 1.0017 |
| structured | 597.4 | 587.7 | 597.4 | 0.9838 |
| code | 652.2 | 627.2 | 657.9 | 0.9575 |
| shell_ops | 740.8 | 686.1 | 714.8 | 0.9427 |
| tool_json | 728.2 | 752.2 | 809.6 | 0.9783 |
| **mean** | | | | **0.9728** |

**This is the open cell.** The controls themselves disagree by up to 11 % (tool_json 728 vs 810), so C8 on this lane
is noisy — but B sits below *both* controls in four of five classes, which is not what a launch-removal change should
do. Two candidate mechanisms, neither tested: (a) the skipped kernel was acting as a scheduling/sync point that the
8-stream decode benefits from; (b) at the larger capture sizes the eager break lands differently and the memo's
decision interacts with graph replay. Reopen with a C8-only bracket (≥4 reps/class, both controls, plus a `hit`
counter check per cell).

Temperature-0 parity (`perclass_run.sh`): ratios 1.0086 / 1.0128 / 1.0102 / 1.0263 / 1.0206 — B never slower, and
acceptance counts move by ≤1.5 %, inside this lane's documented boot-to-boot spread.

## Identity — two independent instruments

1. **In-process verification on the live lane** ([`raw/verify-b.log`](raw/verify-b.log), dedicated boot with
   `L2_VERIFY=1`): on every cache hit the original op is re-run and `torch.equal()`-ed against the shared result.
   **`verify_ok 2,670 · verify_bad 0`** — 0 mismatches, temperature 1.0, including the wider decode batches
   (`src 32 · hit 120` forwards).
2. **Teacher-forced divergence** (prefill only, 79,544 positions over 39 docs):
   B-vs-A **1.085 %** top-1 flips, A-vs-A2 (control-vs-control floor) **1.066 %**, A2-vs-B **1.050 %**;
   mean |Δlogp| 0.0284 vs the 0.0276 floor. B is at the floor — no fidelity signal.

## Not run

- **Gauntlet.** The runner's `cell` helper lost the `cd` for two cells (it joined argv with `$*` and the inner
  `bash -c` ate the `cd`); the gauntlet therefore ran from the wrong cwd and exited rc=2 on all three arms. The TF
  cells hit the same bug and were re-run by hand (that is why [`raw/`](raw) carries the compare logs). The published runner
  ([`scripts/cardTR2_runner.sh`](scripts/cardTR2_runner.sh)) fixes it; the as-run copy is kept verbatim as
  [`raw/cardTR2_runner.as-run.sh`](raw/cardTR2_runner.as-run.sh).
- **C16+ / 1M / soak / BFCL / GSM8K.** None were in the pre-registration.

## Why it is not adopted

+1.1 % at C1 and a clean identity story is a real result, but the box's production shape is a 24-seat agent lane
where C8 is the cell that matters, and −2.7 % there is unexplained. Keep the overlay published and reproducible,
do not put it in the launch line, and settle C8 first.

## Reopen conditions

- A C8-focused bracket (4 reps/class, both controls, `hit`-counter check per cell) recovers C8 to ≥0.99 of the
  control mean **and** keeps C1 ≥1.005 → adopt.
- Or the mechanism is identified and removable (e.g. the skip needs to keep a scheduling anchor at ≥5 streams).
- A rebase changes `flashmla.py` and the call site moves: re-derive rather than port.

## Files

- [`PREREG-TR2.md`](PREREG-TR2.md) — frozen bars.
- [`share_topk.py`](share_topk.py), [`test_share_topk.py`](test_share_topk.py) — the overlay and its 9 CPU tests
  (the same tests pass inside the image: `docker run --rm -v $PWD:/l2:ro --entrypoint python3 <image> /l2/test_share_topk.py`).
- [`sitecustomize.py`](sitecustomize.py) — installs the pin-hot hook and then the overlay; a slide-in, not a rebuild.
- [`launch_tr2.sh`](launch_tr2.sh) — the BR3/v28-prob launcher plus the overlay mount and `L2_SHARE`.
- [`scripts/cardTR2_runner.sh`](scripts/cardTR2_runner.sh) — the bracket (A → B → A2), cells, restore.
- [`tr2_verdict.py`](tr2_verdict.py) — the gate arithmetic, from receipts only.
- [`verify_b.sh`](verify_b.sh) — the in-process identity boot; [`restore_after.sh`](restore_after.sh) — puts the
  production pair and watchdog back.
- [`raw/`](raw) — runner log, verdict JSON, per-arm facts + L2 counters + per-class tables, TF compare logs.

Lane restored to `dsv41-vllm-1017-h2-REF` (+`-rtx`) with the fail-stop watchdog; `:30006` un-isolated.
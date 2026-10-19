# Benchmarking recipes with llm-inference-bench

Every recipe's throughput and quality rows that come from a standard bench use one pinned tool:
[`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench).
Our own instruments (knee/replay/teacher-forced divergence/BFCL gate) stay authoritative for
recipe headlines and promotion. The standard bench gives comparable, repeatable baselines.

## Pinning

- Clone at a release tag. Record `tag @ commit` in `results/<run-id>/bench-version.txt`. Current: **v0.7.7 @ `c71ec1f2`**.
- Run with `LLM_BENCH_NO_UPDATE_CHECK=1` (the script otherwise offers a self-upgrade on start) and `--display-mode plain`.
- Do not use `lil-bench` (the container wrapper) unless you mean to upload. We publish here, not on their board.
- The upstream repo has **no license file**. Do not copy its code or datasets into this repo. Publish only our output JSON and logs.

## Tiers

| Tier | When | Command shape | Time |
|---|---|---|---|
| T0 gate | every launch | `/v1/models` id, exact-word chat smoke, parsed `tool_calls[]` smoke | ~2 min |
| T1 perf | any recipe change | `--concurrency 1,4,8 --contexts 0,32768,131072 --duration 30` | ~10 min |
| T2 quality | engine / kernel / quant / speculation change | `--test-profile needle-checksum`; `--test-profile gsm8k --profile-runs 200 --compare-baseline <last promoted gsm8k json>`; optional `--tool-eval` subset | ~10–60 min |
| T3 promotion | before a new default route, or after a hardware/fabric change | full matrix, full GSM8K + MMLU-Pro, full `--tool-eval`; multi-node adds `--p2pmark-only` | hours |

**Promotion bar:** a paired McNemar p < 0.05 accuracy *loss* on T2 fails. Needle-checksum must stay EXACT N/N. T1 must stay within the run-to-run band of the last promoted run on the same workload.

## Labelling rules (the checker does not enforce these; reviewers do)

1. A T1 row's `workload` says **"llm-inference-bench synthetic filler"**, the context, and the concurrency. Speculative-decoding acceptance depends on content, so filler tok/s never replaces a recipe's own prompt-class headline.
2. Save the vLLM/SGLang `Running:`/`Waiting:` log lines from the bench window as `<run>.engine-load.txt`. A cell where other clients were running is relabelled or re-run, never averaged in.
3. Note any server-side cap the bench hit (e.g. `--max-num-seqs` below the bench's concurrency).
4. Scrub LAN IPs and local paths from the bench JSON before committing. The bench records `--host` and its own output path.

## C1 headline rule (speculative-decoding recipes)

The first `metrics:` entry of a recipe is its index headline. For a recipe that serves with a drafter (MTP, DSpark,
DFlash, EAGLE), that entry follows three rules.

1. **Mean of 3.** C1 tok/s is the mean of three measured runs on one boot (RonanLabs `kbench_chat_ds.py` pools, or
   three reps of the recipe's own instrument). List the three values in `aggregation`. A single run is not a headline.
2. **Acceptance next to tok/s.** Set `acceptance_len` = 1 + accepted / drafts from the `vllm:spec_decode_*` (or SGLang
   equivalent) counter deltas around the same three runs. This is tokens per verify step, bonus token included.
   el8's "accepted tokens per draft" excludes the bonus token: convert (add 1) before publishing. A tok/s change with
   an acceptance change is a drafter/content effect; a tok/s change at flat acceptance is a kernel/scheduler effect.
3. **Split by content class.** Acceptance depends on content (on DeepSeek-V4.1-Flash: prose ~2.3, shell/tool JSON
   ~4.7-5.0 tokens per step), so one blended number hides the spread. Add `by_class` with tok/s and `acceptance_len`
   per class (prose, structured, code, shell_ops, tool_json) from the same boot. State the class instrument and its
   sample count in `aggregation`.

`scripts/render_index.py` prints `acceptance_len` and `by_class` in the README index when present. Older headlines without
them stay valid; reviewers ask for both on the next headline change.


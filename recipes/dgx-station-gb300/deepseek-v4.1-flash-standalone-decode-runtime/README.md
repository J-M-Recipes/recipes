# DeepSeek-V4.1-Flash standalone decode runtime - one DGX Station GB300

Status: **experimental**, first recipe for this runtime, 2026-11-08. A research decode runtime, not an API server:
one Python process, one sequence at a time, greedy only, no vLLM in the process. Public repo:
[jmeadlock/gb300-dsv41-decode @ ed6582e](https://github.com/jmeadlock/gb300-dsv41-decode/commit/ed6582e051b9b743f1a68b88750c627d2a63ca03)
(Apache-2.0). Blog: <https://al-engr.com/gb300-dsv41-decode.html>. Card: `card.png`.

The question the project asks is how short one GB300's single-user decode step can get with stock weights, all 384
experts and no KV quantization. This page is where the answer stands on Sunday 2026-11-08, with the numbers it rests on
and what they do not show. Receipts win over this prose: every number below is in `results/2026-11-08-v1-recipe/raw/`
and indexed in `results/2026-11-08-v1-recipe/throughput.csv`.

## At a glance

Both columns are the same instrument (CUDA events around the verify and draft graphs, steps >= 32 of each sequence, 8
fixture prompts x 256 new tokens, greedy, DSpark k=5 = 6 rows per step). Name the column whenever you quote one.

| | headline (cold) | steady state (warmed) | source |
|---|---|---|---|
| ms per 6-row step | **18.62** | 17.92 | `raw/headline-20261108-main-ss.json` |
| ms per emitted token (GPU time) | **6.41** (156 tok/s) | 6.17 (162 tok/s) | same |
| tokens per step | 2.903 | 2.903 | same |
| cold-expert share of routed calls | 4.83 % | 1.55 % (after 2,686 manager steps) | same |

| other single-stream numbers | value | source |
|---|---|---|
| decode wall ms/token, all steps, incl. end-of-sequence overrun | 6.17 (excl. overrun 6.11, GPU 6.09, host gap 0.02) | `raw/headline-20261108-main-ov2.json` |
| time to first token, 31-54-token prompts, grouped prefill | 0.95 s (reference prefill was 4.35 s / TTFT 3.9 s) | `raw/prefill-bench-m4p2-20261108.json` |
| 1-row step (no speculation) | 9.76 ms/step, 102.4 tok/s | `raw/fl-s3-beside-20261108.log` |
| HBM free after build | 2.72 GiB of 249.8 | headline receipt |
| **confirmation run at the shipped tree** (code = ed6582e; docs-only commit 8f24466), one receipt with the poll fix and grouped prefill together | 18.58 ms/step, 6.40 ms/token headline; steady 17.93 / 6.17; decode wall incl. overrun 6.17, GPU 6.09, host gap 0.02; overrun 8 steps; **TTFT 820 ms** mean in-run (the 943 ms above was measured under the kineto profiler); tokens equal 8/8; streams 8/8; self-replay 8/8 | `raw/headline-20261108-v1-recipe.json` |

**Against the production lane on the same 8 prompts - two framings, not a ratio.** Production on this box (the sibling
sidecar recipe, "Pg61") ran 7.66 ms per engine step and 2.55 ms per token, wall-clock, with a first chunk 20-34 ms
after the request; but 129 of its 384 experts per layer are computed on a second GPU (the RTX PRO 6000 sidecar), so it
is a two-GPU number. This runtime is one GPU, and its 18.62 / 6.41 are GPU time. The like-for-like wall-clock figure
here is 6.17 ms/token, about 2.4x slower per token than Pg61, with a first token about 30x later. The repo's own bar
(under 7.36 ms per 6-row step, restated 2026-11-07) is **not met**. See `## Known limits`.

Per fixture prompt (headline receipt; `ms/step` is the >= 32-step window, absent where the answer ended before step 32):

| prompt | prompt tokens | emitted | finish | tokens/step | steps | ms/step | grouped TTFT (4 streams) | reference prefill |
|---|---|---|---|---|---|---|---|---|
| prose | 42 | 256 | length | 2.49 | 103 | 17.99 | 986 ms | 3,828 ms |
| code | 47 | 218 | stop | 4.19 | 52 | 22.52 | 1,012 ms | 4,600 ms |
| math | 54 | 256 | length | 4.40 | 58 | 18.98 | 984 ms | 4,685 ms |
| json | 38 | 18 | stop | 4.50 | 4 | - | 909 ms | 4,294 ms |
| shell | 34 | 256 | length | 3.67 | 70 | 18.44 | 872 ms | 3,953 ms |
| tool | 49 | 21 | stop | 5.00 | 4 | - | 1,010 ms | 4,870 ms |
| long | 46 | 256 | length | 2.63 | 97 | 18.06 | 966 ms | 4,593 ms |
| story | 31 | 192 | stop | 2.11 | 91 | 18.65 | 856 ms | 3,940 ms |

## What this runs

- **Model.** `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, as shipped (MXFP4
  routed experts, FP8 attention and Engram, nothing re-quantized). 475.24 GiB, 96,085 tensors. Weight *layout* may
  change in memory; values are bit-identical to the checkpoint.
- **Process.** One Python process in a `vllm/vllm-openai:v0.31.0` container used only for its CUDA / torch / TileLang /
  DeepGEMM / FlashMLA / Triton stack. vLLM is never imported. `--network none`; the container runs the headline
  instrument once and exits.
- **Prefill.** DeepSeek's reference `model.py` (MIT, unchanged) with its reference `fp4_gemm` kernel, one launch per
  expert, but the surrounding work grouped into GEMM launches on 4 CUDA streams with torch's uninitialized-memory fill
  off. ~89,900 expert launches per prefill become ~10,400 GEMM launches. Equal to the reference prefill on first token,
  logits, `main_hidden`, every cache and replay, 8/8 prompts, 3 stream counts.
- **Decode.** The runtime's own CUDA-graph step, two graphs (verify | draft), DSpark k=5 from the checkpoint's `mtp`
  head, greedy exact-match acceptance. Kernels: DeepGEMM fp8xfp4 grouped GEMM over the expert pool in place (16-row
  contiguous layout), Triton MXFP8 split-K GEMV for the dense projections, fused glue clusters, FlashMLA sparse prefill
  op on the bf16 caches, fixed-order fp32 for the head and the mHC mixes.
- **Memory.** 301 hot experts per layer in HBM (the sibling recipe's seeded v26ar map plus the 6 next-most-routed per
  layer), 83 cold experts per layer in pinned Grace memory read in place over NVLink-C2C (UVA) by a staging kernel that
  copies the step's cold experts into 16 spare HBM slots per layer on a forked stream. A residency manager promotes and
  demotes experts between steps (policy moves experts, not numerics). Engram tables (188.8 GiB) live in pinned host
  memory. Receipt: 207.6 GiB read into HBM, 259.4 GiB pinned, 2.72 GiB HBM free after build.
- **Not here:** an HTTP server, a tokenizer front end, sampling, batching, a scheduler, long context.

## Hardware

One NVIDIA DGX Station GB300 (`hardware/dgx-station-gb300.yaml`): GB300 with 249.8 GiB of HBM visible to CUDA, Grace
LPDDR5X (448.8 GiB MemAvailable before the runtime's host allocation), NVLink-C2C, driver 595.99.02, CUDA 13.0 in the
container. The RTX PRO 6000 in the chassis is not used. This round's receipts record the GPU, driver, CUDA, torch,
Triton and clocks but not the OS or kernel, so `results/2026-11-08-v1-recipe/system.json` is assembled from the receipts
and says so; it is not `collect_system_snapshot.sh` output.

Needed from the host: the checkpoint on local NVMe mounted read-only; `--ulimit memlock=-1:-1`, `--cap-add IPC_LOCK` and
`--ipc=host` (the Grace tier is `cudaHostRegister`'d); the GB300 passed by CDI name (index 0 on this box is a display
GPU); no other GPU job. The boot refuses to start unless MemAvailable covers the pinned allocation plus a 16 GiB reserve.

## Software

- Image `vllm/vllm-openai:v0.31.0`, digest `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`
  (taken from the sibling Kimi-K3 recipe, which records it for this tag; the decode receipts record library versions
  but not the image digest, and this write-up did not re-pull it).
- torch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0, Python 3.12.3 aarch64 (headline receipt); TileLang 0.1.12, CuTe DSL 4.7.1,
  FlashInfer 0.7.0.post1 (donor only), transformers 5.17.0 / tokenizers 0.23.2 (tokenizer only) from `raw/ENVIRONMENT.md`
  (verified 2026-11-06); DeepGEMM 2.8.0 wrappers from the image's vLLM tree (`raw/VENDORED.md`).
- Runtime: `jmeadlock/gb300-dsv41-decode` at `ed6582e051b9b743f1a68b88750c627d2a63ca03` (merge m4-p2), checkout mounted
  at `/work`. It is not vendored here. The three headline receipts carry `git_commit: null` (no git in the container),
  so the commit pin comes from the repo history, not from the receipts; see the tree-state limit below.
- Nothing is bind-mounted over the image's packages. Mounts are the repo checkout (`/work`), the checkpoint (`/ref`,
  read-only), a scratch/receipt directory (`/work/scratch`) and four JIT caches (`/root/.tilelang`, `.triton`, `.nv`,
  `.deep_gemm`).
- Hot map `docs/maps/hot-v26ar-seed-301.json` (sha256 `4b68d831...ac6`; copy in `raw/`).

## Launch

```
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170 \
WORK=$HOME/gb300-dsv41-decode GPU=<GB300 CDI name> \
bash scripts/launch-headline.sh [extra bench/headline.py args, e.g. --no-steady-state]
```

`scripts/launch-headline.sh` refuses a `WORK` that is not at `ed6582e`, a checkpoint whose `config.json` sha256 is not
`8be45ce0...1879` (the value in the receipt), or an image that does not report the pinned digest, unless you set
`ALLOW_OTHER_COMMIT=1`, `ALLOW_OTHER_CHECKPOINT=1` or `ALLOW_IMAGE_MISMATCH=1` (a receipt from an overridden run is not
this recipe's number). It then runs `docker run --rm --network none --device nvidia.com/gpu=$GPU ...` with
`bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla
--fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency-drain 64 --max-new 256`, which is the `argv` recorded
in all three headline receipts. A whole run takes 6-10 minutes with the steady-state pass (578 s, 564 s and 353 s in the
receipts); boot is ~61-63 s.

The launcher was written from the repo's box scripts (`bench/box/m4p2.sh`, `docs/A2-2-BOX-PLAN.md`). It passes `bash -n`
and a stub-`docker` render whose argv matches the receipt's; **no container has been started from it**. Cold JIT
compile and the first GEMV tune are not timed anywhere.

Notable defaults at the pinned commit are listed under `launch.notable_flags` in `recipe.yaml`: residency on, pool SF
layout `beside`, staging `kernel-overlap`, hot map 301, row-conditional small wins (R=1 only), grouped prefill, per-step
poll with tail sync.

## Verify

There is no server health check. A run is good when its JSON receipt shows:

- `errors == []`;
- `streams.exact == 8` of 8: the emitted 256-token stream equals the runtime's own 1-row greedy run, with every logits
  row bit-equal (e.g. prose 256/256 rows);
- `replay_ok == 8`: self-replay (fast reps vs parity reps) byte-identical, logits digests equal;
- `overflow.steps == 0`: no step needed more than 16 cold experts staged (see Known limits: an overflow produces wrong
  tokens);
- the per-prompt `tokens_sha256` equal to the ones in `raw/headline-20261108-main-ss.json`.

Gates behind the numbers, as the repo ran them (box commands in `bench/box/m4p2.sh`; **not re-run for this write-up**):
`tests/gpu/test_prefill_fast_gpu.py` 4/4 (`raw/m4p2-gpu-test-20261108.log`: 7 gates, 12 layers and 20 buffers equal, 144
tokens and 144 logit rows equal under reference and grouped prefill); `bench/prefill_bench.py` equality to the reference
prefill on 8/8 prompts at 1, 4 and 8 streams (`summary.all_equal: true`).

What is not verified: image digest re-read, a Hub-vs-local checkpoint hash comparison, a receipt that records the git
commit, any quality benchmark (tool calls, BFCL, GSM8K, needle), teacher-forced comparison against the reference
kernels, anything at temperature above 0, anything past 310 positions. `verification.gates` in `recipe.yaml` says the same.

## Results

Run id `2026-11-08-v1-recipe` (all receipts copied from the repo's `docs/`, see `results/2026-11-08-v1-recipe/README.md`).
Measured from Saturday 2026-11-07 afternoon to Sunday 2026-11-08 morning on the reference Station.

**The ladder** (headline cell, GPU ms per 6-row step and per token; each row is one headline run on that tree):

| rung | ms/step | ms/token | receipt |
|---|---|---|---|
| fused 6-row step bit-equal to the 1-row step (T9b-8) | 20.304 | 6.994 | `headline-20261107j.json` |
| + residency manager on | 19.931 | 6.865 | `headline-resid2-20261107.json` |
| + staging overlap (kernel-overlap, NPROG 48,16,16) | 19.342 | 6.662 | `headline-s1-shapes-20261107.json` |
| + partial SF pool, pool-spare 20 | 19.064 | 6.566 | `headline-s2-C-20261108.json` |
| + SF beside weights | 18.741 | 6.455 | `headline-s3-B-20261108.json` |
| + hot map 301 | 18.620 | 6.414 | `headline-hot301-20261108.json` |
| two-column headline run (this page's headline) | 18.624 | 6.415 | `headline-20261108-main-ss.json` |

The emitted 256-token streams (`tokens_sha256`) of all 13 receipts from the first row through the declined arms below
are equal to the final headline's, 8/8 prompts. Earlier rungs (405 ms at first light, then DeepGEMM, GEMV, FlashMLA,
fixed-order fp32) changed the streams and are A3 class; they are in `raw/LEDGER.md`.

**Wall-clock columns** (`raw/headline-20261108-main-ov2.json`, per-step poll): decode wall 6.166 ms/token including 8
overrun steps (0.057 ms/token), 6.109 excluding, GPU 6.086, host gap 0.024. The old every-8-steps poll issued 97 no-op
steps over the fixture (1.8 s). Steady-state pass: wall 5.926 incl. overrun. TTFT in that run (reference prefill) 3,877 ms.

**Prefill** (`raw/prefill-bench-m4p2-20261108.json`): reference 4,345 ms mean; grouped 993 / 943 / 963 ms at 1 / 4 / 8
streams (TTFT 999 / 949 / 970 ms, TTFT = prefill + the first draft, 2.9 ms). Inside the headline run
(`raw/headline-m4p2-20261108.json`) the same grouped prefill measured 787 ms mean; the receipts do not explain the
difference (the benchmark run used `--kineto`). The slowest remaining piece is ~10.4k per-expert launches at ~90 us each;
`raw/prefill-anatomy-20261108.log` has the 3,312-3,672 expert calls per prefill and 19 % GPU busy for the reference path.

**Distance to production** (`raw/m3n1-downstream-20261107.json`, base arm, measured on the 11/07 night tree): teacher-forced
on the production lane's 1,476 greedy tokens, 55 top-1 flips (margin-weighted 28.125, max |dlogprob| 2.685). Free-run
on the fixture: 2 of 8 streams identical to production (json, tool); the others diverge at positions 1-149 near ties
(`raw/fl-s3-beside-20261108.log`). Production versus its own rerun is 7/8 identical (`raw/LEDGER.md`, 11/06 11:27).

**Declined or null rungs** (headline ms/step against the 20.304 tree unless stated; receipts in `raw/`):

| arm | result | verdict |
|---|---|---|
| PDL programmatic edges | 20.297; with L2 prefetch 20.626 | 0 gain / +0.32; GPU test failed on caches (hazard) |
| mHC mixes fork at 6 rows | 20.645 | +0.34 |
| fused q/kv tail at 6 rows | 21.803; after a fix 20.452 | +1.5 / +0.15, parked |
| GEMV v3 at 1 row | 10.13 vs 9.97 ms | +0.16, stream differs (A3) |
| DeepGEMM MegaMoE | faster per layer, but 4 of 40 layers added 49-58 flips vs production | declined for numerics (`raw/M2-L3-NUMERICS.md`) |
| more residency slots (pool 24 / 28) | 18.766 / 18.740 vs 18.741 | null |
| 8 spare slots + pool 16 | 18.924, overflow 13 steps, streams 7/8, self-replay 6/8 | INVALID |
| hot map 304 | 18.563, 1.71 GiB HBM free | opt-in only |

## Known limits

- **Single stream, greedy, no server.** One sequence, temperature 0, exact-match acceptance. No API, batching,
  scheduler, sampling, probabilistic drafting. No concurrency number exists to quote.
- **Context.** Every receipt uses 31-54-token prompts plus 256 new tokens: at most 310 positions. The loop is capped at
  `max_pos` 4,095. Nothing from 311 to 4,095 was measured, nothing longer exists, and prefill was never timed on a long
  prompt. A 32K arm is proposed (`raw/SUNDAY-20261108.md`), not run. Any "4k" claim would be about the cap, not a measurement.
- **Fixture-bound.** Eight prompts; the json (18 tokens) and tool (21 tokens) answers end in 4 steps and never enter the
  >= 32-step window, so the 2.90 tokens/step is carried by six prompts (2.11 story to 4.40 math per prompt). The hot map
  was fitted to router counts from the sibling lane's October campaign, so the fixture's cold share is out-of-sample.
- **Bit-identity is narrower than it sounds.** Shown by receipts: prefill == reference prefill (8/8); emitted streams of
  13 receipts from the 20.30 ms tree onward are byte-equal to the final's (8/8); the 6-row verify step == the 1-row step
  (8/8); self-replay 8/8. Not shown, and not true: that the decode step equals DeepSeek's reference kernels. DeepGEMM
  replaces `fp4_gemm` (drift 0.0004-0.057 nats against the reference path), FlashMLA replaces the reference `sparse_attn`
  (0.0006-0.016 nats against the bit-identical path), the Triton GEMV replaces the reference fp8 GEMM, fixed-order fp32
  replaces cuBLAS for the head and mixes. The repo calls this A3 class: parity by flip count. A summary line such as "one
  GPU, bit-identical" (`raw/SUNDAY-20261108.md`) holds for the prefill and for the stream-to-stream chain above, not for
  the decode kernels against DeepSeek's reference; "reference `model.py` unchanged" is true of the vendored file only.
- **No quality gate.** No tool-call parse, BFCL, GSM8K, needle, or teacher-forced bracket against the reference kernels
  has been run on this runtime. The only fidelity numbers are the production-distance ones above.
- **Three receipts, three tree states.** Headline cold/steady (18.62 / 17.92) is from main at 07:04 CST with the old poll
  schedule and reference prefill; the per-step-poll numbers from main at 09:14 CST (reference prefill); the grouped-prefill
  headline from the `m4-p2` branch worktree at 09:45 CST (old poll). `ed6582e` (09:57) merges both. GPU columns agree
  (18.62 / 18.58 / 18.62 ms) and the streams are equal, but TTFT 0.95 s with the per-step poll was never measured in one
  run. A first measurement on `ed6582e` through `scripts/launch-headline.sh` is the obvious next receipt.
- **Overflow produces wrong tokens, and no receipt shows a stop.** The runtime counts cold experts that exceed the 16 spare
  slots and the receipt reports `overflow.steps`; the arm with 8 slots overflowed 13 steps and produced wrong streams
  (7/8). Treat `overflow.steps != 0` as an invalid run.
- **Thin HBM margin.** 2.72 GiB free after build; the 304-hot map left 1.71 GiB, under the 2 GiB staging headroom. About
  260 GiB of Grace memory is pinned for the life of the process.
- **GEMV tuning is part of the numerics.** The config choice is timed on the box and persisted in `DSV41_GEMV_CACHE`;
  configs with a different split-K partition can differ by an ulp. The receipts record the cache path, not its hash.
- **Reproducibility gaps.** Image digest borrowed from a sibling recipe; no Hub-vs-local checkpoint hash comparison in
  these receipts (the repo's 11/06 T2 boot verified 93,338/93,338 loaded tensors); `launch-headline.sh` and
  `rollback.sh` untested end to end; OS and kernel not recorded.
- **Comparison caveats.** Pg61 is two GPUs and wall-clock; this runtime is one GPU and GPU time. The `bar_ms_per_step: 5.0`
  inside the receipts is the original bar; James restated it to 7.36 on 2026-11-07 (`raw/PROJECT.md`).
- **The Station is the project's dev box,** not a serving lane. Production DSV4.1 has run on Sparks 1-4 since 2026-11-06.

## Rollback

Nothing here replaced a serving lane, so there is nothing to restore. `bash scripts/rollback.sh [--dry-run]` stops the
headline container (it is `--rm`) and prints how to bring the former Station lane back: the sibling sidecar recipe's
`scripts/launch-pg61.sh up` and the repo's `CUTOVER.md` rollback section (`raw/CUTOVER.md`), on James's say-so only.
`--dry-run` and `bash -n` were run; the live path was not.

## Credits

Mechanisms this recipe runs:

- **DeepSeek** - the checkpoint, the DSpark drafter (in-checkpoint `mtp` head), the reference `inference/` code
  (`model.py`, `kernel.py`, `engram.py`; MIT) used unchanged for the prefill and as the oracle, and the reference TileLang
  kernels (`fp4_gemm`, `act_quant`, `sparse_attn`) the prefill calls. **DeepGEMM** (grouped fp8xfp4 GEMM, 2.8.0
  wrappers shipped in the vLLM image), **FlashMLA**, **TileLang** 0.1.12 and **Triton** 3.7.1.
- **vLLM** v0.31.0 - only as the library image; **FlashInfer** as a kernel donor.
- **J&M sibling recipe** `deepseek-v4.1-flash-vllm-uva-dspark` - the seeded v26ar hot map (295) and the router counts the
  301 map was extended from; model-card parameter counts carried from its card.
- Runtime code, the benchmark instruments and the ladder are ours (`jmeadlock/gb300-dsv41-decode`, Apache-2.0, authors
  James Meadlock and Milo; the model behind each commit is named in the commit or document byline). Astra (Nous, openai/gpt-6-astra)
  reviewed the plan on 2026-11-07 and shaped the gap-ledger order and the A3 definition (`raw/LEDGER.md`, 18:05 entry).

Measured against (a lane we compare with, not something this recipe runs):

- **Pg61**, the production lane on this box until 2026-11-06: **Garner McCloud's** engine
  ([garnermccloud/dgx-station-recipes @ 8043a6811a5f](https://github.com/garnermccloud/dgx-station-recipes/tree/8043a6811a5feda70fdfb6e827eebb2ec25c65d1))
  on `vllm/vllm-openai:v0.31.0` plus our patches, with the cold experts computed on the RTX PRO 6000 by the peer-tier sidecar
  (sibling recipe `deepseek-v4.1-flash-vllm-uva-dspark-sidecar`). Its 7.36 ms/step is the project's baseline and bar.

Read, filed as cards, and **not adopted** (none of it is in the recipe yet): the **sgl-nvidia-mega** SGLang
low-latency plugin (earlier PDL releases; `PROJECT.md` lists it as "reference source, not a dependency"), and DeepGEMM
**MegaMoE**, which we spiked and declined on numerics (`raw/M2-L3-NUMERICS.md`).

## Finalisation checklist

- [ ] One run of `scripts/launch-headline.sh` at `ed6582e` that records the commit (git in the container, or
      `RECEIPT_GIT_COMMIT` in the environment) and covers per-step poll + grouped prefill together.
- [ ] Image digest and checkpoint hash re-read on the box; `system.json` replaced by `collect_system_snapshot.sh` output.
- [ ] A quality gate: teacher-forced bracket against the reference kernels on the same weights, tool-call parse, one public
      bench we ran ourselves.
- [ ] A long-context arm (32K prefix first) and prefill TTFT on a long prompt.
- [ ] Decide what overflow does: fail-stop or a larger spare budget.
- [ ] Pin the GEMV tune cache (hash) or show a fresh tune lands on the same bits.

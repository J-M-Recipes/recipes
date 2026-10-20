# DeepSeek-V4.1-Flash-1017 on 4× DGX Spark — vLLM fork (spark-ds41f r6x) TP4 over sparknet RoCE, 1M context

**Status: experimental** (2026-10-20). Lane C of the split GhettoRak rack: Sparks 5–8, port 8002, serving since the
FINALr2 boot (2026-10-20 02:10 UTC). Native 1017 weights, nothing re-quantized. Christopher Owen's r6x config as shipped,
on our switch / 4 KiB-page shape. A nine-arm Round-1 ladder and a Round-2 ladder found no knob worth +2%, so this
publishes the authors' config with the ladder as evidence that the obvious knobs are flat. Served teacher-forced
quality gate: pending.

Sibling: [`dgx-spark-4x/deepseek-v4.1-flash-sglang-tp4-roce`](../deepseek-v4.1-flash-sglang-tp4-roce/) (Lane B: SGLang,
knapcio v2.4, Sparks 1–4, same switch, same instrument). Same model on one GB300:
[`dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark`](../../dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/).
Blog: [DeepSeek Flash 1017: TensorFold vs SGLang vs vLLM on 4 DGX Sparks](https://al-engr.com/dsv41-1017-three-way-tensorfold-sglang-vllm-4x-dgx-spark.html) (Round 2 section).

| like for like (same v41bench v1 / prefill_reps / long_check / mixed_probe) | **C — vLLM fork (this recipe)** | B — SGLang sibling |
|---|---:|---:|
| C1 per-stream tok/s, 8-category mix, mean of 2 runs | **104.3** (104.2 / 104.3) | 95.6 (95.0 / 96.2) |
| C1 aggregate tok/s, mean of 2 | **93.0** | 81.2 (80.0 / 82.4) |
| C4 / C6 aggregate tok/s | **230.9 / 288.9** | 203.8 / 252.8 (B arms: C6 250–262) |
| 93K cold prefill, quiet control 2026-10-20 (3-rep median) | 5,993 | **6,219** — level (B +4%) |
| 131K needle prefill | **5,724** | 4,427 |
| 300K needles ×3 (tok/s) | **5,462 / 5,471 / 5,453** (no first-touch dip) | 3,484 / 4,029 / 4,741 (run 1 low) |
| Short-chat TTFT behind a cold 93K prefill | 3.03 s | **1.7–1.8 s** |
| KV pool | **6,537,516 tokens, fixed** | 4.5–5.9M, varies by boot |
| Warm boot to HTTP ready | **89–101 s** | ~6.5 min |
| Context / qeval content score | 1M / 71 of 75 | 1M / 71 of 75 |

C wins decode at every concurrency and long-context prefill. B wins mixed traffic (a short request behind a long
prefill). 93K prefill is level. The B column comes from the sibling's B1a receipts (`spark1:~/bench-r2/B1a/`); that
recipe owns those numbers. The C column is C0ctl2 from this bundle.

## What this runs

`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` as shipped: routed experts in MXFP4, attention / dense / Engram in
FP8, and the in-checkpoint DSpark draft. Tensor parallel 4, one rank per Spark. DSpark k=5 (greedy draft, block
rejection, adaptive verification at row cost 0.035). TileLang attention / linear / MoE kernels. Engram tables on disk.
sparknet one-shot RoCE all-reduce / all-gather handles messages ≤ 2 MiB and NCCL handles the rest. 1,048,576-token
context, 16 sequences, chunked prefill 8192, one prefill at a time, prefix caching, thinking off by default.

The stack is **not stock vLLM**: stock vLLM cannot serve V4.1 on GB10 (SM121) today. It is Owen's
[spark-ds41f](https://github.com/christopherowen/spark-ds41f) at the r6x promotion commit `fc3ee36d659e` ("Promote r6x
for TP4 (#407)"). That repo builds one image from the Local Inference Lab vLLM fork with its patch chain, TileLang,
TileKernels, a patched NCCL and sparknet. We built it locally on each node.

Our shape differs from Owen's. He qualifies r6x on a switchless four-node ring (kernel 7.0, 64 KiB pages,
memory-saver). Ours is a CRS804 switch, kernel 6.17, 4 KiB pages and no memory-saver. That is the git-ignored local shape
`tp4-switch-4k` (`results/2026-10-19-vllm-ladder-r2/raw/ds41f-local-config/`), which changes:

- sparknet `oneshot-ring4` → `oneshot-direct` (`SPARKNET_ROCE_TOPOLOGY=direct`)
- `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`, `NCCL_IB_MERGE_NICS=0`, `NCCL_CROSS_NIC=0`
- socket ifname `enp1s0f0np0`
- KV 8 GiB/rank (Owen's ring: 10.5 GiB)

The 4 NCCL channels, 4 MiB buffers and GID 3 are inherited from his `tp4-ring`. Owen's topology checker refuses
`oneshot-direct` on four nodes, so the rendered `docker run` is launched directly.

## Hardware

See `hardware/dgx-spark-4x.yaml`. Four DGX Spark GB10 (Sparks 5–8, the second pod on the shared CRS804), two 200G RoCE
rails per node (10.0.0.N and 10.0.1.N, MTU 9000, lossy RoCEv2), driver 595.84, CUDA 13.2, kernel 6.17.0-1031-nvidia,
4 KiB pages. GPU clocks are capped at 2200 MHz (2190 measured under load on all four). Per-node snapshots:
`results/2026-10-19-vllm-ladder-r2/system/spark{5..8}.json` (the head's is copied to `system.json`).

Memory per rank: weights + draft 72.19 GiB, KV 8.0 GiB (fixed by `--kv-cache-memory-bytes`), CUDA graphs 1.98 GiB.
The container is capped at `--memory=112g` and pinned to `--cpuset-cpus=5-9,15-19`. Head host MemAvailable floor during
the C0ctl2 battery: 24.0 GiB.

## Software

- Image `vllm-ds41f-kkref:19f2c20ed4d6-r6x`, Id `sha256:c1c93733e852…`, identical on all four nodes. It is a local build
  (~2h16m per node; command in `raw/build/ds41f-buildcmd.sh`) and is not pullable. Base
  `vllm/vllm-openai@sha256:3cf14c52…` (nightly `af1c01499b28`); the engine reports `0.29.1rc1.dev187+gaf1c01499`.
- Dockerfile: spark-ds41f `docker/Dockerfile` at `fc3ee36d`, sha256 `b4860112914…` (not vendored).
- Six pinned trees, matching the config's `expected_labels` (image labels `local.spark3.*`; full list in
  `results/…/image.txt`):
  - vLLM: `local-inference-lab/vllm` karmic-kraken-beta @ `19f2c20ed4d6` (tree `4d6851c8`)
  - b12x @ `236ddff0`
  - NCCL v2.30.7-1 + patches
  - TileLang 0.1.15 @ `b95ee4ff`
  - TileKernels 2.0.0 @ `66258df6`
  - **sparknet 0.3.0 @ `3bc8d10` (3 commits past the 0.3.0 release; upstream is now 0.4.0, not in this image)**
- Weights: 48 shards, 510,296,708,312 bytes on every node. spark5's shard sha256 values equal the Hub LFS etags, and
  the same bytes copied to spark1 passed `sha256sum -c` 48/48 (`raw/weights/`). Sparks 6–8 are size-verified only.

## Launch

```bash
export SPARK_HOSTS="spark5 spark6 spark7 spark8"   # rank order; rank 0 = head; must resolve on the nodes too
# each node: image vllm-ds41f-kkref:19f2c20ed4d6-r6x built from spark-ds41f @ fc3ee36d,
#            weights in ~/models/DeepSeek-V4.1-Flash-1017, a clone of spark-ds41f in ~/spark-ds41f (its cache/ = JIT/AOT caches)
bash scripts/cluster.sh push
bash scripts/cluster.sh up      # drop caches + compact on all four, MemAvailable ≥ 100 GiB gate, ranks 3,2,1 then 0, wait for /v1/models
```

`scripts/launch-vllm-tp4.sh <rank>` is **generated** by `scripts/render_launcher.py` from the live head command of the
measured boot. The transform is kept in the repo. It assert-replaces the name, host IP, per-rank peer HCAs, model and
cache mounts, image, rank and master address, and copies everything else byte-for-byte. Rendered for rank 1 with a
stubbed `docker`, its arguments match the live spark6 container's `docker inspect .Args` exactly. Its env is a subset of
spark6's; the remaining variables are image defaults. Overrides: `MODEL_DIR`, `DS41F_DIR`, `MASTER_ADDR` (default
`10.0.0.5`), `VLLM_HOST_IP`, `IMAGE`, `NAME`. API: `http://<head>:8002/v1`, served model
`deepseek-v4.1-flash-spark-vllm`, no key.

Boot (C0ctl2, warm): weights 20.6 s, model load 29.6 s, engine init 22.4 s; HTTP ready 89 s after `docker run` on the
head. The first boot on an empty cache took 351 s (Round 1).

## Verify

```bash
export BASE_URL=http://<head>:8002/v1
python3 scripts/gates.py out/                          # /v1/models, exact-reply smoke, get_weather tool round, preemptions
bash scripts/run_battery.sh out/ mylabel full mixed greedy   # gates, qeval, needles, C1..C6, prefill reps, acceptance, mixed, greedy probe
```

The Python tools are the campaign's own files (`spark5:~/bench/laneC/tools`, `~/ladder`). They are identical to the
SGLang sibling's except for the lane string in `gates.py`. The only edits:

- endpoints come from `BASE_URL`
- the served model comes from `MODEL`
- the encoding dir comes from `DSV41_ENCODING_DIR`
- the box paths are removed

`v41bench_tf.py` needs DeepSeek's `encoding/` directory from the checkpoint, because the 1017 checkpoint ships no chat
template.

Every R2 boot passed the functional gates, including the serving boot FINALr2: `FUNCTIONAL_GATES PASS` (models,
`LANE_C_OK` smoke, `get_weather` call + tool-result round, 0 preemptions).

Identity across arms:

- qeval content matches C0 on all 75 tasks (71 of 75 correct).
- The 40-prompt greedy raw-completion probe matches C0ctl2 on 40 of 40 on every arm.
- The bench C1 per-request token/char signature matches on 9 of 9.

These checks show the arms did not change outputs relative to each other. They do not measure fidelity to the
original model.

**Quality gate: pending.** We have not run teacher-forced Δlogprob against the original on this lane, and we have not
run a public benchmark.

## Results

Run `results/2026-10-19-vllm-ladder-r2/` contains:

- `RESULTS.md`, the box-side Round-2 table
- `throughput.csv`, every cell of every arm
- `image.txt`, `launch-command.txt`, `system.json` + `system/`
- `raw/<ARM>/` for FINAL (Round-1 full battery), C0ctl2, C7ced, C8pp2, C8pp2b (not run), C9nvfp4, FINALr2 (serving
  boot) and FINAL-live
- `raw/contention/`

Instrument: prompt set v1, T=0, thinking off. C1 is per-stream (mean of 2 runs); C4/C6 are aggregate. 93K is the 3-rep
cold median. Mixed is a short chat's TTFT behind a cold 93K prefill. Round 2 ran 2026-10-20 01:14–02:12 UTC with one
change per boot, every boot warm, against the same-window control C0ctl2.

| arm | change | C1 per-stream | C4 | C6 | 93K prefill | 300K needle | mixed TTFT s | acceptance | outputs vs C0ctl2 | verdict |
|---|---|---:|---:|---:|---:|---|---:|---:|---|---|
| **C0ctl2** | none (shipped config, warm reboot) | **104.27** | **230.94** | **288.94** | **5,998** | 3/3 (5,462 / 5,471 / 5,453) | 3.026 | 0.6119 | reference | control |
| C7ced | Owen's CED non-final-row skip overlay | 104.19 (−0.1%) | 231.21 | 287.30 (−0.6%) | 6,055 (+0.9%) | 3/3 (+1.6%) | 2.872 | 0.6119 | identical | REVERT |
| C8pp2 | `--max-parallel-prefills 2` | 104.46 | 211.68 (−8.3%) | 257.77 (−10.8%) | 6,023 | 1/1 (5,449) | 2.962 | 0.6119 | identical | REVERT |
| C8pp2b | + `--long-prefill-token-threshold 6048` | — | — | — | — | — | — | — | — | not run (C8pp2 TTFT did not improve) |
| C9nvfp4 | NVFP4 draft head + Markov | 106.00 (+1.7%) | 233.18 | 280.77 (−2.8%) | 6,050 | 1/1 (5,428) | 3.002 | 0.6103 | identical | REVERT |
| FINALr2 | restore shipped config | — | — | — | — | — | — | — | greedy 40/40 = C0ctl2 | **serving** |
| FINAL (Round 1, 10-19 07:38 UTC) | shipped config | 103.50 | 229.54 | 286.61 | 5,977 | 3/3 (5,413 / 5,411 / 5,339) | 2.962 | 0.6119 | — | prior round |

C0ctl2 per category at C1 (tok/s per stream, acceptance): format 149.0 (0.91), coding 133.2 (0.83), math 126.8 (0.70),
json 104.5 (0.56), reasoning 101.3 (0.54), prose 78.9 (0.41), summary 78.8 (0.39), narrative 62.0 (0.31). Speed tracks
draft acceptance.

Other C0ctl2 results:

- Cold prefill by length: 2K 4,518, 8K 5,987, 32K 6,165, 64K 6,099, 93K 5,998 tok/s.
- Needles: 20K 5,822 and 131K 5,724, both PASS.
- KV pool 6,537,516 tokens on every boot.
- Max `clocks.gr` 2190 MHz on all four nodes.
- No new NV_ERR_NO_MEMORY in any arm.

Cross-lane contention A/B (2026-10-20 05:00–05:07 CDT, `raw/contention/`): C's 93K prefill went from 5,993 quiet to
6,009 while the SGLang pod prefilled back-to-back. B went from 6,219 to 6,242 the other way round. That is no effect:
two TP4 pods on one CRS804 can be benched the same day at single-stream long-prefill load.

## Known limits

- **Tuning found nothing.** No arm cleared +2% without a >2% regression:
  - The CED skip overlay gave +0.9–1.6% on prefill (+4.1% on the single 131K shot) with bit-identical outputs. Owen
    measured +10.6–11.1% for it on TP3, so the gain depends on the shape.
  - Two parallel prefills cost 8–11% of C4/C6.
  - The NVFP4 draft head traded C6 for C1.
- **Mixed traffic is B's.** A short chat behind a cold 93K prefill waits ~3.0 s here vs 1.7–1.9 s on the SGLang sibling.
- **1M is configured, 300K is the deepest needle.** It passes 3/3 at ~5.45K tok/s, with no first-touch dip on this
  lane. There is one needle per length, so this tests retrieval, not long-context reasoning.
- **Not stock, not pullable.** This is a local build of a fork on a pre-release vLLM nightly. Pin it by the deployment
  commit and the six tree labels; a rebuild will have a different image Id.
- **The shape is unqualified upstream.** `tp4-switch-4k` (switch, 6.17, 4 KiB pages, 8 GiB KV) is ours, not Owen's.
- **Hygiene as run:** caches were dropped and memory compacted before each boot. The R2 runner did not set THP to
  `never`; THP is `madvise` on Sparks 5–8.
- **Cold JIT:** the first boot on an empty cache took 351 s and read 7.7% lower prefill than the same config warm.
  Bench warm boots only.
- **Quality vs the original is unverified.** Teacher-forced Δlogprob and a public benchmark are both pending.
- The Round-1 arm receipts (C0ctl, C1rail, C2merge, C3*, C4*, C5*, C6*) stay on spark5. Their numbers are in the blog
  post and are not mirrored here.

## Rollback

`SPARK_HOSTS=… bash scripts/cluster.sh down` stops `dsv41-karmic-kraken` on all four nodes; the container is kept. The
other published DeepSeek-V4.1 lanes are:

- the SGLang sibling on 4× Spark: [`dgx-spark-4x/deepseek-v4.1-flash-sglang-tp4-roce`](../deepseek-v4.1-flash-sglang-tp4-roce/)
- the single-GB300 recipe: [`dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark`](../../dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/)

## Credits

- **Christopher Owen**: [spark-ds41f](https://github.com/christopherowen/spark-ds41f) (r6x config, patch chain, build,
  and the CED skip overlay tested here) and [sparknet](https://github.com/christopherowen/dgx-spark-networking)
  (one-shot RoCE collectives).
- **Local Inference Lab**: the `karmic-kraken-beta` vLLM fork and b12x.
- **TileLang** and **DeepSeek TileKernels**: the kernel family. **vLLM**: the base. **DeepSeek**: the model.
- **knapcio**: the four-Spark TP4 findings that Owen's `docs/inspiration.md` lists as hypotheses to reproduce. The
  SGLang sibling runs knapcio's stack; this lane does not use his Engram pack.

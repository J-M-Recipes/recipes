# DeepSeek-V4.1-Flash-1017 on 4× DGX Spark — vLLM fork (spark-ds41f r6x) TP4 over sparknet RoCE, 1M context

**Status: verified** (2026-10-23). Lane C of the split GhettoRak rack: Sparks 5–8, port 8002. Native 1017 weights,
nothing re-quantized. Christopher Owen's r6x config, on our switch / 4 KiB-page shape, plus four changes adopted
2026-10-22/23 that were each measured against a same-window control: **KV 10.5 GiB/rank** (+31% KV pool), **32 seats**,
**late admission at layer 20** (Owen's 0222 overlay, mounted) and **sparknet 0.5.0** (mounted over the image's 0.3.0).
Teacher-forced quality gate against the previous control: **PASS, |Δlogprob| = 0.0** over 5,485 forced positions.

Two ladders (nine arms in Round 1, four in Round 2) found no vendor knob worth +2%. The wins below came from the
authors' newer work plus our own shape, not from blind tuning, and one measured lever — halving the prefill chunk
threshold — was **declined** because it trades cold-prefill throughput for guest latency.

Sibling: [`dgx-spark-4x/deepseek-v4.1-flash-sglang-tp4-roce`](../deepseek-v4.1-flash-sglang-tp4-roce/) (Lane B: SGLang,
Sparks 1–4, same switch, same instrument). Same model on one GB300:
[`dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark`](../../dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/).
Blog: [DeepSeek Flash 1017: TensorFold vs SGLang vs vLLM on 4 DGX Sparks](https://al-engr.com/dsv41-1017-three-way-tensorfold-sglang-vllm-4x-dgx-spark.html).

| v41bench v1, T=0, thinking off, same instrument | **this boot 2026-10-23** | same lane 2026-10-20 | B — SGLang sibling 2026-10-20 |
|---|---:|---:|---:|
| C1 per-stream tok/s, 8-category mix | **105.21** | 104.27 | 95.6 |
| C1 aggregate tok/s | **93.89** | 92.99 | 81.2 |
| C4 / C6 aggregate tok/s | **233.16 / 291.75** | 230.94 / 288.94 | 203.8 / 252.8 |
| 93K cold prefill (median) | 5,338 (single rep) | 5,998 (3-rep) | **6,219** |
| Short turn behind a **cached 68K prefix** | **2.147 s** | ~3.14 s (8 GiB control) | not measured |
| Short turn behind a **cold 413K prefill** | **1.196 s** (L20 only) | 1.885 s | not measured |
| KV pool | **8,580,566 tokens** | 6,537,516 | 4.5–5.9M, varies by boot |
| Warm boot → HTTP ready | 101 s | 89–101 s | ~6.5 min |
| qeval content score | 71 / 75 | 71 / 75 | 71 / 75 |
| Teacher-forced vs control | **PASS, 0.0** | pending | n/a |

Read the prefill row carefully: 5,338 is **not** a regression from 5,998. The lane moved from 16 seats to 32 with the
chunk threshold halved (8096 → 3856), which is what costs cold-prefill throughput and what the guest shape wants. The
sibling's 6,219 came from a different boot on a different engine; it owns that number.

## What this runs

`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` as shipped: routed experts in MXFP4, attention / dense / Engram in
FP8, and the in-checkpoint DSpark draft. Tensor parallel 4, one rank per Spark. DSpark k=5 (greedy draft, block
rejection, adaptive verification at row cost 0.035). TileLang attention / linear / MoE kernels. Engram tables on disk.
sparknet one-shot RoCE all-reduce / all-gather handles messages ≤ 2 MiB and NCCL handles the rest. 1,048,576-token
context, **32 sequences**, chunked prefill 8192, one prefill at a time, prefix caching, thinking off by default.

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
- KV 10.5 GiB/rank, 32 seats, chunk threshold 3856

The 4 NCCL channels, 4 MiB buffers and GID 3 are inherited from his `tp4-ring`. Owen's topology checker refuses
`oneshot-direct` on four nodes, so the rendered `docker run` is launched directly.

## What changed on 2026-10-23, and what was refused

Every row is a separate boot measured against a same-window control on the same box. Full receipts in
`results/2026-10-23-adopted-l20-net/` and in the campaign's round-5 notes.

| change | effect | verdict |
|---|---|---|
| KV 8 → **10.5 GiB/rank** | pool 6,537,516 → **8,580,566** (+31.2%); host floor 20.70 GiB | **adopted** |
| **late admission @ layer 20** (Owen 0222) | guest turn behind a 413K cold prefill 1.885 → **1.196 s**; 93K prefill −0.28% (inside A/A spread); paired C1 flat | **adopted** |
| **sparknet 0.5.0** | paired decode **+0.81% C1 [0.44, 1.16]**, +1.16% C2, +1.32% C4; TTFT and prefill flat | **adopted** |
| chunk threshold 3856 → **1928** | warm-cache turn 2.20 → **1.50 s**, guest turn −41%, but 93K prefill **−12.9%** | **declined** (prefill cost) |
| `VLLM_L2_PREFETCH_GRID=2` | paired C1 −0.23% [−0.57, +0.10], 2/9 prompts faster | refused (null on switched TP4) |
| 37-file F1022 stack, layer 30 | decode +1.05% but C1 TTFT +27% and mixed turn 2.43 s | refused (latency trade) |

Two of these need care when quoted. Late admission's C8 TTFT read **0.463 s** on a single wave against 0.359–0.364 s on
three controls — not dismissed, not proven; only C1 paired results survive boot-to-boot comparison on this rig. And L20
+ sparknet together read C1 **+1.37%**, larger than the parts, which is encouraging rather than established, because
those are different boots.

## Hardware

See `hardware/dgx-spark-4x.yaml`. Four DGX Spark GB10 (Sparks 5–8, the second pod on the shared CRS804), two 200G RoCE
rails per node (10.0.0.N and 10.0.1.N, MTU 9000, lossy RoCEv2), driver 595.84, CUDA 13.2, kernel 6.17.0-1031-nvidia,
4 KiB pages. GPU clocks are capped at 2200 MHz (2190 measured under load on all four). Per-node snapshots:
`results/2026-10-19-vllm-ladder-r2/system/spark{5..8}.json` (the head's is copied to `system.json`).

Memory per rank: weights + draft 72.19 GiB, KV **10.5 GiB** (fixed by `--kv-cache-memory-bytes`), CUDA graphs 1.98 GiB.
The container is capped at `--memory=112g` and pinned to `--cpuset-cpus=5-9,15-19`. Head host MemAvailable floor during
the 2026-10-23 battery: 20.79 GiB.

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
  - sparknet 0.3.0 @ `3bc8d10` **vendored in the image** — the lane mounts **0.5.0** over it (below)
- Weights: 48 shards, 510,296,708,312 bytes on every node. spark5's shard sha256 values equal the Hub LFS etags, and
  the same bytes copied to spark1 passed `sha256sum -c` 48/48 (`raw/weights/`). Sparks 6–8 are size-verified only.

### The two mounted overlays (`overlays/`)

Neither needs a rebuild, both are read-only, and both are checksummed in `overlays/SHA256SUMS` (41 files):

- **`overlays/late-admission/vllm/`** — four files over `/opt/spark3/candidate/vllm/vllm/`. Owen's 0222 patch,
  unmodified: `git am` on our tree head `7316c7f9f` reproduces his commit `91eaaa59e1ba`. Enabled with
  `--env VLLM_LATE_ADMISSION_LAYER=20`. Unset means off; the model has **40** decoder layers and any value ≥ 40 fails at
  boot; `0` means "decide at layer 0", not "off".
- **`overlays/sparknet-0.5.0/sparknet/`** — 37 Python files over
  `/usr/local/lib/python3.12/dist-packages/sparknet`. Owen's `dgx-spark-networking` v0.5.0, tag `ed6af327`. The package
  is **not on PyPI** and never has been — the image vendors 0.3.0 with no pip metadata, so `pip install
  sparknet==0.5.0` fails by design. Mount the tree, or build the wheel the way his `docker/Dockerfile` does. All four
  ranks must switch together: a mixed set fails at collective setup.

`overlays/README.md` carries the provenance for both.

## Launch

```bash
export SPARK_HOSTS="spark5 spark6 spark7 spark8"   # rank order; rank 0 = head; must resolve on the nodes too
# each node: image vllm-ds41f-kkref:19f2c20ed4d6-r6x built from spark-ds41f @ fc3ee36d,
#            weights in ~/models/DeepSeek-V4.1-Flash-1017, a clone of spark-ds41f in ~/spark-ds41f (cache/ = JIT/AOT caches),
#            and this recipe checked out somewhere (OVERLAY_DIR defaults to $PWD/overlays)
bash scripts/cluster.sh push
bash scripts/cluster.sh up      # drop caches + compact on all four, MemAvailable >= 100 GiB gate, ranks 3,2,1 then 0, wait for /v1/models
```

`scripts/launch-vllm-tp4.sh <rank>` is **generated** by `scripts/render_launcher.py` from the live head command of the
2026-10-23 10:54 UTC boot. The transform is kept in the repo: it assert-replaces the name, host IP, per-rank peer HCAs,
model and cache mounts, the two overlay mounts, image, rank and master address, and copies everything else
byte-for-byte.

**Validated, not assumed:** `scripts/verify_launcher.py <rank> --host <node>` renders the launcher with a stub `docker`
and diffs it against the live container's `docker inspect` — 80/80 arguments, 103/103 env pairs, 8 mounts, checked on
ranks 1 and 3 on 2026-10-23. Rerun it after any config change; it is the cheapest guard against a recipe that has drifted
from the box.

Overrides: `MODEL_DIR`, `DS41F_DIR`, `OVERLAY_DIR`, `VLLM_TREE`, `MASTER_ADDR` (default `10.0.0.5`), `VLLM_HOST_IP`,
`IMAGE`, `NAME`. API: `http://<head>:8002/v1`, served model `deepseek-v4.1-flash-spark-vllm`, no key.

Boot (2026-10-23, warm): HTTP ready **101 s** after `docker run` on the head. The first boot on an empty cache took
351 s (Round 1).

## Verify

```bash
export BASE_URL=http://<head>:8002/v1
python3 scripts/gates.py out/                          # /v1/models, exact-reply smoke, get_weather tool round, preemptions
bash scripts/run_battery.sh out/ mylabel full mixed greedy   # gates, qeval, needles, C1..C6, prefill reps, acceptance, mixed, greedy probe
python3 scripts/verify_launcher.py 1 --host spark6     # launcher == the live container
```

The Python tools are the campaign's own files (`spark5:~/bench/laneC/tools`, `~/ladder`). They are identical to the
SGLang sibling's except for the lane string in `gates.py`. The only edits: endpoints come from `BASE_URL`, the served
model comes from `MODEL`, the encoding dir comes from `DSV41_ENCODING_DIR`, box paths removed. `v41bench_tf.py` needs
DeepSeek's `encoding/` directory from the checkpoint, because the 1017 checkpoint ships no chat template.

**Quality gate (2026-10-23).** Teacher-forced comparison against per-token temp-0 baselines from the kept 8 GiB control
container on the same box: 45 held-out segments, **5,485 forced positions**, p99 |Δlogprob| **0.0** (bar ≤ 0.05), max
**0.0** (bar ≤ 0.5), self-repeat exactly 0.0 on both sides, top-1 agreement 1.0, perplexity identical to the last digit.
That makes the config change lossless **relative to what this lane already served**. It does not make the lane match the
model card, and no public benchmark has been run by us. Teacher forcing also does not exercise T>0 sampling under a
different batch mix — the weaker evidence there is qeval 71/75 and a 40/40 byte-identical greedy probe.

Identity across arms: qeval content matches C0 on all 75 tasks (71 correct, same four misses); the 40-prompt greedy
raw-completion probe matches 40/40; bench C1 per-request token/char signature matches 9/9.

## Results

`results/2026-10-23-adopted-l20-net/` — the adopted boot (2026-10-23 10:54–11:10 UTC):

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| aggregate tok/s | 93.89 | 153.35 | 181.05 | 233.16 | 261.66 | 291.75 |
| per stream tok/s | 105.21 | 87.18 | 71.69 | 68.43 | 61.08 | 57.30 |
| mean TTFT s | 0.158 | 0.189 | 0.218 | 0.231 | 0.253 | 0.286 |

**Second boot of the same config the same morning** (11:55 UTC, after a deliberate stop/start cycle, same launch line):
C1 **94.34** aggregate, C4 **234.55**, C6 **290.15**, 93K prefill **5,372.0**, qeval 71, boot 101 s, KV pool 8,580,566,
MemAvailable floor 21.08 GiB, 0 preemptions. Every headline lands within ~0.6% of the table above, which is the
boot-to-boot repeatability this lane does have at C1. The 413K mixed probe reproduced too: short-turn TTFT
**0.901 / 1.352 / 0.935 / 1.262 s**, all far below the control's 1.885 s. Receipts: `boot2-*` in the same results dir.

- 93K cold prefill **5,338.1** tok/s (one rep, 32 seats / threshold 3856).
- Warm-cache guest turn (68K resident prefix + 1.5K suffix) **2.147 s** median of 5 (2.08 / 2.15 / 2.38 / 1.88 / 3.38).
- Short turn behind a 413K **cold** prefill: **1.196 s** with L20 (every sample 0.93–1.49 s) vs 1.885 s with no overlay
  (12 samples, lowest 1.55 s).
- KV pool **8,580,566** tokens = 8.18 full-length sequences at 1M (was 6.23).
- qeval 71/75. Greedy probe 40/40 identical to the 8 GiB control. 0 preemptions. DSpark acceptance 0.7151 at an average
  accepted length of 3.334 tokens per step. No new NV_ERR_NO_MEMORY in dmesg end-to-end (counts fell by 9 on every node,
  log rotation) and no Xid 13/43.
- Streaming short-chat ladder (a **different** instrument — 192-token chats, do not compare to the table above):
  68.27 / 144.35 / 214.30 / 289.02 / 342.84 / 380.39 aggregate at 1/4/8/16/24/32 streams; per stream 72.97 → 14.04;
  p50 TTFT 0.194 → 1.096 s; 0 errors.
- 300K needles 3/3 PASS **on the 2026-10-20 shape**; the adopted config was not re-needled in this round, so treat long-context retrieval as inherited evidence rather than re-verified. Battery today was gates, qeval, greedy probe, the streaming ladder and the 413K mixed probe.

Historical, `results/2026-10-19-vllm-ladder-r2/` — the Round-2 ladder at 16 seats / KV 8 GiB, one change per boot
against C0ctl2 (C1 104.27 per stream, 92.99 aggregate, 288.94 at C6, 5,998 93K prefill, 3.026 s mixed, acceptance
0.6119): C7ced +0.9–1.6% prefill with identical outputs (**revert**), C8pp2 −8.3/−10.8% at C4/C6 (**revert**), C9nvfp4
+1.7% C1 / −2.8% C6 (**revert**). That folder also holds `throughput.csv`, `image.txt`, `launch-command.txt`,
`system.json` and per-arm receipts.

## Known limits

- **Two ladders found nothing in the vendor knobs.** No Round-1 or Round-2 arm cleared +2% without a >2% regression.
  Everything in the "what changed" table came from Owen's newer work or from our own shape decision, and one of those
  (1928 chunks) was refused on its own cost.
- **Prefill is lower than the 2026-10-20 publication on purpose** — 32 seats and a 3856 chunk threshold, chosen for guest
  latency. If you want throughput instead, L20 without the seat change is the config to benchmark; the numbers here do
  not transfer.
- **C8 TTFT at layer 20 is unresolved** (0.463 s single wave vs 0.359–0.364 s on three controls). Not dismissed as noise.
  Only paired C1 results survive boot-to-boot comparison here; C4+ needs an interleaved harness.
- **1M is configured, 300K is the deepest needle** — and that run is from 2026-10-20, not from the adopted config. It
  passes 3/3 at ~5.45K tok/s with no first-touch dip. One needle per length: retrieval, not long-context reasoning.
- **Not stock, not pullable.** A local build of a fork on a pre-release vLLM nightly. Pin by the deployment commit and the
  six tree labels; a rebuild will have a different image Id.
- **The shape is unqualified upstream.** `tp4-switch-4k` is ours, not Owen's, and his topology checker refuses
  `oneshot-direct` on four nodes.
- **Overlays are copies, so they age.** If late admission or sparknet land upstream, these files are behind by
  construction. SHA256SUMS plus `overlays/README.md` pin what they came from.
- **Hygiene as run:** caches dropped and memory compacted before each boot; THP is `madvise` on Sparks 5–8, not `never`.
- **Cold JIT:** an empty cache boot took 351 s and read 7.7% lower prefill than the same config warm. All numbers above
  are warm boots.
- **Quality is proven against the control, not against the model card** — see Verify.

## Rollback

`SPARK_HOSTS=… bash scripts/cluster.sh down` stops `dsv41-karmic-kraken` on all four nodes; the container is kept. To
return to the pre-2026-10-23 configuration, unset `VLLM_LATE_ADMISSION_LAYER` and drop the two overlay mounts, or reboot
`--kv-cache-memory-bytes 8589934592` with `--long-prefill-token-threshold 8096 --max-num-seqs 16`; the 2026-10-20
numbers above were measured on exactly that shape. The other published DeepSeek-V4.1 lanes:

- the SGLang sibling on 4× Spark: [`dgx-spark-4x/deepseek-v4.1-flash-sglang-tp4-roce`](../deepseek-v4.1-flash-sglang-tp4-roce/)
- the single-GB300 recipe: [`dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark`](../../dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/)

## Credits

- **Christopher Owen**: [spark-ds41f](https://github.com/christopherowen/spark-ds41f) (r6x config, patch chain, build,
  the late-admission overlay adopted here, and the CED skip overlay tested in Round 2) and
  [sparknet](https://github.com/christopherowen/dgx-spark-networking) (one-shot RoCE collectives; 0.5.0 adopted here).
- **Local Inference Lab**: the `karmic-kraken-beta` vLLM fork and b12x.
- **TileLang** and **DeepSeek TileKernels**: the kernel family. **vLLM**: the base. **DeepSeek**: the model.
- **knapcio**: the four-Spark TP4 findings that Owen's `docs/inspiration.md` lists as hypotheses to reproduce. The
  SGLang sibling runs knapcio's stack; this lane does not use his Engram pack.

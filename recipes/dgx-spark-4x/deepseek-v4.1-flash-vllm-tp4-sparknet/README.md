# DeepSeek-V4.1-Flash-1017 on 4× DGX Spark — vLLM fork (spark-ds41f candidate 60) TP4 over sparknet RoCE, 1M context

**Status: verified** (2026-11-16). Lane C of the split GhettoRak rack: Sparks 5–8, port 8002. Native 1017 weights,
nothing re-quantized. **This lane runs Christopher Owen's engine** — [spark-ds41f](https://github.com/christopherowen/spark-ds41f)
**candidate 60**, built locally from his pinned sources, on our switch / 4 KiB-page shape, with two mounted overlays
(one his bug-fix lineage, one our re-port) and the prefill-yield setting his c60 profile adds:

- **`--prefill-yield short`** (the setting his c60 `tp4-switched` profile ships — the code default is `off` — with the
  ≥256K deep-context chunk step-down that comes with it)
- **32 seats**, `--max-num-batched-tokens 8192`, `--long-prefill-token-threshold 3856`, KV **10.5 GiB/rank**
- **late admission at layer 20** (in-tree in c60; `VLLM_LATE_ADMISSION_LAYER=20`)
- **`so-bitmask`** (Zee-B33's fix for spark-ds41f#1194) and **`dsml-c60`** (our re-port of the structural DSML rule)

Against a same-night r6x control on the same four nodes, the promoted config reads **C1 123.18 tok/s per stream
(+19.2%)**, **C16 545.67 aggregate (+15.7%)**, **cold prefill 8,554 / 8,726 / 8,554 / 7,849 tok/s at 32K / 131K / 300K /
1M (+56% … +104%)**, and it is **faster on the rows this shape exists for**: a short turn behind a cold 413K prefill
**0.704 s (from 1.012 s)** and a warm-cache guest turn **2.048 s (2.063 s)**. Every promotion gate passed; the arm
that turned prefill yield off (A2) lost exactly those mixed rows and was declined.

| v41bench v1, T=0, thinking off, same instrument | **this boot 2026-11-16** (candidate 60, A3) | same-night r6x control | delta |
|---|---:|---:|---:|
| C1 per stream, 8-category mix (mean of 3) | **123.18** | 103.33 | **+19.2%** |
| C1 aggregate tok/s | **105.12** | 92.06 | +14.2% |
| C16 aggregate tok/s (mean of 3) | **545.67** | 471.76 | **+15.7%** |
| cold 32K prefill, median of 3 | **8,554.5** | 5,481.0 | +56.1% |
| cold 131K prefill, median of 3 | **8,725.7** | 5,314.8 | +64.2% |
| cold 300K prefill, median of 3 | **8,554.0** | 4,956.0 | +72.6% |
| cold 1M prefill, 1 rep | **7,848.8** | 3,853.7 | +103.7% |
| short turn behind a **cold 413K prefill** | **0.704 s** | 1.012 s | **−30.4%** |
| short turn behind a **cached 68K prefix** | **2.048 s** | 2.063 s | −0.7% |
| agent turn, 200K cached context | **0.511 s** | 0.791 s | −35.4% |
| 1M needle (3 passphrases, 10/50/90% depth) | **3/3, 131.2 s** | 3/3, 263.3 s | 2.0× faster |
| qeval content score | 71 / 75 | 71 / 75 | same 4 misses |
| KV pool (engine startup line) | **12,034,343 tokens** (11.48× at 1M) | 8,580,566 (8.18×) | same 10.5 GiB/rank — see below |
| warm boot → HTTP ready | **135 s** | (container already up) | cold JIT: 1,392 s |

Read two things carefully. **Do not read the KV line as a measured 40% capacity gain**: both engines reserve the same
10.5 GiB per rank and the startup line moved between builds (`kv_cache_utils.py:2854` → `:2879`); whether the extra
tokens are real concurrent capacity or a change in how the line counts them was not tested — the 1M row is a
single-sequence needle, not eleven concurrent ones. And **only C1 is tight enough to read a
1% delta** (±0.2% within a boot); C16 moves ±2% boot-to-boot on this rig and `mixed_warm` has a slow tail, so the
mixed rows are reported as "not worse", not as a win beyond their spread.

## What this runs

`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` as shipped: routed experts in MXFP4, attention / dense / Engram in
FP8, and the in-checkpoint DSpark draft. Tensor parallel 4, one rank per Spark. DSpark k=5 (greedy draft, block
rejection, adaptive verification at row cost 0.035). TileLang attention / linear / MoE kernels. Engram tables on disk.
sparknet one-shot RoCE all-reduce / all-gather handles messages ≤ 2 MiB and NCCL handles the rest. 1,048,576-token
context, **32 sequences**, chunked prefill 8192, one prefill at a time, prefix caching, thinking off by default.

The engine changed under this recipe on 2026-11-16. It is **Owen's candidate 60**: vLLM tree `afcaac35…`, the
`karmic-kraken-beta` fork plus his c60 patch series, TileLang and TileKernels for GB10, a patched NCCL, sparknet
0.5.0 in the image, `bin/spark` as the renderer, and `--load-format direct` — the b12x loader is gone from this build.
The image reports the same `0.29.1rc1.dev187+gaf1c01499` version string as r6x because the base nightly is unchanged;
pin the engine by the **vLLM tree label** (`afcaac35…`), not by that string.

The two levers that matter in this shape are Owen's, not ours. **Prefill yield (`short`, 10 s limit — his profile sets
it, the engine's own default is `off`)** lets a step
carry the waiting prompts whose prefill finishes within one chunk instead of one long prompt's whole aligned chunk;
above 262,144 computed tokens (and only with decodes runnable) his deep-context interleave steps the chunk onto the
2,048 grid and adds pure decode steps after ≥1 s of accumulated prefill stall. That is what turns a 413K prefill from
a wall into a background job. And **the c60 engine itself** is where the prefill comes from: his qualification reads
9,521 / 9,924 / 9,830 tok/s at 32K / 131K / 262K on a switched four-Spark box.

Our shape differs from his. His `tp4-switched` profile runs 16 seats, `--max-num-batched-tokens 16384`,
`--long-prefill-token-threshold 16288` and late admission at layer 30; ours runs 32 seats, 8,192, 3,856 and layer 20,
because this lane serves guests rather than one stream, and because we declined the larger batch budget on our own
mixed-traffic evidence in October. Our node map, page size (4 KiB), kernel (7.0.0-1025) and memory-saver-off are
ours. Worth knowing when comparing numbers: **his own c60 tp4-switched receipts booted 4 KiB pages without the
memory saver too** — every suite's `live_warnings` records `running '7.0.0-1024-nvidia' / 4096` on his nodes — so his
figures and ours are page-comparable even though his profile declares 64 KiB.

## Hardware

See `hardware/dgx-spark-4x.yaml`. Four DGX Spark GB10 (Sparks 5–8, the second pod on the shared CRS804), two 200G RoCE
rails per node (10.0.0.N and 10.0.1.N, MTU 9000, lossy RoCEv2), driver 580.178.04, CUDA 13.0, kernel 7.0.0-1025-nvidia (DGX OTA 7.6.0; all four nodes checked 2026-11-16),
4 KiB pages. `nvidia-smi` on spark5 reads SM 2190 MHz with a 2418 MHz application clock and no clock lock (2026-11-16); clocks were not changed for this run. Snapshot: `results/2026-11-16-r7-candidate60/system.json` (spark5, collected 2026-11-17 02:42 UTC); Sparks 6–8 report the same
kernel, page size and driver. The arm files in `results/2026-11-16-r7-candidate60/config/` declare `6.17.0-1031-nvidia`, copied
from the October recipe — that field is metadata the renderer does not enforce and was stale; `scripts/c60cfg/` carries the corrected value.

Memory per rank: weights + draft 72.19 GiB, KV **10.5 GiB** (fixed by `--kv-cache-memory-bytes`), CUDA graphs ~2 GiB
(c60 widens the capture to 203 rows because copy drafts verify more rows than k=5 — the boot log prints it). The
container is capped at `--memory=112g` and pinned to `--cpuset-cpus=5-9,15-19`. MemAvailable floor during the C16
load on the promoted boot: **25.2 GiB** on the tightest node (the r6x control read 22.2 GiB).

## Software

- Image **`spark-ds41f:candidate-60`**, Id `sha256:2f2bf6e1091b500f2a62f170746134a4fdc57e68fc8893f7d8122e8fc5b3ddf2`,
  identical on all four nodes on every r7 boot. It is a local build from pinned sources (not pullable), and a rebuild
  elsewhere will not reproduce the Id.
- Deployment repo **christopherowen/spark-ds41f @ `fcc97650`** (the checkout the renderer comes from), image built
  from its `docker/Dockerfile`. The vLLM tree inside is **`afcaac3594acdea245d9543dbbea2376104470ba`**, confirmed
  from the image's own `local.spark-ds41f.vllm.tree` label by the in-image gate run
  (`results/2026-11-16-r7-candidate60/image-gate-run.log`).
- Pinned trees inside the image (image labels): vLLM `afcaac35`, NCCL `6af10aa7`, TileLang `741806c7`,
  TileKernels `cb2fa41c`, sparknet `e06f6a06` (0.5.0 + the single-slot transfer cap), and **no b12x label** — the
  loader is gone.
- Weights: unchanged from the r6x recipe — 48 shards, 510,296,708,312 bytes per node, the same bytes that were
  sha256-verified against the Hub LFS etags and re-verified after the copy to spark1
  (`results/2026-10-19-vllm-ladder-r2/raw/weights/`). Not re-hashed tonight; the mount is the same canonical path on
  every node.
- Public API: `http://<head>:8002/v1`, served model `deepseek-v4.1-flash-spark-vllm`, **no key** (the rendered
  `docker run` in `results/2026-11-16-r7-candidate60/rendered-spark5.sh` carries no API-key argument).

### The two mounted overlays

Both are read-only bind mounts, both on all four ranks or none, both checksummed in `overlays/SHA256SUMS`.
`overlays/README.md` has the provenance, the mount targets and the kill switches. In one line each:

- **`overlays/so-bitmask/vllm/v1/structured_output/__init__.py`** — c60's file plus **Zee-B33's three-line fix for
  [spark-ds41f#1194](https://github.com/christopherowen/spark-ds41f/issues/1194)** (bitmask sized from
  `max(num_speculative_tokens, num_verification_tokens)`), mounted over
  `/opt/spark-ds41f/candidate/vllm/vllm/v1/structured_output/__init__.py`.
- **`overlays/dsml-c60/vllm/parser/deepseek_v41.py`** — our re-port of the structural-DSML-closer rule onto c60's
  parser, mounted over `/opt/spark-ds41f/candidate/vllm/vllm/parser/deepseek_v41.py`. Kill switch
  `VLLM_DSV41_STRUCTURAL_DSML=0`; the module logs `structural DSML closers active` at startup and refuses to
  half-apply.

Three more overlay directories are kept but are **r6x-only** (their `/opt/spark3/...` mount targets do not exist on a
c60 image): `late-admission/` (in-tree in c60), `sparknet-0.5.0/` (c60 ships a newer 0.5.0), `dsml-structural/` (the
r6x parser; `ImportError` on c60). Do not mount them here.

## Launch

```bash
export SPARK_HOSTS="spark5 spark6 spark7 spark8"     # rank order; rank 0 = head; must resolve on the nodes too
# each node: image spark-ds41f:candidate-60, weights in ~/models/DeepSeek-V4.1-Flash-1017,
#            cache dir ~/spark-ds41f/cache, the two overlays in ~/r7-overlays
# render host: a spark-ds41f checkout at fcc97650 in ~/spark-ds41f-r7-c60, plus
#            scripts/c60cfg/nodes-switched.laneC.local.json -> config/
#            scripts/c60cfg/arms/*.json -> experiments/2026-11-16-c60-prefill/arms/
bash scripts/launch-r7-c60.sh up
```

`launch-r7-c60.sh` is the campaign runner that booted every arm on 2026-11-16, with pool A/B replaced by an explicit
host list. It renders each node with Owen's own `bin/spark --cluster-config <arm> render <node>`, post-processes the
result (container name, `--restart=no`, the two overlay mounts, cache path), gates on `MemAvailable ≥ 100 GiB` after
dropping caches and compacting, then launches ranks **3, 2, 1, 0** and waits for `/v1/models`. The docker command's
body is not rewritten by us — that is the point of using his renderer. `STOCK=1` renders the same arm with no overlay
mounts at all (arm A3b's shape).

Overrides: `ARM` (default `A3-our-shape-yield-short`), `CFG`, `DS41F_DIR`, `RENDER_HOST`, `OVERLAY_DIR`, `NAME`
(default `dsv41-r7-lane`; the arms ran as `dsv41-r7-<arm>`), `MODEL_DIR`, `PORT`, `MIN_AVAIL_GIB`, `BOOT_TIMEOUT`.

Boot: **135 s** warm (caches populated; the promoted boot went 20:04→20:06 CST), **1,392 s** cold on an empty JIT
cache. The first c60 boots of the night paid the ~23 min compile; every arm after them was warm. A c61-overlay boot
compiles its extra TileLang kernels again and read 211 s.

## Verify

```bash
export BASE_URL=http://<head>:8002/v1
python3 scripts/gates.py out/                       # /v1/models, exact-reply smoke, get_weather tool round, preemptions
bash scripts/run_battery.sh out/ mylabel full mixed greedy
BASE_URL=http://<head>:8002 bash scripts/post_arm.sh out/post-r7   # greedy x2 + under load, so_probe, DSML live probe
python3 scripts/so_probe.py http://<head>:8002 deepseek-v4.1-flash-spark-vllm 32 2   # the #1194 guard probe
```

Before trusting a number, check the mount identities inside the container — a mount that landed on a directory is
invisible to `docker ps`:

```bash
docker exec <container> sha256sum /opt/spark-ds41f/candidate/vllm/vllm/v1/structured_output/__init__.py \
                                    /opt/spark-ds41f/candidate/vllm/vllm/parser/deepseek_v41.py
#   d286743334c81755c3ee3bc8d33eb3bf85760a9833085289154474ca6fbe165e   structured_output/__init__.py (so-bitmask)
#   8a8b293708eb15b763ed6fd4c6f52e85bff830e254f3029b69a194951d0c35c6   parser/deepseek_v41.py        (dsml-c60)
#   844e47009c9607fe11a1d801e43519d318dc75c6fd545ddc2f08b33fdcc074a7   the image's own parser (stock c60)
```

**What the gates proved on 2026-11-16**, all on the promoted boot (`results/2026-11-16-r7-candidate60/`):

- **Determinism.** Two greedy runs on one boot: **40/40 byte-identical**. Under 15 background streams: **40/40**
  identical to idle. And **40/40 across two boots** — `r7-a3` on Sparks 5–8 vs the same config on Sparks 1–4.
- **Quality.** qeval content **71/75**, the same four misses as the r6x control; 1M needle **3/3**; smoke and the
  `get_weather` tool round pass with **0 preemptions**; the DSML live probe passes (`ALL_PASS`).
- **Mixed traffic.** 0.704 s behind a cold 413K prefill and 2.048 s on a warm 68K prefix — both inside the gate.
- **Structured output.** `so_probe` **64/64** (32 concurrent `response_format` requests whose answers copy 40 records
  out of the prompt, two waves, server alive). The r6x control reads 64/64 too.
- **In-image parser gate.** 25 cases × 5 modes × 2 paths, `ALL GATES PASS`; stock fails B/C/D/E/E2/J, the overlay
  fixes all six and changes only N, N2, O2, Q, R.

There is **no teacher-forced non-inferiority run in this round**. The r6x baselines come from the kept r6x container
and a different engine build invalidates per-token baselines; that gate needs a fresh reference boot before it can be
re-run. Quality evidence here is qeval + needle + determinism + the tool probes, which is weaker, and it is stated as
weaker.

## Results

The promoted boot (2026-11-16 20:04–20:32 CST, ranks on Sparks 5–8), against the same-night r6x control: the table at
the top of this page, and `results/2026-11-16-r7-candidate60/README.md` for the five-lane comparison, the gate table
and the two declined arms.

- C1 by category on the promoted boot, per stream tok/s: coding **164.42**, format **160.30**, json **151.10**,
  math **137.57**, reasoning **123.71**, summary **95.90**, prose **90.74**, narrative **73.52** (counting-ceiling
  category 187.90, excluded from the headline). C16 per stream: coding 56.80, format 62.20, json 45.57, math 53.51,
  reasoning 41.56, summary 28.79, prose 30.84, narrative 22.91.
- 1M needle **3/3** in **131.2 s** (control: 263.3 s).
- KV pool **12,034,343 tokens** = 11.48 full-length sequences on the engine's own line (control 8,580,566 / 8.18);
  same 10.5 GiB per rank; concurrent capacity at that size was not tested.
- 0 preemptions at the functional gates (`vllm:num_preemptions_total 0`); the post-suite's own scrape is in
  `results/2026-11-16-r7-candidate60/post/r7-a3/metrics-after-gates.txt`.
- The same config on the other pool (Sparks 1–4) reads C1 122.72, C16 541.56, 131K prefill 8,645.8 — the two pools
  agree within ~1%, which is why a pool-B arm could be compared against a pool-A control at all.

## Known limits

- **candidate 60 is an unfrozen candidate, not a release.** Owen's `1107` release is the qualified artifact; c60 is a
  patch series on the `karmic-kraken-beta` fork published 2026-11-16 and it can move under us. Pin the tree
  (`afcaac35…`) and the image Id, and re-run the gates before adopting anything newer.
- **The image is a local build from pinned sources and is not pullable.** ~23 min of JIT on the first boot, a
  different image Id on any rebuild. There is no registry digest to fall back on — the tree label is the identity.
- **Boot cost:** 1,392 s cold (empty JIT/AOT cache), **135 s** warm. All published numbers are warm boots.
- **c61 was tested and not adopted.** Its C1 effect is gated on a single request and reads +0.81% against its
  same-pool twin — inside this instrument's spread — while C16 reads −2.41%. Six mounted files are not worth that.
  Details in the results README.
- **The DSML live probe passes on stock c60 too.** Our `dsml-c60` overlay is not needed for the single quoted closer
  the live probe builds; it covers the **6 synthetic genuine-atom cases** where c60's token-provenance rule still cuts
  the value. The in-image 25-case gate is the evidence for the overlay; the live probe is not.
- **`so-bitmask` is insurance.** We did not try to reproduce the #1194 crash: `so_probe` read 64/64 on r6x (which has
  no copy drafts) and on every c60 arm — all of which carried the fix. Stock c60 without the fix was never probed. The
  reproductions in the issue are other people's rigs.
- **qeval does not discriminate here.** 71/75 on every lane and every arm, same four misses
  (`code_interval_intersect`, `json_count`, `json_escape`, `math_m9`) — inherited from the model, not from this config.
- **No teacher-forced gate this round** (see Verify), and **no speculative-decoding acceptance measurement**: the
  historical r6x figure (0.7151 accepted/drafted, 3.334 tokens per verify step, 2026-10-23) does not transfer to a
  different engine build and was not re-measured. The C1 headline therefore carries no `acceptance_len`.
- **1M is configured and one 1M needle passes** — a single sequence, retrieval only, not long-context reasoning, and
  not eleven concurrent 1M sessions. The KV line implies 11.48 of them; nobody has run that.
- **C16 is a ±2% row on this rig**, and `mixed_warm` has a 1.5–3.0 s tail. Do not read 1% deltas out of either.
- **The hardware block is a 2026-10-19 snapshot** carried forward (`system.json` in this run's directory is a
  placeholder). The nodes are the same four; nothing about them was re-collected tonight.
- **The shape is unqualified upstream.** `tp4-switch-4k` is ours: 32 seats, 8,192 batched tokens, threshold 3,856,
  KV 10.5 GiB, our node map, our two NIC-combining overrides. Owen does not test this shape.
- **Overlays are copies, so they age.** If #1194 is fixed upstream or the structural rule lands, these files are
  behind by construction; `SHA256SUMS` and `overlays/README.md` pin what they came from.
- **Hygiene as run:** caches dropped and memory compacted before every boot; THP is `madvise` on Sparks 5–8, not
  `never`.

## Rollback

`bash scripts/rollback-to-r6x.sh` with `SPARK_HOSTS` set: it stops `dsv41-r7-lane` on all four nodes (container kept),
then `docker start dsv41-karmic-kraken` **rank 3 → 2 → 1 → 0** (the kept r6x containers from 2026-10-23), waits for
`/v1/models`, and prints the KV line — **8,580,566 tokens (8.18×)** means r6x answered, 12,034,343 (11.48×) means the
r7 lane did. The r6x-era launcher is still in `scripts/` (`cluster.sh` → `launch-vllm-tp4.sh`) and does not launch
this lane.

Overlay-only rollback, without touching the engine: recreate the containers with `STOCK=1` (no mounts), or set
`VLLM_DSV41_STRUCTURAL_DSML=0` for the parser alone, or `docker cp` the image's own files back
(parser sha256 `844e4700…`).

The other published DeepSeek-V4.1 lanes:

- the SGLang sibling on 4× Spark: [`dgx-spark-4x/deepseek-v4.1-flash-sglang-tp4-roce`](../deepseek-v4.1-flash-sglang-tp4-roce/)
- the single-GB300 recipe: [`dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark`](../../dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/)
- blog: [DeepSeek Flash 1017: TensorFold vs SGLang vs vLLM on 4 DGX Sparks](https://al-engr.com/dsv41-1017-three-way-tensorfold-sglang-vllm-4x-dgx-spark.html) (r6x era)

## Credits

**Christopher Owen** ([@christopherowen](https://github.com/christopherowen)) — **this lane runs Christopher Owen's
engine.** [spark-ds41f](https://github.com/christopherowen/spark-ds41f) candidate 60: the 689-patch series on Local
Inference Lab's vLLM fork, the TileLang kernels for GB10, `bin/spark` build/render, the `tp4-switched` profile we
extend, and the features this recipe's numbers are made of — one-shot RoCE collectives and the switched transport in
[sparknet](https://github.com/christopherowen/dgx-spark-networking), late admission, **prefill yield**, deep-context
interleave, prompt-cache retention, the 8-bit vocab screen, the DSpark draft budget, the c61 decode branch we tested
and declined, and the determinism/testing method. We built his code from his pinned sources; our part is the shape,
the overrides, the two overlays and the measurements.

**Zee-B33** — diagnosed and fixed the structured-output bitmask crash ([spark-ds41f#1194](https://github.com/christopherowen/spark-ds41f/issues/1194)).
Our `so-bitmask` overlay is their three-line fix, applied verbatim. Reproductions: **mjcryptocloud** (4-node, 4 KiB)
and **hxfhd** (TP3).

**Hector Isaza** ([@hecisaza](https://github.com/hecisaza)) / TensorFold patch `0052` (issue by **ecohash-co**) — the
DSML-closer idea in [MiaAI-Lab's TensorFold recipe](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold).
Our `dsml-c60` parser is our own re-port, not their code.

**Local Inference Lab** — the `karmic-kraken-beta` vLLM fork and b12x. **TileLang** and **DeepSeek TileKernels** — the
kernel family. **DeepSeek** — the model and the DSpark draft. **vLLM** — the base. **NVIDIA** — NCCL.

**knapcio** — the original four-Spark TP4 findings that Owen's `docs/inspiration.md` lists as hypotheses to reproduce.

Field data quoted in research only, not adopted: **neko-legends** (switched 4-Spark 1107 site figures).

Work credit: research and porting workers ran on DeepSeek-V4.1-Flash on our GB300; orchestration and measurements by
Milo (session model claude-opus-5-5). This page was drafted by a DeepSeek-V4.1-Flash worker on our GB300 and reviewed
by Milo (claude-opus-5-5).

## History

This page describes the r7 lane only. The r6x era is kept, not retold:

- `results/2026-10-23-adopted-l20-net/` — the r6x adopted config (KV 10.5 GiB, 32 seats, late admission at layer 20,
  sparknet 0.5.0) with its two boots and receipts.
- `results/2026-10-19-vllm-ladder-r2/` — the Round-1/Round-2 ladders that found nothing in the vendor knobs, plus the
  hardware snapshot and weight hashes still referenced above.
- `results/2026-11-06-dsml-structural-closers/` — the r6x structural-DSML overlay, its 25-case matrix and live probe.
- `scripts/cluster.sh`, `scripts/launch-vllm-tp4.sh`, `scripts/render_launcher.py`, `scripts/verify_launcher.py` — the
  r6x launcher and its validator, unchanged, for the kept r6x containers.

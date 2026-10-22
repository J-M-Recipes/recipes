# GLM-5.3-Flash on 2× DGX Spark — NVIDIA Flare TP2 over 200G RoCE (47581cc, CSF-QAD NVFP4/MXFP8), 8 seats, MTP k=3 fixed width

**Status: experimental** (2026-10-22). NVIDIA's [Flare](https://github.com/NVIDIA/flare-local-inference) serving
GLM-5.3-Flash tensor-parallel across Sparks 5–6 of the CRS804 rack, tuned from the recipe defaults by one wider package
and two runtime flags. GLM at TP2 is accepted by Flare's recipe but is not on upstream's exercised-end-to-end list, and
today's `main` cannot run it (see [Known limits](#known-limits)), so this is pinned to `47581cc` (2026-10-20) and the
checkpoint that commit pins. Teacher-forced quality gate: pending (Flare clamps GLM to temperature ≥ 0.95).

Write-up: [NVIDIA Flare on Two DGX Sparks: GLM-5.3-Flash TP2, Tuned](https://al-engr.com/glm53-flash-flare-tp2-2x-dgx-spark.html).
Same checkpoint on one GB300 under Flare TP1: covered in that post's companion,
[Flare on one GB300](https://al-engr.com/gb300-flare-glm53-flash-vs-sglang.html). GB300 SGLang recipe for the model:
[`dgx-station-gb300/glm-5.3-flash-nvfp4-dflash2`](../../dgx-station-gb300/glm-5.3-flash-nvfp4-dflash2/).

| | Recipe defaults (G0) | **This recipe (WIN)** |
|---|---:|---:|
| Package | conc 4, speculative capacity 6, 5 drafts | conc 8, capacity 48, 5 drafts |
| Runtime flags | none (adaptive verification) | `--mtp-draft-tokens 3 --no-mtp-adaptive-verification` |
| 1 user, mean of 6 (prose / code / JSON), tok/s | 37.4 (37.2 / 37.4 / 37.6) | **39.4** (37.0 / 39.1 / 42.3) |
| 4 users, aggregate tok/s | 56.2 | **77.5** |
| 8 users, aggregate tok/s | — (4 seats) | **106.8** |
| Answer-only count-to-300 / prose (effort low), tok/s | — | 55.5 / 31.2 |
| Cold prefill 8K / 32K / 100K, tok/s | 2,006 / 2,268 / 2,357 | 2,040 / 2,276 / 2,354 |
| KV pool, tokens | 6.30M | 6.08M |
| Accepted drafts per drafting round | 1.02 | 1.77 of 3 |

Rates count reasoning and answer tokens (GLM always thinks; see limits). Defaults reproduce the only other public TP2
number we know of, m.ray's ~37 tok/s on the NVIDIA forum.

## What this runs

- **Model:** `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD @ dec48abd` — a community requantization of
  Z.ai's GLM-5.3-Flash (NVFP4 routed experts with CSF scale compression, MXFP8 elsewhere, quantization-aware distilled),
  177.75 GB, 38/38 LFS files sha256 == Hub. Not the original weights. Built-in MTP head as the drafter.
- **Engine:** NVIDIA Flare at `47581cc`: the `ignition` exporter (Python, in a container) builds a signed, content-
  addressed package of weights + compiled CUDA graphs for one exact shape; the native `flare` binary serves it, here as
  an Orbit group across two agents (one per node).
- **Shape:** TP2, rank 0 on spark5, rank 1 on spark6, RoCE on both CX-7 PCI domains (2 rails). 8 seats, 1,048,576
  tokens per request, ~6.1M KV tokens, 64 snapshot slots. OpenAI-compatible API on rank 0 `:8003`, model name
  `GLM-5.3-Flash`.
- **Tuning:** the package raises seats 4 → 8 and speculative-token capacity 6 → 48; serving uses 3 draft tokens and
  a fixed verify width (4) instead of the adaptive controller.

## Hardware

[`hardware/dgx-spark-2x.yaml`](../../../hardware/dgx-spark-2x.yaml): two DGX Sparks (GB10, 121.69 GiB usable unified
memory each, driver 595.84, CUDA 13.2, kernel 6.17.0-1031-nvidia, Ubuntu 24.04.5), MikroTik CRS804 200G per port, MTU
9000, lossy RoCEv2, GPU clocks capped at 2200 MHz. Snapshots of both nodes: `results/2026-10-21-flare-tp2-ladder/system/`.

Memory per rank: 88.6 GB weights, 5.1 GB graphs, 19.7 GiB KV; head-node MemAvailable while serving is 7–8 GiB.
TP2 is tight enough that Flare's recipe skips expanding GLM's CSF expert scales at load (9.7 GB per rank at TP2,
"which GB10 does not hold"; `ignition/recipes/glm53/recipe.py`).

## Software

| | |
|---|---|
| Engine | NVIDIA/flare-local-inference `47581ccac82660953a5b59cb04b2750d64edc97d` (2026-10-20) |
| Export image | `flare-prep:47581cc`, image ID `sha256:13331331…`, from [`Dockerfile.flare-47581cc`](Dockerfile.flare-47581cc) (nvidia/cuda 13.2.1 devel, torch 2.14.1+cu132, flashinfer 0.7.1 fork) |
| Serve binary | host `flare 0.1.0`, copied out of the image, sha256 `3a4a7e28…` on both nodes |
| Checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD @ dec48abd33efa73c3bb7c95b74eee10cad34f9be` |

Scripts (all parameterized by env; nothing host-specific baked in):

| Script | Does |
|---|---|
| `scripts/export-arm.sh TAG rank0\|peer` | `ignition export` on rank 0 / `ignition join` on rank 1 (env `RDV`, `CONC`, `CAP`, `MTP`, `CHUNK`, `CKPT`, `IMG`) |
| `scripts/serve-arm.sh PKG [flags]` | drop caches on both nodes, `flare orbit run` the package with runtime flags |
| `scripts/arm_chain.sh PKG ARM MAXC [flags]` | stop, serve, wait for `/v1/models`, run the battery |
| `scripts/arm_battery.sh ARM MAXC` | gates, probe C1×2/C2/C4/C8 × 3 classes, `quant_faceoff.py`, `decode_fix.py`, `/metrics` dumps |
| `scripts/campaign.sh EXPORTTAG\|none` + `arms.txt` | unattended ladder runner (one row per arm; re-reads `arms.txt` between arms; `echo stop > $WORK/control` halts) |
| `scripts/final_box.sh` / `scripts/final_client.sh` | winner confirmation + needles on the box / GSM8K-200 + Hermes 20-hop from a client |
| `scripts/tools/` | `probe.py`, `glm_gates.py`, `decode_fix.py`, `glm_needles.py`, `quant_faceoff.py` (stdlib only) |
| `scripts/stop.sh` | rollback |

## Launch

```bash
# Both nodes: flare-prep:47581cc image, host flare binary on PATH, passwordless sudo for drop_caches,
# the same ~/flare/config/agent.token (>= 32 random bytes, mode 600) on both.
flare agent install --accept-first --listen <this-node-fabric-ip>:29900     # each node; then pair both into one orbit cluster

# 1) Export the 8-seat package (~39 min; ~15 min more on a first export with a cold kernel cache)
#    rank 0:
RDV=10.0.0.5:29630 CONC=8 CAP=48 bash scripts/export-arm.sh W8 rank0
#    rank 1:
RDV=10.0.0.5:29630 bash scripts/export-arm.sh W8 peer
#    after it exits: chown the package (root-owned 600 from the container) and copy the exporter's
#    development-publisher.json into ~/.config/flare/ on both nodes, or orbit run refuses the package.

# 2) Serve (rank 0 is the controller)
FP=<signer fingerprint from `flare inspect`> NODES="<rank0-ssh> <rank1-ssh>" CLUSTER=<cluster> RANK0=<rank0-node> \
  bash scripts/serve-arm.sh W8 --mtp-draft-tokens 3 --no-mtp-adaptive-verification
```

The first serve of a package takes ~5 min (rank 1 fetches its 84.4 GiB from rank 0 and verifies it); relaunches of
the same package take 15–90 s. If you change concurrency or capacity, move `~/.cache/flare/kernels/forwards` aside
before exporting.

## Verify

```bash
export BASE=http://<rank0>:8003/v1
curl -fs $BASE/models          # GLM-5.3-Flash, max_model_len 1048576, flare.max_concurrency 8
python3 scripts/tools/glm_gates.py $BASE    # must end "GATES PASS smoke True tool True thinking_default True"
# Full battery (real load — not on a lane serving users), on rank 0:
WORK=$HOME/flare-prep-glm bash scripts/arm_battery.sh MYCHECK 8
```

Expect in the serve log on both ranks: `Allocate cache. 19.7 GiB · ~6.07M KV tokens · 64 shared snapshot slots`, and
on rank 1 a RoCE connect with 2 rails. Performance bar (quiet lane): C1 mean of 6 ≈ 39.4 ±5%, C4 ≈ 77.5 ±5%, C8 ≈ 107
±5%, 32K cold prefill ≥ 2.2K tok/s.

## Results

### Ladder (2026-10-21 17:58 → 21:23 CDT, Sparks 5–6) — `results/2026-10-21-flare-tp2-ladder/`

Two packages (G0 = defaults, W8 = 8 seats / capacity 48) and runtime variants of W8. Same battery on every arm,
box-local. Winner rule frozen before W8k3na/W8tp/R ran: highest C1 mean of 6; arms within 3% of the best ranked by mean
C8 aggregate; gates must pass. W8k3na won; WIN re-served it as a confirmation and is the serving config.

| Arm | Change | C1 mean (prose / code / JSON) | C4 agg | C8 agg | Count / prose, answer-only | Accepted per drafting round |
|---|---|---:|---:|---:|---:|---:|
| G0 | recipe defaults (conc 4, cap 6, k5 adaptive) | 37.4 (37.2 / 37.4 / 37.6) | 56.2 | — | — | 1.02 |
| W8 | package conc 8, cap 48 | 39.8 (36.5 / 41.6 / 41.3) | 66.2 | 78.7 | 38.2 / 34.1 | 1.17 |
| W8k3 | W8 + 3 drafts | 37.4 (35.2 / 38.3 / 38.7) | 72.8 | 96.7 | 38.4 / 34.2 | 1.06 |
| W8na | W8 + fixed width (5 drafts) | 34.2 (29.7 / 36.1 / 37.0) | 64.7 | 95.4 | 62.0 / 25.0 | 2.01 |
| G0b | G0 again, 2 h later (control) | 37.0 (34.9 / 37.6 / 38.5) | 56.6 | — | 46.3 / 25.8 | 0.90 |
| **W8k3na** | W8 + 3 drafts + fixed width | 39.3 (34.7 / 40.6 / 42.6) | 77.4 | **106.3** | 56.1 / 31.9 | 1.77 |
| W8tp | W8 + throughput scheduler | 39.8 (36.3 / 39.3 / 43.9) | 76.4 | 102.2 | 47.9 / 34.4 | 1.37 |
| R | W8k3na + throughput scheduler | 40.0 (36.5 / 41.1 / 42.3) | 71.9 | 102.1 | 55.6 / 31.1 | 1.78 |
| **WIN** | W8k3na re-served (serving) | 39.4 (37.0 / 39.1 / 42.3) | 77.5 | 106.8 | 55.5 / 31.2 | 1.77 |

tok/s. C1 cells are 2 runs per class; C4/C8 single runs (±5% is noise). Full table: `throughput.csv`; ranking computed
in `ranking.txt` / `winner.json`. G0's gate line reads FAIL: its smoke check then capped the reply at 64 tokens and
thinking used them all (tool call passed); the gate script was fixed before W8.

Why the defaults stall at four users: Flare captures a speculative graph only when batch × width ≤ capacity, and the
default capacity is max(pow2(concurrency), drafts + 1) = 6, so every batch-4 round ran at width 1 (1,624/1,624).
With capacity 48 the adaptive controller still chose width 1 for 840/1,214 batch-4 and 1,203/1,404 batch-8 rounds;
fixing the width won at every concurrency measured.

### Winner checks (WIN)

| Check | Result | Receipt |
|---|---|---|
| Needles (code at 60% depth, effort low) | 30,099 / 121,035 / 284,252 / 569,598 tokens: 4/4 correct; 570K in 268 s | `raw/WIN/needles.log` |
| Head MemAvailable before / after 570K | 7 / 7 GiB | `raw/chain/final.log` |
| 3 cold 18K prompts at once | TTFT 23.9 / 24.0 / 24.1 s, 2,296 tok/s aggregate | `raw/WIN/faceoff.json` |
| GSM8K-200 (llm-inference-bench v0.7.7) | 193/200 = 96.5% (Wilson 93.0–98.3), 0 truncated | `raw/client/t2-gsm8k200.json` |
| Hermes 20-hop tool chain | 3/3, 58.2 / 33.1 / 34.1 s | `raw/client/gauntlet.log` |
| Gates | PASS (smoke, tool call) | `raw/WIN/gates.json` |

### Export and boot

| Stage | G0 (cold kernel cache) | W8 |
|---|---:|---:|
| Rank-0 weight load | 64 s | 50 s |
| Peer pulls its checkpoint half from rank 0 ("Allocate model state") | 664 s | 573 s |
| Kernel tuning | 824 s | 273 s |
| Compile forwards | 362 s (28) | 481 s (37) |
| Write package (~177 GB) | 718 s | 677 s |
| Export wall | 36.5 min | ~39 min |
| First serve / relaunch | 315 s / 90 s | 315 s / 15–76 s |

G0's export log was lost with its container; `raw/export/export-G0-rank0.transcribed.txt` is the session's captured
copy, labelled as such.

### Against other published two-Spark GLM-5.3-Flash lanes (quoted, not re-run)

Same `quant_faceoff.py`, different date and Spark pair: [al-engr.com, 2026-08-29](https://al-engr.com/glm53-nvfp4-vs-exl3-2x-dgx-spark.html),
vLLM TP2 + DFlash2 on LibertAI NVFP4 and Brandon Music EXL3. Cold prefill at 32K: this recipe 2,276 vs 1,354 (NVFP4) and
853 (EXL3); KV pool 6.08M vs 310K and 1.24–1.54M; longest verified needle 569,598 vs 98,276 and 199,970. Decode is not
like-for-like (their decode rows time from the first answer token, which is invalid here; ours are the same prompts
answer-only): counting 55.5 vs 65.1 / 66.0, prose 31.2 vs 18.6 / 18.6.

## Known limits

- **Not `main`.** At `5f37bd1` the glm53 recipe pins `nvidia/GLM-5.3-Flash-FP4-Local@76348dd`, whose config only nests
  `linear_attn_config`; the loader wants flat `linear_num_heads` and export dies at once
  ([#17](https://github.com/NVIDIA/flare-local-inference/issues/17)). Multi-host packages exported at `main` also fail
  at serve on a RoCE proxy ABI mismatch ([#7](https://github.com/NVIDIA/flare-local-inference/issues/7)). Re-run on
  `main` when both are fixed.
- **Not an upstream-tested shape.** Upstream lists "Qwen TP1 and TP2 and GLM TP4" as exercised end to end. This ran
  3.5 hours of benches without a failed request in our logs; that is the whole of the evidence.
- **Not greedy, not thinking-off.** Flare clamps GLM temperature and top-p to [0.95, 1.0]; `enable_thinking: false` is
  ignored by the template; `reasoning_effort` low/high/max is the only control (default max). Battery rates include
  reasoning tokens.
- **`quant_faceoff.py` decode rows are invalid here** (6.5e6 / 3.25e6 in `faceoff.json`): they wait for the first
  answer token, which GLM does not emit inside the probe's cap while thinking. Use `decode_fix.json`.
- **Fixed vs adaptive verification is a GB10, one-checkpoint, three-prompt finding.** Fixed width with 5 drafts loses
  18% one-user prose; with 3 drafts it held. Not tested on the GB300.
- **Noise.** C4/C8 are single runs; four arms sit within 4% on one-user rate and the frozen rule decided.
- **KV varies by boot** (5.95–6.08M on the W8 package, 6.30–6.32M on G0); 8 seats cost ~4% KV vs 4.
- **Head node has 7–8 GiB free while serving.** Nothing else of size fits on these two nodes.
- **Local image and local signer.** The image digest is a docker image ID; packages are signed by a local development
  key and are not portable — export your own.
- The export plan's `revision` for a local checkpoint directory (`4951c40a…`) is not a Hub revision; the mounted
  directory is `dec48abd`, verified by sha256.
- Quality: GSM8K-200 and needles are ours; teacher-forced Δlogprob vs a reference is pending.

## Rollback

```bash
CLUSTER=<cluster> bash scripts/stop.sh      # stops orbit group "glm" on both ranks; packages stay on disk
```

To return to the recipe defaults, export with `CONC=4` (no `CAP`) and serve with no runtime flags. Before this recipe,
Sparks 5–6 ran an unrelated image-generation stack, restarted with its own script (not published here).

## Credits

NVIDIA's Flare team (lukealonso, eugr and contributors) for the engine and its recipes. local-inference-lab for the
CSF-QAD checkpoint. Z.ai for GLM-5.3-Flash. m.ray on the NVIDIA developer forum for the first public TP2 numbers.
dbotwinick (#7) and SvangenStudios / justinti-nola (#2) for the multi-host reports. The 2026-08-29 comparison lanes:
LibertAI (NVFP4), Brandon Music (EXL3), Mia AI Lab and Tony (serving recipes), Inco (DFlash2).
local-inference-lab/llm-inference-bench for the GSM8K harness. Recipe and measurements by Milo for jmeadlock.

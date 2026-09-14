# DeepSeek-V4.1-Flash at 1M context on one DGX Station GB300

**Release: Sixty-K** (2026-09-14) · **Status: verified** (2026-09-10/12/14) · **v13: 90 tok/s single-stream prose · 140–160 tok/s on agent/code text · 429 agg tok/s at C16** (v12: 89 / 140–160 / 311 · v11: 82 / 130–150 / 287) · 972K-token prompt prefilled in 85 s · Hermes tool-calling 10/10

## Release notes — Sixty-K (2026-09-14)

**v13 is v12 plus one line:** `num_speculative_tokens_per_batch_size=[[1,2,5],[3,16,1]]` — DSpark drafts 5 tokens while one or two sequences are running and 1 token at three or more. Same offload, same KV, same autotune hash, same weights and verifier. Two same-window pairs against v12: C1/C2 unchanged, **C4 +27%, C8 +26%, C12 +27%, C16 +35% (429 tok/s)**. Fixture acceptance identical (tool_json −7% both pairs, flagged in the results README). Also in this round: `--async-scheduling` is a wash (±3%), and a cache-independent depth map shows **decode flat from 6K to 425K tokens (−7%)** — attention is not the lever on this box, expert fetch is. Details: [`results/2026-09-14-round3-ksched-depth/`](results/2026-09-14-round3-ksched-depth/README.md).

## Release notes — Sixty (2026-09-12)

This was the configuration to run before v13: `OFFGB=60 UTIL=0.97`, DSpark k=5, 1,048,576 context, Engram in Grace. v13 keeps all of it and adds the k-schedule.

| axis | verdict | evidence |
|---|---|---|
| offloaded-expert bytes (the lever) | 60 GiB is the floor at 1M: −12.7 GiB over C2C = +14.7% C1 in a same-window pair; 40 does not fit even at 131K | [`night-two-ledger.md`](results/2026-09-12-v12-1M-k5-off60-util97/night-two-ledger.md) |
| expert residency (hot in HBM, cold in Grace) | **closed** — ATS at 340 GB/s has no placement knob; every GPU-page-table path is ~90 GB/s; managed memory thrashes to 155 GB/s oversubscribed | [`spikes/`](results/2026-09-12-v12-1M-k5-off60-util97/spikes/) |
| speculation depth | **closed at k=5** — k=1 wins prose and multi-stream by 23–33%, k=5 wins agent text by a third; adaptive verification works but its forced 7 GiB graph capture costs more residency than it earns | [`ksweep/`](results/2026-09-12-v12-1M-k5-off60-util97/ksweep/), [vllm#56626](https://github.com/vllm-project/vllm/issues/56626) |
| where the step goes | MoE expert streaming 64% of GPU time at C1, 88% at C8; Grace fetch ≈ 9 ms of a 24 ms step | [`profile/`](results/2026-09-12-v12-1M-k5-off60-util97/profile/), [`routing/`](results/2026-09-12-v12-1M-k5-off60-util97/routing/) |
| host-side (THP, unpinned, rust frontend, language-model-only) | nothing adopted | [`failure-ledger.md`](research/failure-ledger.md) |

Not scheduled: a cuDNN discrete-mode MoE backend (per-expert pointers; the only way to cash the routing skew; 1–2 weeks), `--async-scheduling` (unmeasured; zero memory; likely small at C1), KV-dtype audit (capacity, not speed). Not pursued: REAP pruning, EGM/two-tensor row-map. The **batch-size K-schedule** became v13 on 2026-09-14 (+26–35% at C4–C16, C1 flat). `--async-scheduling` measured 2026-09-14: wash. The rebase probe onto `dsv41-optimized` is deprioritized by the depth map (decode flat to 425K; attention megakernels would not move throughput here).

Operating it: `docker update --restart unless-stopped dsv41-vllm-v13-1M-ksched-BOUND-REF`; health is `GET /v1/models` on the lane port; hot restart ~4 min, cold ~8 with the seeded autotune cache. Previous bound configs stay as stopped containers for rollback.

## What this runs

[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) @ `df42c109`, **as shipped** — 510 GB of MXFP4 routed experts, FP8 Engram table, FP8 attention. Nothing is re-quantized. It does not fit the ~250 GiB of HBM the GB300 exposes, so this recipe puts 60 GiB of expert weights (v12; v11 used 70) and the 189 GiB Engram table in Grace LPDDR5X and lets the GPU read them over NVLink-C2C through vLLM's UVA offload backend. Full native context (1,048,576 tokens) stays on. The in-checkpoint DSpark drafter runs at k=5.

The result is a single-box agent lane: prose decode is C2C-bound at ~89 tok/s, but the text agents actually emit — shell, code, tool-call JSON — runs 140–160 tok/s because the drafter accepts 64–91% of its guesses there. The number to quote is the one for your workload.

## Hardware

Profile: [`hardware/dgx-station-gb300.yaml`](../../../hardware/dgx-station-gb300.yaml) · snapshot: [`results/2026-09-10-v11-1M-k5-lpt6144/system.json`](results/2026-09-10-v11-1M-k5-lpt6144/system.json)

| | observed |
|---|---|
| GPU | NVIDIA GB300, **250.7 GiB visible** (nvidia-smi 256,703 MiB; spec says 288) |
| Host | Grace 72× Neoverse-V2, 494.5 GiB LPDDR5X, NVLink-C2C |
| OS / kernel / driver | Ubuntu 24.04.4 · 6.17.0-1032-nvidia-64k · 595.84 · CUDA 13.0 in-container |
| Storage | **checkpoint on local NVMe** (RAID0 2× SN850X, XFS). From a CIFS share the load is 10× slower and the boot is 92 min. |
| Do not | set `CUDA_VISIBLE_DEVICES` — DGX OS already hides the display GPU |

## Software

| | pin |
|---|---|
| Image | `vllm/vllm-openai:deepseekv41-flash-0909` @ `sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d` (upstream day-0 build, 2026-09-10) |
| vLLM | `0.1.dev20904+g179dd0fa9` |
| Libraries | torch 2.13.0+cu130 · transformers 5.17.0 · flashinfer 0.6.18 · triton 3.7.1 |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` @ `df42c109f1defefcbfcedbe7d905718a12266e40`, 510,313,343,553 bytes, byte-verified |

### Image pin is a moving target — rebase candidate identified (2026-09-13)

This image was built from vLLM staging branch `dsv41-feat` @ `e47aa780`, which no longer exists
under that name: it was replaced by **`dsv41-optimized`** (HEAD is the same `e47aa780`,
2026-09-10). DeepSeek-V4.1-Flash reached **vLLM main** the same day (#56228, then **#56214
"[Model] Support DeepSeek-V4.1-Flash" merged 2026-09-11**), so main is now *ahead* of this pin
on model support while the remaining perf kernels stay staged on `dsv41-optimized`.

Not yet tested here. The candidate is `vllm/vllm-openai:nightly-2671fedf…` (2026-09-13 06:15Z;
cu129 sibling 06:27Z), which carries #56228/#56214 plus DeepSelect top-k (#56464) and the Triton
input-metadata fusion (#56562). It predates #56512 (Engram prefetch/DP-shard) and #56682, so
those want the next nightly. **v0.29.0 will not work — it predates the DSV4.1 merge.** All flags
in this recipe still map to current mainline names; `--engram-config '{"cpu_offload":true}'`
becomes redundant once #56512 is in the image (CPU offload becomes the default there).

Everything above is measured against the pinned image and stands until a rebase is run
same-window against the live `-BOUND-REF`.

Full lock: [`results/…/software-lock.txt`](results/2026-09-10-v11-1M-k5-lpt6144/software-lock.txt) · exact command: [`launch-command.txt`](results/2026-09-10-v11-1M-k5-lpt6144/launch-command.txt).

## Launch

```bash
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40 \
TAG=v13-1M-ksched OFFGB=60 UTIL=0.97 SEQS=16 SPEC=dspark:5 KSCHED='[[1,2,5],[3,16,1]]' CTX=1048576 \
EXTRA='--long-prefill-token-threshold 6144' \
bash scripts/launch-dsv41-vllm.sh
# ~8 min cold with a seeded autotune cache, ~4 min hot restart. Then:
bash scripts/smoke_vllm.sh
```

The server command inside the container:

```
vllm serve /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 60 \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 1048576 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,16,1]]}' \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 --port 30006
```

Flags that matter, and why:

- **`--cpu-offload-gb 60` with `--gpu-memory-utilization 0.97`** is v12's notch. ~~60 → −2.61 GiB at 1M+DSpark~~ — that was measured at util 0.94. At 0.97 it fits: KV 4.89 GiB = 2.33M tokens = 2.2 concurrent 1M requests, and the 12.7 GiB of experts that move back into HBM buy +12% single-stream decode in a same-window comparison (89.2 vs 79.5). 40 still does not fit even at 131K. The fetch tax is ~4 ms of an ~11 ms step (~36%), measured against an all-HBM two-Station run; util alone moves nothing (at 70 it only grew KV). If you need more than two concurrent 1M contexts, use the v11 flags (`OFFGB=70 UTIL=0.94`, 82 tok/s, 4.5× at 1M).
- **`--max-model-len 1048576`** costs 5.5 GiB of KV pool relative to 131K and nothing in decode speed. There is no reason to run this model small.
- **`--speculative-config dspark k=5` + batch-size schedule `[[1,2,5],[3,16,1]]` (v13).** The drafter is in the checkpoint (`mtp.*`). At one or two streams k=5 wins agent text by a third over k=1 and loses prose by 8%; at three or more streams k=1 wins every point by 23–33% because a k=5 verify window multiplies unique experts fetched per step 3.7× and the batch already fills the step. The schedule takes each where it wins: C1 unchanged, C4–C16 +26–35%. If your lane is single-stream prose only, static k=3 (~+10% prose) is still a fair choice.
- **`--long-prefill-token-threshold 6144`** is what makes one lane serve mixed traffic. Without it a 480K prefill starves a short request to 18 s TTFT (vLLM [#51454](https://github.com/vllm-project/vllm/issues/51454)); with it, 0.9–1.1 s, and the long prompt pays +16%. The older `--max-num-partial-prefills` flags are gone from this vLLM.
- **`--tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice`** — without these, agent turns come back as narrated text.

### The autotune trap

FlashInfer autotunes the MXFP4 MoE kernels and caches the result under a hash of the engine config. **Every** change to `--max-num-seqs`, the speculative config, or `--max-model-len` produces a new hash, an empty directory, and a **74-minute** retune (py-spy: `trtllm_fp4_block_scale_moe` at 39 ms per candidate × thousands). Mount `/root/.cache/vllm` persistently, and when you change config, copy `autotune_configs.json` from a previous hash dir into the new one **before** the tuner reads it — the window is roughly 60 s after `Graph capturing finished`. The log receipt you want is `Loaded 210 configs`; if you see `Autotuning process starts` and then nothing for a minute, you missed and should kill and relaunch. We missed it four times in one day. This recipe's hash is `08c89d94`.

## Verify

| gate | how | last pass |
|---|---|---|
| schema | `scripts/check_recipe.py` | 2026-09-11 |
| digest | image sha256 matches; `/model` is the `df42c109` revision, byte-verified | 2026-09-11 |
| health | `/v1/models` 200; log shows `Available KV cache memory: 10.05 GiB`, `Loaded 210 configs`; `smoke_vllm.sh` 5/5 | 2026-09-11 |
| quality | smoke (arith 323, count 1–60, prose, parsed `tool_call`, thinking→`36`); **Hermes harness 10/10** tool calls with correct answers and side effects ([harness-summary.md](results/2026-09-12-v12-1M-k5-off60-util97/harness-summary.md) (v12 re-run 2026-09-12, 10/10)). No requant, so no divergence gate. | 2026-09-11 |
| performance | `knee.sh` C1 within 5% of 89.2 tok/s warm vs a same-window v11 control (79.5) | 2026-09-12 |

**On instruments.** `knee.sh` runs each concurrency twice and the two runs agree within ~1%; it is the decision metric. `replay.py` replays 24 real agent turns and is useful as a workload-shaped smoke test, but at temperature 0 it swings **±17 tok/s** run to run (batched MoE + speculative decode are not bit-deterministic). We published an overnight "win" on it and retracted it the next morning when the knee said the opposite. Bench on the tight instrument.

## Results

### v13 — v12 + `num_speculative_tokens_per_batch_size=[[1,2,5],[3,16,1]]` (current)

Run [`2026-09-14-round3-ksched-depth`](results/2026-09-14-round3-ksched-depth/) · raw: [`throughput.csv`](results/2026-09-14-round3-ksched-depth/throughput.csv) · two same-window pairs against the v12 container, 18:02 and 18:27 CDT.

**Decode (knee, prose prompts; mean of both pairs, v12 control mean in parentheses)**

| C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|
| **90.7** (90.6) | 127.2 (127.2) | **223.4** (175.9) | **319.1** (252.6) | **339.3** (266.5) | **429.0** (318.8) |

C1 fixture classes and acceptance are the v12 numbers (the schedule is k=5 there). Decode vs prompt depth on the same boot: 124 / 117 / 114 / — / 122 tok/s at 6.5K / 53K / 106K / 212K / 425K — flat. Memory picture identical to v12 (hash 9ac7b387 hit, KV 4.87 GiB).

### v12 — `OFFGB=60 UTIL=0.97` (superseded by v13, same flags minus the schedule)

Run [`2026-09-12-v12-1M-k5-off60-util97`](results/2026-09-12-v12-1M-k5-off60-util97/) · raw: [`throughput.csv`](results/2026-09-12-v12-1M-k5-off60-util97/throughput.csv) · how it was chosen: [`night-two-ledger.md`](results/2026-09-12-v12-1M-k5-off60-util97/night-two-ledger.md) · warm, on-box, measured against a same-window v11 control boot (79.5 C1) because the reference drifts ~3% day to day.

**Decode (knee, prose prompts, DSpark k=5)**

| C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|
| **89.2** | 124.9 | 173.4 | 233.7 | 270.8 | **311.4** |

**Decode by content class (C1, `agent_fixture.sh`)** — every class moved +10–12% with acceptance unchanged, which is what a pure bandwidth change looks like:

| class | tok/s | DSpark accept | tok/step |
|---|---|---|---|
| shell_ops | 160.3 | 90.8% | 4.54 |
| code | 153.7 | 63.8% | 3.19 |
| tool_json | 141.9 | 87.8% | 4.39 |
| structured | 122.7 | 49.5% | 2.48 |
| prose | 97.7 | 29.8% | 1.49 |

Cost: KV 4.89 GiB = 2.33M tokens = **2.2 concurrent full-1M requests** (v11: 4.5). Host 432/494 GiB. First boot on the new flags pays the 74-minute autotune (new hash `9ac7b387`); after that ~8 min.

Prefill, mixed-traffic and harness numbers were re-measured on v12 on 2026-09-12 (prefill 207K in 11.4 s, starve shorts 0.8–1.1 s under a 480K prefill, Hermes harness 10/10) and did not move — they are not offload-bound. See [harness-summary.md](results/2026-09-12-v12-1M-k5-off60-util97/harness-summary.md).

**Where the decode step goes** (torch profiler, k=5): MoE expert streaming is 64% of GPU time at C1 and 88% at C8; attention 17%/6%. A k=5 verify window touches 3.7× the unique experts of one token (router property, ±3% across categories), a quarter of them in Grace — which is why k=5 is +100% on shell and −11% on prose vs k=0. Details: [profile/](results/2026-09-12-v12-1M-k5-off60-util97/profile/README.md), [routing/](results/2026-09-12-v12-1M-k5-off60-util97/routing/README.md). **Speculation depth (k sweep + adaptive, 2026-09-12 morning):** static k=1 wins prose (+8%) and every multi-stream point (+23–33% C1–C16) but loses the agent categories by ~a third; DSpark adaptive verification (V2 runner, confidence head) lands between them with no losing category but misses the promotion bar, partly because the V2 runner's graph capture costs ~6 GiB of residency. v12 static k=5 stays the reference. Table and raw JSON in [ksweep/](results/2026-09-12-v12-1M-k5-off60-util97/ksweep/README.md). **Follow-up (2026-09-14):** replacing the boot-profiled verify-cost table with a running median of real verify-step times (env-gated patch, two same-window pairs) confirmed the boot table under-prices verify at decode sizes by 11–40%, but the controller only gains C1 +3.3% / C8 +1.7% and loses C2/C4 (−6/−8%); tool-JSON +15% is the one consistent mover. Not adopted. The missing term is unique experts per step, not milliseconds per token. [2026-09-14-onlinevc-adaptive-off66/](results/2026-09-14-onlinevc-adaptive-off66/README.md).

**Why the offload lever stops here:** three spikes (VMM/EGM mixed backing, managed memory) show that on GB300 the only fast GPU→Grace read path is the pinned/ATS one the UVA offloader already uses (~340 GB/s); every path that allows per-row placement runs at ~90 GB/s, and oversubscribed managed memory settles at 155 GB/s. Details and scripts in [results/…/spikes/](results/2026-09-12-v12-1M-k5-off60-util97/spikes/README.md).

### v11 — `OFFGB=70 UTIL=0.94` (baseline; use if you need >2 concurrent 1M contexts)

Run [`2026-09-10-v11-1M-k5-lpt6144`](results/2026-09-10-v11-1M-k5-lpt6144/) · raw: [`throughput.csv`](results/2026-09-10-v11-1M-k5-lpt6144/throughput.csv) · warm, on-box.

**Decode (knee, prose prompts, DSpark k=5)**

| C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|
| 82.1 | 119.2 | 164.5 | 237.0 | 251.6 | 286.5 |

Same-night re-measurement of this config on 2026-09-11 gave 79.5 / 229.4 / 281.7 — the ~3% drift band.

**Decode by content class (C1, `agent_fixture.sh`)**

| class | tok/s | DSpark accept | tok/step |
|---|---|---|---|
| shell_ops | 150.0 | 90.8% | 4.54 |
| code | 145.6 | 63.8% | 3.19 |
| tool_json | 131 | 87.8% | 4.39 |
| structured | 115.8 | 49.5% | 2.48 |
| prose | 92.8 | 29.8% | 1.49 |

Real Hermes transcripts (24 turns): ~49% acceptance, real tool-call turns 138–171 tok/s.

**Prefill (cold KV, thinking off)**

| tokens | TTFT | tok/s |
|---|---|---|
| 6,538 | 0.37 s | 17,900 |
| 103,814 | 5.38 s | 19,300 |
| 207K | 11.3 s | 18,300 |
| 415K | 24.9 s | 16,700 |
| **972,435** | **85.0 s** | 11,400 |

Zero preemptions at 972K; HBM peaked at 243 GiB. Warm prefix on an 8.8K-token system prompt: 0.665 → 0.360 s.

**Mixed traffic (`starve.py`)**: four short requests fired during a 480K prefill see 0.89–1.09 s TTFT (18.2 / 13.2 / 8.2 / 3.2 s without `--long-prefill-token-threshold`).

**Boot**: 8 min cold from local NVMe with a seeded autotune cache; 4 min hot `docker start`. From CIFS with a retune: 92 min.

## Known limits

See `limits:` in [`recipe.yaml`](recipe.yaml). The short version: decode is C2C + kernel bound and the offloaded-expert byte count is the lever (~0.76 tok/s per GiB moved into HBM; v12 spends the last HBM headroom on it and pays in KV); quote the tok/s for your content class, not the best one; the autotune hash trap is real; `reasoning_effort: medium` is rejected by the template (use `low|high|xhigh|max`); ~110 GB of host shmem is unaccounted for and the host has no room for a second big model.

## What didn't work

[`research/failure-ledger.md`](research/failure-ledger.md) — SGLang at 3.3 tok/s, four vLLM UVA failures before the first bind, the offload bracket, the autotune misses, a first overnight of decode experiments (THP, unpinned host memory, `--language-model-only`, Rust frontend) that produced one real lesson about instruments and zero adopted flags, and a second night (k-sweep, util-only, off60) that produced v12 — see [`night-two-ledger.md`](results/2026-09-12-v12-1M-k5-off60-util97/night-two-ledger.md).

## Rollback

Every bound config is kept as a stopped container `dsv41-vllm-<tag>`. `docker stop` the lane, `docker start` the previous one; ~4 min. Production surfaces on other ports are never part of this lane.

## Credits

DeepSeek for the model and the in-checkpoint DSpark drafter. The vLLM team for the day-0 image, UVA offload backend, and DeepSeek-V4.1 support. Community data points that shaped decisions: 0xSero and Tech2Wild (4× RTX PRO 6000 and DGX Spark builds), Fraser Price (`dspark-vllm`), LMSYS (Engram huge-page finding). Blog write-up: [al-engr.com](https://al-engr.com/gb300-deepseek-flash-41-testing.html).

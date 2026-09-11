# DeepSeek-V4.1-Flash at 1M context on one DGX Station GB300

**Status: verified** (2026-09-10/11) · 82 tok/s single-stream prose · 130–150 tok/s on agent/code text · 287 agg tok/s at C16 · 972K-token prompt prefilled in 85 s · Hermes tool-calling 10/10

## What this runs

[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) @ `df42c109`, **as shipped** — 510 GB of MXFP4 routed experts, FP8 Engram table, FP8 attention. Nothing is re-quantized. It does not fit the ~250 GiB of HBM the GB300 exposes, so this recipe puts 70 GiB of expert weights and the 189 GiB Engram table in Grace LPDDR5X and lets the GPU read them over NVLink-C2C through vLLM's UVA offload backend. Full native context (1,048,576 tokens) stays on. The in-checkpoint DSpark drafter runs at k=5.

The result is a single-box agent lane: prose decode is C2C-bound at ~82 tok/s, but the text agents actually emit — shell, code, tool-call JSON — runs 130–150 tok/s because the drafter accepts 64–91% of its guesses there. The number to quote is the one for your workload.

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

Full lock: [`results/…/software-lock.txt`](results/2026-09-10-v11-1M-k5-lpt6144/software-lock.txt) · exact command: [`launch-command.txt`](results/2026-09-10-v11-1M-k5-lpt6144/launch-command.txt).

## Launch

```bash
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40 \
TAG=v11-1M-k5-lpt6144 OFFGB=70 UTIL=0.94 SEQS=16 SPEC=dspark:5 CTX=1048576 \
EXTRA='--long-prefill-token-threshold 6144' \
bash scripts/launch-dsv41-vllm.sh
# ~8 min cold with a seeded autotune cache, ~4 min hot restart. Then:
bash scripts/smoke_vllm.sh
```

The server command inside the container:

```
vllm serve /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 70 \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 1048576 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.94 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 --port 30006
```

Flags that matter, and why:

- **`--cpu-offload-gb 70`** is the notch, not a guess. 40 → KV −6.96 GiB even at 131K. 60 → fits at 131K without spec (12.03 GiB) but −2.61 GiB at 1M+DSpark. 70 → 10.05 GiB KV = 4.79M tokens = 4.5 concurrent 1M requests. Halving offload does not halve decode time; C2C traffic is ~30% of the token budget, kernels are the rest.
- **`--max-model-len 1048576`** costs 5.5 GiB of KV pool relative to 131K and nothing in decode speed. There is no reason to run this model small.
- **`--speculative-config dspark k=5`.** The drafter is in the checkpoint (`mtp.*`). k=5 wins agent text by ~10% over k=3; k=3 wins prose by ~10% with 70% acceptance vs 60%. Pick for your traffic.
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
| quality | smoke (arith 323, count 1–60, prose, parsed `tool_call`, thinking→`36`); **Hermes harness 10/10** tool calls with correct answers and side effects ([harness-summary.md](results/2026-09-10-v11-1M-k5-lpt6144/harness-summary.md)). No requant, so no divergence gate. | 2026-09-11 |
| performance | `knee.sh` C1 within 5% of 82.1 tok/s warm (re-check 80.1) | 2026-09-11 |

**On instruments.** `knee.sh` runs each concurrency twice and the two runs agree within ~1%; it is the decision metric. `replay.py` replays 24 real agent turns and is useful as a workload-shaped smoke test, but at temperature 0 it swings **±17 tok/s** run to run (batched MoE + speculative decode are not bit-deterministic). We published an overnight "win" on it and retracted it the next morning when the knee said the opposite. Bench on the tight instrument.

## Results

Run [`2026-09-10-v11-1M-k5-lpt6144`](results/2026-09-10-v11-1M-k5-lpt6144/) · raw: [`throughput.csv`](results/2026-09-10-v11-1M-k5-lpt6144/throughput.csv) · warm, on-box.

**Decode (knee, prose prompts, DSpark k=5)**

| C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|
| 82.1 | 119.2 | 164.5 | 237.0 | 251.6 | 286.5 |

**Decode by content class (C1, `agent_fixture.sh`)** — this is where DSpark earns its keep:

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

See `limits:` in [`recipe.yaml`](recipe.yaml). The short version: decode is C2C + kernel bound so offload tuning is nearly exhausted at 70; quote the tok/s for your content class, not the best one; the autotune hash trap is real; `reasoning_effort: medium` is rejected by the template (use `low|high|xhigh|max`); ~110 GB of host shmem is unaccounted for and the host has no room for a second big model.

## What didn't work

[`research/failure-ledger.md`](research/failure-ledger.md) — SGLang at 3.3 tok/s, four vLLM UVA failures before the first bind, the offload bracket, the autotune misses, and a full overnight of decode experiments (THP, unpinned host memory, `--language-model-only`, Rust frontend) that produced one real lesson about instruments and zero adopted flags.

## Rollback

Every bound config is kept as a stopped container `dsv41-vllm-<tag>`. `docker stop` the lane, `docker start` the previous one; ~4 min. Production surfaces on other ports are never part of this lane.

## Credits

DeepSeek for the model and the in-checkpoint DSpark drafter. The vLLM team for the day-0 image, UVA offload backend, and DeepSeek-V4.1 support. Community data points that shaped decisions: 0xSero and Tech2Wild (4× RTX PRO 6000 and DGX Spark builds), Fraser Price (`dspark-vllm`), LMSYS (Engram huge-page finding). Blog write-up: [al-engr.com](https://al-engr.com/gb300-deepseek-flash-41-testing.html).

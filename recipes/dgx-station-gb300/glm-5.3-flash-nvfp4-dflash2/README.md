# GLM-5.3-Flash NVFP4 + DFlash2 on one DGX Station GB300

**Status: verified** (round 2, September 11, 2026) · **252 tok/s** single-stream · **743 agg at C8** (DFlash2) · **2,214 agg / 50 per user at C48** and **4,025 agg / 34 per user at C128** (autoregressive) · 1M context · quality gated by teacher-forced divergence vs the FP8 original

Round 1 (September 1) is preserved below; two of its conclusions were wrong and are corrected here.

![Serving topology](diagrams/topology.svg)

## What this runs

GLM-5.3-Flash (320B MoE, 18B active; hybrid KDA linear attention + DeepSeek sparse attention + MLA) in **NVIDIA's first-party NVFP4** (`nvidia/GLM-5.3-Flash-NVFP4`), served by **released-track SGLang nightly** with the **DFlash2** block-diffusion draft at block size 7. One GPU, nothing offloaded, 1M context.

Two configs from the same weights and image:

- **Interactive / agent lane** (`SPEC=dflash`, 48 KDA slots): 252 tok/s single-stream, ~740 agg at C8, 576K-token KV pool. This is the daily driver.
- **Batch lane** (`SPEC=none`, 640 bf16 KDA slots): 48 users at 50 tok/s each, 128 users at 34 tok/s each, sub-1.5 s TTFT p95. 201K-token KV pool.

The knob between them is `--max-mamba-cache-size`: SGLang reserves 5 KDA state slots per running request, and the default budget caps the server at **7 concurrent requests** regardless of GPU headroom. Round 1 misread that cap as "DFlash2 loses at C16". It does not — see Results.

## Hardware

Profile: [`hardware/dgx-station-gb300.yaml`](../../../hardware/dgx-station-gb300.yaml) · snapshot: [`results/2026-09-01-flash-dflash2/system.json`](results/2026-09-01-flash-dflash2/system.json)

| | observed |
|---|---|
| GPU | NVIDIA GB300, **250.7 GiB visible** (spec says 288 GB; nvidia-smi says 256,703 MiB) |
| Host | Grace 72× Neoverse-V2, 494.5 GiB LPDDR5X, NVLink-C2C |
| OS / kernel / driver / CUDA | Ubuntu 24.04.4 · 6.17.0-1031-nvidia-64k · 595.84 · 13.2 |
| Do not | set `CUDA_VISIBLE_DEVICES` — DGX OS's docker config already hides the display GPU; setting it makes the container see 0 GPUs |

## Software

| | pin |
|---|---|
| Image | `glm53-nvfp4-sglang:gb300-v2` @ `sha256:f710fd42cf749cbc6c3e1799244b5bdefd3b8f34d978aa810a9c1f97f86b143a` — a frozen `docker commit`; rebuild with [`scripts/build-image.sh`](scripts/build-image.sh) |
| Base | `lmsysorg/sglang:v0.5.18` @ `sha256:9e148f5a…` (2026-08-21) |
| SGLang | 0.5.18 + [PR #36507](https://github.com/sgl-project/sglang/pull/36507) `glm-5.3-flash-support` @ `0a6b5d8c` (carries [#36708](https://github.com/sgl-project/sglang/pull/36708) DFlash hidden-state adapter). Released SGLang ≤ 0.5.18 cannot load this architecture — see [`research/failure-ledger.md`](research/failure-ledger.md) for the ten walls between "hardware works" and "first token". |
| Libraries | torch 2.13.0+cu130 · transformers 5.16.1 · triton 3.7.1 · cuDNN 9.14 |
| Target | [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) @ `aa28e1f54130286c95fee10d0705c74ce8743734` (182 GB) — mount the `original/` snapshot dir, not the wrapper |
| Draft | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `7d74cdd881ed7e32c31175984a67823127b66cfe` (2.2 GB) |

Full lock: [`results/2026-09-01-flash-dflash2/software-lock.txt`](results/2026-09-01-flash-dflash2/software-lock.txt).

## Launch

```bash
MODEL_DIR=/path/to/GLM-5.3-Flash-NVFP4/aa28e1f5…/original \
DRAFT_DIR=/path/to/GLM-5.3-Flash-DFlash2 \
API_KEY=… bash scripts/launch-dflash2.sh
# wait for /health, then — not optional —
API_KEY=… bash scripts/warmup.sh
```

The server command inside the container ([`scripts/launch-dflash2.sh`](scripts/launch-dflash2.sh)):

```
python3 -m sglang.launch_server --model-path /model --host 0.0.0.0 --port 30000 \
  --quantization modelopt_fp4 --trust-remote-code \
  --api-key $API_KEY --served-model-name glm-5.3-flash \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --cuda-graph-max-bs 32 --max-running-requests 32 \
  --max-prefill-tokens 8192 --chunked-prefill-size 8192 \
  --speculative-algorithm DFLASH --speculative-draft-model-path /draft \
  --speculative-draft-attention-backend fa4
```

Flags that matter:
- `--tool-call-parser glm47 --reasoning-parser glm45` — **required for any agent or tool use.** Without them the server still answers, but `<tool_call>` markup comes back as plain text and thinking leaks into `content`. Plain-chat smoke tests pass while every agent turn silently dies.
- `--speculative-draft-attention-backend fa4` forces the draft's KV to bf16; the target's KV stays FP8. Block size 8 comes from the draft's `dflash_config`; EAGLE-style `--speculative-num-steps`/`topk` do not apply.
- For C16+ batch work, drop the three `--speculative-*` lines (`scripts/launch-ar.sh`).

## Verify

| gate | how | last pass |
|---|---|---|
| schema | `scripts/check_recipe.py` | 2026-09-06 |
| digest | `docker image inspect` matches; model dirs are the pinned revisions | 2026-09-06 |
| health | `/health` 200; `warmup.sh` completes C1/4/8/16/32 | 2026-09-02 |
| quality | coherent output on the prose/code set; DFlash2 accept length 2.77–3.95 on prose; tool calls parse with the parsers on. **No teacher-forced divergence gate was run for this recipe** — see Known limits. | 2026-09-02 |
| performance | C1 median-of-3 within 5% of 234.2 tok/s after warmup | 2026-09-02 |

**Warm up or your benchmarks lie.** The first request at each new batch shape after a restart pays up to 30 s of kernel autotune (CUDA graphs are on the whole time — it's per-shape JIT). The same applies to prompt-length classes: the first 8k/32k/64k prompt after restart pays ~16 s; warm, those prefill in 0.3–2 s. Our own day-one "concurrency cliff" (220 agg tok/s, 15 s TTFT at C8+) was this artifact. Bench warm or bench wrong.

## Results

### Round 2 — run [`2026-09-11-nvidia-nvfp4-b7-1M`](results/2026-09-11-nvidia-nvfp4-b7-1M/) · raw [`throughput.csv`](results/2026-09-11-nvidia-nvfp4-b7-1M/throughput.csv)

Instrument: every batch shape is hit once and **discarded** (first hit after boot is kernel autotune), then 3 measured reps; spread reported. Single-stream uses the round-1 "recipe method" (history essay, 512 tokens) so numbers are comparable.

**Which NVFP4?** Teacher-forced on 40 prompts × 128 greedy tokens from the **FP8 original** (`zai-org/GLM-5.3-Flash`, served via vLLM UVA offload as an oracle):

| quant | mean \|Δlogp\| vs FP8 | median | p95 | argmax ≠ FP8 |
|---|---|---|---|---|
| **nvidia** (attention + shared experts unquantized) | **0.136** | 0.022 | 0.65 | **15.6%** |
| LibertAIDAI (round-1 weights) | 0.147 | 0.026 | 0.69 | 15.9% |

nvidia is closer on 27/40 prompts; bootstrap 95% CI on the gap is [0.004, 0.017] — real, small. Both quants flip ~16% of the original's greedy tokens, and 10.3% of the time they flip to the *same* token: that is the NVFP4 tax on routed experts, and no quantizer's recipe avoids it. Speed is a wash (238 vs 235.5 C1 at block 8). Pick nvidia for provenance and the measured edge; do not expect it to feel different.

**Single-stream (C1), DFlash2 block 7, 1M ctx:**

| prompt class | tok/s | DFlash2 accept rate |
|---|---|---|
| history essay 512 (recipe method) | **252.5** | — |
| code 400 | 253.4 | 0.41 |
| shell-ops 200 | 254.1 | 0.39 |
| math (step-by-step) | 291 | 0.61 |
| prose 300 | 174.7 | 0.29 |
| plain autoregressive, any | 138.8 | — |

Block 7 (drafter block 8 minus the target token) vs the round-1 default of 8: +18 tok/s on the recipe method, +35 on code, same acceptance, 10% more KV pool. Acceptance is the lever left: 0.39 mean here vs 0.54–0.67 reported by ebfio on a vLLM stack.

**Concurrency, 2.4K-token prose prompt → 384 tokens, per-user speed and TTFT:**

| users | AR (640 bf16 slots) | DFlash2 b7 (120 slots) |
|---|---|---|
| 1 | 142 tok/s · 0.2 s | **210** · 0.2 s |
| 8 | 668 agg · 90/user · 0.3 s | **743 agg · 106/user** · 0.4 s |
| 16 | 1,075 · 72 · 0.5 s | **1,180 · 86** · 0.4 s |
| 24 | ~1,350 · ~62 | **1,482 · 72** · 0.5 s ← DFlash2 slot cap |
| 32 | 1,648 · 55 · 0.6 s | (cap) |
| **48** | **2,214 · 50 · 0.8 s** | (cap) |
| 96 | 3,406 · 39 · 1.1 s | |
| **128** | **4,025 · 34 · 1.4 s** | |

DFlash2 wins per running request through at least C24. It loses on **memory**: each DFlash2 slot carries 7 draft-token intermediate KDA states in fp32 (`--mamba-ssm-dtype bfloat16` crashes the DFlash2 verify kernel on this image), so 120 slots cost 41 GB. Autoregressive with bf16 state fits 640 slots in 43 GB. That, not verification cost, is the crossover.

Hermes agent harness on the daily config: 10/10 tool-calling turns, 10/10 parsed tool calls (`scripts/tool_harness.py`).

### Round 1 — run [`2026-09-01-flash-dflash2`](results/2026-09-01-flash-dflash2/) · raw [`throughput.csv`](results/2026-09-01-flash-dflash2/throughput.csv)

LibertAIDAI NVFP4, frozen PR-branch image, DFlash2 block 8, default KDA budget (7 running). Warm steady state, 256-token generations.

| concurrency | AR (tuned) | **DFlash2** | winner |
|---|---|---|---|
| C1 | 135.5 tok/s | **234.2 tok/s** (median of 3) | DFlash2 +73% |
| C4 | 375.8 agg | **511.4 agg** | DFlash2 +36% |
| C8 | 647.6 agg | **825.2 agg** | DFlash2 +27% |
| C16 | **1051.2 agg** | 796.3 agg | ~~AR~~ — **slot cap, not AR; see round 2** |
| C32 | **1162.5 agg** | 835.8 agg | ~~AR~~ — same |
| TTFT, 9th request during 8 running | 0.18 s | 0.19 s | tie |

Cold prefill, DFlash2 config, warm shapes (nonce-defeated prefix cache): 6,625 tok 0.29 s (~22.5k tok/s) · 26,393 tok 1.10 s · 52,739 tok 2.01 s · 105,434 tok 4.00 s (~26.3k tok/s). Prefill is unchanged by round 2.

For reference, [catid/dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks) publishes 187.1 tok/s C1 (DFlash2) and 964.9 agg C16 (AR) on the same silicon with 8K prompts; the [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash) GB300 MTP row (277 C1) is measured with `SGLANG_SIMULATE_ACC_LEN=3`, i.e. simulated acceptance.

## Known limits

- **Concurrency is capped by KDA state slots, not compute.** Default budget → 7 running requests. Raise `--max-mamba-cache-size` (5 slots per request); bytes come out of the KV pool one-for-one. `no_buffer` strategy (3 slots/req) needs page_size 1 and does not work with this model's DSA paging.
- **DFlash2 above ~24 users** is a memory problem (fp32 intermediate states × 7 draft tokens per slot), not a speed problem. Use the AR variant for batch.
- **"FP8 KV" is half true.** Only the DSA indexer pool is fp8; the MLA KV stays bf16 with `--kv-cache-dtype fp8_e4m3` ([sglang#36830](https://github.com/sgl-project/sglang/issues/36830), `index_kpool=4`).
- **bf16 KDA state** (`--mamba-ssm-dtype bfloat16`) halves slot cost and is what makes 128 users fit — but SGLang documents it as output-shifting and it is not KL-gated here. Also incompatible with DFlash2 on this image.
- **Native MTP** is in the checkpoint (`num_nextn_predict_layers=1`) but SGLang's GLM-5.3 NextN path crashes on first request / misaccepts ([#37548](https://github.com/sgl-project/sglang/issues/37548), [#36829](https://github.com/sgl-project/sglang/issues/36829)). Not used.
- **Never enable HiCache** for agent traffic on this model: host-tier load-back drops tool calls ([#38031](https://github.com/sgl-project/sglang/issues/38031)).
- **Autotune tax on first hit** of every batch shape after restart (13 s TTFT at C16 observed). Warm every shape you serve; discard the first pass when measuring.
- **NVFP4 disagrees with the FP8 original on ~16% of tokens** whoever quantized it. If that matters, the FP8 original with UVA offload works on this box (~22 tok/s C1 as configured for scoring) — a different recipe.
- **Full GLM-5.3 in FP8 (704 GB) does not fit** one Station with SGLang CPU offload; see [`research/failure-ledger.md`](research/failure-ledger.md).

## Rollback

`docker rm -f glm53-flash`, then relaunch the previous checkpoint dir with the same image and flags — [`scripts/swap-model.sh`](scripts/swap-model.sh) does exactly that. ~5 min including warmup; the served model name stays constant so clients never notice.

## Credits

- [LibertAIDAI](https://huggingface.co/LibertAIDAI) — the NVFP4 quant
- [incoai](https://huggingface.co/incoai) — the DFlash2 draft model
- SGLang PR [#36507](https://github.com/sgl-project/sglang/pull/36507) / [#36708](https://github.com/sgl-project/sglang/pull/36708) authors — the model support that made this possible
- [catid](https://github.com/catid/dgx_station_benchmarks) — the public baseline we measured against

Longer write-up: [al-engr.com/gb300-glm-53-testing.html](https://al-engr.com/gb300-glm-53-testing.html). Origin repo: [jmeadlock/gb300-glm-flash-recipe](https://github.com/jmeadlock/gb300-glm-flash-recipe).

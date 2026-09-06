# GLM-5.3-Flash NVFP4 + DFlash2 on one DGX Station GB300

**Status: verified** (2026-09-01/02, re-pinned 2026-09-06) · 234 tok/s single-stream · 825 agg tok/s at C8 · sub-second cold prefill to 8k

![Serving topology](diagrams/topology.svg)

## What this runs

GLM-5.3-Flash (355B-class MoE) in NVFP4, served by SGLang with the **DFlash2** block-diffusion draft model for speculative decoding. Interactive-optimised: the biggest win is single-stream and C4–C8; above C16 plain autoregressive is faster and `scripts/launch-ar.sh` is the same recipe without the three `--speculative-*` flags.

The whole box is one GPU; nothing is offloaded. Model + draft + FP8 KV cache (2.72M-token capacity) fit in the ~250 GiB of HBM the machine actually exposes.

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

Run [`2026-09-01-flash-dflash2`](results/2026-09-01-flash-dflash2/) · raw: [`throughput.csv`](results/2026-09-01-flash-dflash2/throughput.csv) · warm steady state, 256-token generations, benched on-box (no network in path).

| concurrency | AR (tuned) | **DFlash2** | winner |
|---|---|---|---|
| C1 | 135.5 tok/s | **234.2 tok/s** (median of 3) | DFlash2 +73% |
| C4 | 375.8 agg | **511.4 agg** | DFlash2 +36% |
| C8 | 647.6 agg | **825.2 agg** | DFlash2 +27% |
| C16 | **1051.2 agg** | 796.3 agg | AR |
| C32 | **1162.5 agg** | 835.8 agg | AR |
| TTFT, 9th request during 8 running | 0.18 s | 0.19 s | tie |

Cold prefill, production DFlash2 config, warm shapes (nonce-defeated prefix cache):

| prompt tokens | TTFT | prefill rate |
|---|---|---|
| 6,625 | 0.29 s | ~22.5k tok/s |
| 26,393 | 1.10 s | ~24.1k tok/s |
| 52,739 | 2.01 s | ~26.2k tok/s |
| 105,434 | 4.00 s | ~26.3k tok/s |

Day-one untuned baseline: 141.8 tok/s C1. For reference, [catid/dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks) publishes 187.1 tok/s C1 (DFlash2) and 964.9 agg C16 (AR) on the same silicon; the gap is mostly the warmup discipline above.

Swapping a same-architecture NVFP4 checkpoint (e.g. a modified-weight variant) is a relaunch + warmup, ~5–8 min, with the stock drafter retained: measured 218.8 tok/s C1 (−6.5%) and accept length 2.69–3.69 on one such variant — the speculative win survives.

## Known limits

- **DFlash2 loses above C16.** Verification passes compete with batch decode; use `scripts/launch-ar.sh` for batch.
- **FP8 KV cache.** This recipe accepts SGLang's default FP8 target KV for `modelopt_fp4`. That is a quality tradeoff we took here. It is *not* what the GLM-5.3-big recipe on this same machine allows — quality bars are per recipe, and this one is looser.
- **Quality gate is output-judged, not measured.** No teacher-forced logprob divergence was run against an AR reference. DFlash2 is verified-lossless by construction, but "by construction" is not a measurement; a KL gate like recipe 2's would be stronger.
- **Autotune tax on first hit** of every batch shape and prompt-length class. `warmup.sh` covers batch shapes; long-prompt classes still pay once each.
- **Frozen PR-branch image.** Reproducible via `build-image.sh`, but the branch is not a release; when #36507 merges, re-verify on a released tag.
- **Full GLM-5.3 in FP8 (704 GB) does not fit** one Station with SGLang CPU offload — host shm peak exceeds the 744 GiB total. NVFP4 Flash is the proven single-Station model; see [`research/failure-ledger.md`](research/failure-ledger.md).

## Rollback

`docker rm -f glm53-flash`, then relaunch the previous checkpoint dir with the same image and flags — [`scripts/swap-model.sh`](scripts/swap-model.sh) does exactly that. ~5 min including warmup; the served model name stays constant so clients never notice.

## Credits

- [LibertAIDAI](https://huggingface.co/LibertAIDAI) — the NVFP4 quant
- [incoai](https://huggingface.co/incoai) — the DFlash2 draft model
- SGLang PR [#36507](https://github.com/sgl-project/sglang/pull/36507) / [#36708](https://github.com/sgl-project/sglang/pull/36708) authors — the model support that made this possible
- [catid](https://github.com/catid/dgx_station_benchmarks) — the public baseline we measured against

Longer write-up: [al-engr.com/gb300-glm-53-testing.html](https://al-engr.com/gb300-glm-53-testing.html). Origin repo: [jmeadlock/gb300-glm-flash-recipe](https://github.com/jmeadlock/gb300-glm-flash-recipe).

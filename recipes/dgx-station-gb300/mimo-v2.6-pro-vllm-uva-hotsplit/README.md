# MiMo-V2.6-Pro on one DGX Station GB300 — vLLM UVA expert offload + per-expert HBM residency ("hotsplit")

**Status: verified (v23, promoted 2026-09-24).** The residency hook is still out-of-tree. Previous status: experimental (2026-09-22).

## What this runs

The 527 GiB [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) checkpoint (1.02T total / 42B active, 384 routed experts × 69 MoE layers, native MXFP4 experts, revision `54b10491`) on one GPU that has 250 GiB of HBM:

- vLLM's UVA offloader reads 320 GiB of routed experts **in place** from Grace memory over NVLink-C2C (exact-size pinned, no H2D copy);
- Marlin MoE kernels (the auto-picked FlashInfer TRT-LLM MXFP4 backend's autotune crawled ~2 h with 320 GiB of experts in Grace; CORRECTED 2026-09-23: an earlier version blamed an sm_100a-only cubin, which was wrong — see vllm-project/vllm#58031);
- **hotsplit**: after load, each layer's experts are re-homed by *measured decode usage* — hot rows in HBM, cold rows in Grace — and the MoE runs as two Marlin calls with `expert_map`s. Stock `--cpu-offload-gb` decides residency by layer order (layers 1–48 in Grace, 49–69 in HBM); hotsplit decides it by what the router actually picks;
- 262,144-token context (v23), or 1,048,576 with the v24 variant below; reasoning and tool parsers on; no speculative decoding.

## Hardware

One NVIDIA DGX Station GB300: GB300 (250.7 GiB visible HBM), Grace with 494.5 GiB LPDDR5X, NVLink-C2C. Driver 595.91.07, Ubuntu 24.04.5, kernel 7.0.0-1019-nvidia-64k (`results/2026-09-22-hotsplit/system.json`). Checkpoint on local RAID0 NVMe. Host memory after the split: ~120 GiB free — no room for a second large model.

## Software

`vllm/vllm-openai:nightly-d05da62e9ccdf8e342b15bf6785d83224cc165af` (`sha256:f29125bc…`, vLLM 0.29.1rc1.dev422, torch 2.13.0+cu130, FlashInfer 0.6.18.post1). Nothing rebuilt; three bind-mounts:

| file | what |
|---|---|
| `patches/mimo_v2_hotsplit.py` → `models/mimo_v2.py` | stock + the vLLM #58142 fused-fp8-qkv pairing fix + truncation tolerance (#58184) + the hotsplit load hook |
| `patches/hotsplit.py` → `/w/hotsplit.py` | the split, two-bank MoE apply, optional graph-safe live routing counter |
| `patches/mimo_v2_omni.py` → `models/mimo_v2_omni.py` | DFlash interface forwarding; inert here (DFlash off) |

Ranking input: `patches/expert_hist_mix.json` (decode routing counts from 630 real agent turns + a prose histogram at 0.25 weight; the turns themselves are private).

## Launch

```bash
# on the Station: $WORKDIR holds patch/, vllm-cache/, live/, hotsplit.py
mkdir -p $WORKDIR/{patch,vllm-cache,live}
cp patches/mimo_v2_hotsplit.py patches/mimo_v2_omni.py $WORKDIR/patch/
cp patches/hotsplit.py patches/expert_hist_mix.json $WORKDIR/
WORKDIR=$WORKDIR MODEL=/models/MiMo-V2.6-Pro-RL COUNTS=/w/expert_hist_mix.json LIVE=1 \
  bash scripts/launch-hotsplit.sh v23 152.8 262144
```

Required: `--moe-backend marlin`, `VLLM_USE_DEEP_GEMM=0`, `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` (exact-size pinning; hotsplit OOMs the host without it), `--cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight`. **Do not** use `--load-format runai_streamer` on this model (zeroes the fused FP8 qkv_proj → confident garbage).

**1M context (v24):** `bash scripts/launch-hotsplit.sh v24 110 1048576`, which gives 110 GiB of hot experts and a KV pool of 1,209,097 tokens. It costs ~12% at short context (decode 33.0 vs 37.4 at 11.5K). This is the author's serving lane since September 23.

Variants: drop `152.8` for the stock-equal budget (KV 514K); unset `COUNTS` for the stock control; drop `LIVE=1` to remove the live counter (+0.3–0.6 tok/s).

## Verify

- `curl -fs http://localhost:30007/v1/models`; boot ~7 min.
- Log lines: `Using 'MARLIN' Mxfp4 MoE backend`, `Total CPU offloaded parameters: 321.75`, `hotsplit plan: 69 layers, budget 152.8 GiB (stock HBM experts 141.8 GiB), hot cells 8692/26496`, `hotsplit done in ~32s; HBM free 34.7 GiB`, `GPU KV cache size: 302,3xx tokens`.
- Long context: `python3 scripts/longctx_bench.py <tag> 65536 131072 196608 253952` (~45 min on an idle lane).
- `bash scripts/agent_fixture.sh warm` once and discard (first pass reads ~20% low), then `python3 scripts/ttft_bench.py <tag>` and `bash scripts/agent_fixture.sh <tag>`.
- Tool calling through Hermes: `bash scripts/harness_test.sh` then `python3 scripts/score_harness.py /tmp/harness-mimo26`.

## Verification (2026-09-24)

Promotion run `results/2026-09-24-promotion/`. The bars were written into `harness/protocol.yaml` (sha256 `84faecb9…`) before the run. **Control** = the kept v20 container: stock layer-order UVA placement. **Candidate** = the kept v23 container: hotsplit, 152.8 GiB hot. Both use the same image digest, weights and flags; only the `HOTSPLIT_*` environment differs. Each arm was booted, then identity-checked (`docker exec sha256sum` of both mounted patches, hotsplit plan line, KV size), warmed, and measured.

| bar | control (stock placement) | v23 hotsplit | result |
|---|--:|--:|---|
| Teacher-forced perplexity, 80,384 positions / 39 docs | 1.5081 | 1.5077 (−0.025%) | **pass** (≤ 0.5%) |
| Teacher-forced top-1 flips vs control | — | 1.23% | **pass** (≤ 2.0%) |
| BFCL dev (simple_python + multiple, 600) | 93.67% | 93.83% | **pass** (≥ control − 1) |
| BFCL held-out live sibling (1,311, first run on this model) | 78.49% | 78.18% | **pass** (≥ control − 1) |
| Errors, 3,822 requests | 0 | 0 | **pass** |

- **Noise floor.** I rebooted the *control* container and recaptured. The control against its own reboot flips 1.17% of top-1 tokens: mean |Δ| 0.025 vs 0.026, p99 0.41 vs 0.43. Hotsplit therefore adds about 0.05 pt of argmax movement over the lane's own run-to-run nondeterminism.
- **Greedy parity is not a usable signal on this lane.** 512-token greedy outputs were identical on 1/12 prompts control vs v23, and also 1/12 control vs its own reboot. One prompt (“Is 2³¹ − 1 prime?”) answers *No* on one control boot and *Yes* on the next (and on v23). This is vLLM batch/kernel nondeterminism at T=0, not hotsplit. Do not use byte-parity as a regression test for this recipe.
- BFCL was scored with our grader (the same as the DSV4.1 and GLM-5.3-Flash recipes): a same-grader A/B, not a leaderboard number. On held-out, 34 of 1,311 cases disagree between the two arms (19 control-only, 15 v23-only). v23's unsolved cases: wrong-arg-value 170, no-call 52, missing-arg 34, wrong-function 30, errors 0, truncated 0.
- Energy: v23 held-out ran at a mean of 372 W over 1,839 s (control 344 W over 2,390 s), which is **$0.028 per 1000 solved** at $0.15/kWh, energy only.
- **1M row (v24):** needles **3/3 at 1.04M prompt tokens** (10/50/90% depth). TTFT 1,101 s (944 tok/s prefill); decode after TTFT **25.7 tok/s** on a 1,035,393-token prompt.
- Reproduce: `scripts/finish-mimo-pro-2026-09-23.sh` (runner), `scripts/noise-floor.sh`, `scripts/verdict.py`.

## Results

Warm, same image and flags; only the hotsplit env differs.

| | v20 control (layer order) | v21 hotsplit, same HBM budget | **v22/v23 hotsplit, +11 GiB hot** |
|---|---|---|---|
| decode after TTFT, 11.5K-token prompt | 31.2 | 36.3 | **37.4** (v23 with live counter 36.9–37.1) |
| agent fixture: tool_json / shell / structured | 29.5 / 28.6 / 30.0 | 35.7 / 35.4 / 34.8 | **36.6 / 36.4 / 35.9** |
| agent fixture: code / prose | 30.3 / 30.3 | 31.9 / 31.6 | 32.6 / 32.7 |
| synthetic prose C1 (knee prompt) | 30.0 | 29.6–30.0 | 29.7–30.7 |
| C8 aggregate | — | 60.8 | **63.7** |
| prefill, 11.5K prompt | 1,259 tok/s | 1,334 | 1,376 |
| GPU KV cache | 490,466 tok | 513,612 | 302,368 (1.15× one 256K request) |

- **Long context (v23, one request, lane idle):** needle recall at 10/50/90% depth **12/12** at 65K / 131K / 196K / 254K prompt tokens. Decode after TTFT **34.0 / 34.3 / 33.3 / 33.8 tok/s**, and 36.9–37.1 at 11.5K. Prefill 1,350 → 1,262 tok/s, so TTFT is **201 s at 254K**. Bench: `scripts/longctx_bench.py`; rows, method, and two discarded contended rows: `results/2026-09-22-hotsplit/longctx.md`.
- **1M-context variant (v24):** decode 33.0 at 11.5K (−12%), tool_json 32.7, prefill 1,235 tok/s. Needles 3/3 at 65K and 3/3 at 524K; decode 28.6 tok/s and TTFT ~490 s at 524K. ~~**1M itself is not measured.**~~ **CORRECTED 2026-09-24:** measured: 3/3 needles at 1.04M, decode 25.7 tok/s, TTFT 18.4 min (see Verification). My pre-boot speed estimate (−7%) was wrong. Details: `results/2026-09-22-hotsplit/longctx.md`.
- Held-out decode traffic served from HBM at the stock byte budget: **30.4% → 62.2%** (ranked on 630 real agent turns, tested on 270 held out; oracle 62.5%; prefill-ranked 54.9%).
- Hermes harness on the v23 serving boot: **10/10 tool-call turns, 10/10 correct** (`results/2026-09-22-hotsplit/harness-summary.md`).
- Fidelity: 16-layer truncated Pro, stock vs hotsplit, greedy **6/6 token-identical**; residual drift ≤5.0e-4 relative by layer 15; max |Δlogprob| 0.067 (two partial sums change summation order — not bit-exact). ~~Full-depth teacher-forced Δlogprob and a public tool-call suite are **pending**.~~ Both done 2026-09-24; see Verification.
- Where the time goes (v21, 64 C1 decode tokens, torch profiler): 28.8 ms/token, GPU busy 96%. Marlin routed experts 17.3 ms — hot bank 3.4, **cold bank 14.0** (~39% of traffic, ~80% of expert time). Dense GEMMs 5.8 ms, already bandwidth-bound.

All receipts: `results/2026-09-22-hotsplit/` (`facts.md`, `receipts/`, `harness/`).

## Known limits

- **Out-of-tree** (verified 2026-09-24, but not a vLLM feature). Upstream design discussion: [vLLM RFC #57794](https://github.com/vllm-project/vllm/issues/57794#issuecomment-5785394491) (our data posted).
- **The hot list is workload-specific.** Agent-only ranking cost synthetic prose ~5%; mixing a prose histogram in at 0.25 fixed that. Re-rank for your traffic: offline with `scripts/expert_hist2.py` → `scripts/hist2_analyse.py` → `scripts/mixcounts.py`, or online with `LIVE=1` + `scripts/rerank-weekly.sh` (refuses below 50K decode tokens/layer; applies on the next boot; never restarts anything).
- **KV trade.** 60 of 70 layers are sliding-window, so each GiB of hot experts traded buys ~19K KV tokens. 152.8 GiB hot → 302K KV (37.4 tok/s); 141.8 → 514K (36.3); 110 → 1.21M (33.0).
- **No speculative decoding.** In-checkpoint DFlash k=7 (17.2 tok/s) and k=3 (24.5) both lost to k=0 (30.2) under offload on the pre-hotsplit lane.
- **Dense GEMMs are bandwidth-bound.** o_proj is BF16 in the checkpoint (13.1 GiB read per token, ~1.2 ms/token); quantizing it is a model change that needs a quality gate. Not done.
- Not bit-exact. Full-depth teacher-forced flips are 1.23% vs a 1.17% control-reboot floor, and long greedy outputs differ between boots with or without hotsplit.

## Rollback

`docker stop` the lane; `docker start` the kept v22 (same budget, no live counter) or v20 control (stock placement) container, or relaunch without `COUNTS`. Hot restart ~7 min.

## Upstream

- Exact-size pinned UVA memory: [vLLM PR #58185](https://github.com/vllm-project/vllm/pull/58185)
- `num_hidden_layers` truncation guard for MiMo-V2: [vLLM PR #58184](https://github.com/vllm-project/vllm/pull/58184)
- Fused fp8 qkv pairing across loader calls: [vLLM PR #58142](https://github.com/vllm-project/vllm/pull/58142) (vllmellm)
- SM103 MXFP4 autotune: [vLLM #58031](https://github.com/vllm-project/vllm/issues/58031)

Campaign write-up: https://al-engr.com/gb300-mimo-v26-pro-testing.html

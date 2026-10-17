# Full GLM-5.3 (744B, RadixArk NVFP4) on 8× DGX Spark — SGLang v0.5.22 TP8 over 200G RoCE + EAGLE/NextN, 512K context

**Status: experimental** (2026-10-17). GhettoRak's SGLang lane for the full model, and the first SGLang recipe on this
cluster that passes every gate the vLLM lane passes. Serving as the daily lane since 2026-10-17 16:01 CDT (James's call).
Served teacher-forced quality gate pending.

Sibling: [`dgx-spark-8x/glm-5.3-nvfp4-tp8`](../glm-5.3-nvfp4-tp8/) — same cluster, same checkpoint, vLLM fork with MTP k=3.

| | SGLang (this recipe) | vLLM sibling |
|---|---:|---:|
| Image | stock `lmsysorg/sglang:v0.5.22-cu130` | `gb10-glm-5.2:v19-vision` fork |
| KV pool / context | 654,336 / 524,288 | **720,384** / 524,288 |
| C1 code / prose / explanation (tok/s) | **36.2** / 28.7 / 23.7 | 35.7 / **29.9** / **25.8** |
| C4 aggregate (tok/s) | 59.5 (15.7–22.3 per user) | **61.7** (17.1–22.7) |
| Hermes 20-step tool chain, median wall | **31.7 s** (n=3) | 45.8 s (n=4) |
| Deepest needle passed | 514K (~640 tok/s prefill) | 514K (**~1,241**) |
| Draft | NextN as EAGLE, 4 steps, accept 3.05 | MTP k=3, accept 2.89 |
| Cold boot | **~11 min** | 17.5 min |
| 1M context | no (KV does not fit; no 4-bit KV path for DSA) | 754K passes with `nvfp4_ds_mla` |

Read: on raw single requests the two engines trade blows (SGLang wins code, vLLM wins prose/explanation and prefill).
On the agent-shaped workload — short turns over a growing conversation with a 7K-token system prompt in front — SGLang
is 30% faster, repeatably. That, plus a stock upstream image and a 6-minute-shorter boot, is why it is the lane now.

## What this runs

`RadixArk/GLM-5.3-NVFP4` @ `6e389189` (see the sibling for the checkpoint story). Stock SGLang v0.5.22, tensor parallel 8,
one rank per Spark over both 200G RoCE ports, `fp8_e4m3` DSA KV, `flashinfer_sparse_mla` decode, `flashinfer_cutlass` FP4
MoE (the only MoE backend that loads on SM121), in-checkpoint NextN as a 4-step EAGLE chain, CUDA graphs for decode
batch 1–4, FlashInfer autotune **off**.

## Launch

```bash
export GLM53_API_KEY=...            # the launcher reads the key from the env file it pushes; never commit it
export SPARK_HOSTS="spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8"   # rank order, rank 0 = head
bash scripts/cluster.sh push        # launcher + env to every node
SG_CTX=524288 SG_MAXRUN=4 SG_SPEC=4,1,5 SG_CHUNK=4096 SG_MEMFRAC=0.80 SG_GBS=4 \
SG_ENV="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False -e SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.01 -e NCCL_BUFFSIZE=16777216 -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 -e SGLANG_REQ_WAITING_TIMEOUT=30" \
SG_EXTRA="--max-prefill-tokens 4096 --prefill-max-requests 1 --max-queued-requests 8" \
NAME=glm53_sglang_tp8 LAUNCHER=launch-sglang-tp8.sh bash scripts/cluster.sh up
```

Workers 7→1 come up first, head last; "The server is fired up and ready to roll!" on the head after ~11 min. Serves on `:8891`
as model `glm-5.3-nvfp4-sglang`. Solo variant: `SG_MAXRUN=2` and drop `SG_EXTRA` (arm s10, same speed).

## Verify

```bash
BASE_URL=http://<head>:8891/v1 python3 scripts/gates.py          # 401 keyless, model/ctx, ROUTE_OK smoke, tool call + tool round
BASE_URL=http://<head>:8891/v1 python3 scripts/bench.py <tag>    # C1 ×3 prompts (median of 3) + C4
BASE_URL=http://<head>:8891/v1 python3 scripts/needle.py 470000  # ~514K needle; also 32000, 120000
HERMES_HOME=<throwaway> python3 scripts/hermes_gauntlet_reps.py <lane> 'custom[sglang]' glm-5.3-nvfp4-sglang 3
```

Pass bar (recipe.yaml `verification.performance`): C1 within 5% of 36.2 / 23.7 / 28.7, C4 within 10% of 59.5, needles PASS, chain median ≤ 36 s.

## Software

Stock `lmsysorg/sglang:v0.5.22-cu130` (manifest `sha256:e3a4f060…`), no overlay, no patches: SGLang 0.5.22, torch 2.14.1+cu130,
flashinfer 0.7.1. Driver 595.84, kernel 6.17.0-1031-nvidia, DGX OS 7.2.3. Nothing after v0.5.22 on `main` touches SM121;
FlashInfer 0.7.2 lands with an open regression against the exact decode file this lane uses, so a nightly is a risk, not a win.

## The three things that matter on unified memory

1. **`SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.01`.** The DSA indexer scores every query row against every key column
   into an fp32 scratch that the memory planner does not account for; its default budget (20% of "free" memory) assumes a
   discrete GPU. On GB10 that memory is the host's. Without the cap, host free memory falls ~80 KB per prompt token and
   never recovers; the group dies with `NV_ERR_NO_MEMORY` at ~100–140K tokens whatever `--mem-fraction-static` is.
   Five one-change probes bisected it (`results/.../RESULTS.md`); reported on
   [sglang #36941](https://github.com/sgl-project/sglang/issues/36941#issuecomment-6234243195), which had the mechanism
   from beastllama on 2× Spark in August.
2. **`--chunked-prefill-size 4096`.** 8192 and 16384 hang on the first 35K prompt even with the budget bounded (arms s3, s4).
3. **`NCCL_BUFFSIZE=16777216 NCCL_MAX_NCHANNELS=4 NCCL_MIN_NCHANNELS=4`** — copied from the vLLM sibling's launcher.
   +8% prose decode and +15% KV pool at the same fraction (smaller NCCL buffers leave more for cache). The biggest single
   lever in the campaign and it is three env vars.

## Results

### Campaign (2026-10-17, one boot per arm, same gates every arm)

| arm | change | code / prose / explain | C4 | chain ×3 | needles |
|---|---|---|---:|---:|---|
| s0 | baseline 512K, 3 steps, 2 slots | 34.1 / 26.7 / 24.5 | 45.9 | 37.9 s | 35K 131K 514K ✓ |
| s1 | indexer budget 0.03 | 34.0 / 26.6 / 24.5 | 40.9 | 41.4 | ✓ — no effect |
| s3 / s4 | chunk 8192 / 16384 | — | — | hang | ✗ first prompt |
| s5 | 4 draft steps | 35.5 / 27.1 / 24.8 | 40.3 | **33.4** | ✓ |
| s6 | NCCL pin | 35.7 / **28.8** / **25.1** | 41.8 | 35.6 | ✓ pool 655K |
| s7 | shared-experts fusion on | 34.1 / 26.7 / 24.5 | 42.4 | 42.4 | ✓ — no effect |
| s8 | fused MoE finalize | 34.2 / 26.8 / 24.6 | 41.6 | 37.5 | ✓ — no effect |
| s9 | 4 slots + rails | 34.3 / 26.8 / 24.6 | **56.4** | 40.9 | ✓ |
| s10 | 4 steps + NCCL | **36.2** / 28.6 / 23.7 | 41.3 | 33.5 | ✓ |
| **s11** | **s10 + s9 rails (this recipe)** | **36.2 / 28.7 / 23.7** | **59.5** | **31.7** | ✓ |

2-slot arms are slot-capped on C4; compare C4 only across s9/s11. Winner rule was frozen before the first boot: highest
code C1 with all needles green; the shared rails were then confirmed to cost nothing single-stream.

The night before (`gates/sglang-campaign` in the sibling's project, summarized in the blog): 131K sweeps found 2 draft
steps best for throughput (64.0 C4) and 5 steps worst; `--disable-overlap-schedule` −3–5%; tree drafts, cutedsl and
triton MoE refuse on SM121.

### Hermes bench

Same `hermes chat -Q` harness against both lanes from a throwaway home (no memory, no rules, reasoning low, terminal tool):
one-word reply, count to 300, small code task, 400-word artifact, and a 20-step tool chain with a hidden token. Everything
but the tool chain ties; the tool chain is where SGLang pulls ahead (31.7 s vs 45.8 s). Harness: `scripts/hermes_bench_lane.py`,
`scripts/hermes_gauntlet_reps.py`; raw in `results/.../raw/hermes-bench/`.

## Hardware

Same as the siblings (`hardware/dgx-spark-8x.yaml`). GPU clocks are capped at 2,200 MHz on all eight since an uncapped
514K prefill drew ~1.74 kW and tripped the rack strip; capped long prefill is 1,350–1,425 W and decode did not change.

## Known limits

See `recipe.yaml: limits` for the full list. In short: the indexer scratch env var and chunk 4096 are mandatory at 512K;
4 draft steps cost explanation speed; prefill at 514K is half the vLLM lane's; 1M does not fit; tree drafts, cutedsl and
triton MoE, and FlashInfer autotune all refuse on SM121 in v0.5.22.

### Not done

- Teacher-forced Δlogprob vs the vLLM sibling (greedy parity: primes identical; three prompts diverge after 15–73 chars, both coherent).
- An overload test that forces the queue/timeout rails.
- Prefill at 514K is half the vLLM lane's; `--prefill-max-requests 1` and chunk 4096 are the suspects and were not swept separately.

## Rollback

`NAME=glm53_sglang_tp8 LAUNCHER=launch-sglang-tp8.sh SPARK_HOSTS="..." bash scripts/cluster.sh down`, then bring up the
vLLM sibling with its launcher (`NAME=glm53_nvfp4_tp8 LAUNCHER=launch-nvfp4-tp8.sh`). Both images and the weights stay on
every node.

## Credits

beastllama (the indexer memory mechanism, #36941), RadixArk (the checkpoint), the SGLang team (v0.5.22 and the DSA guard
module), 0xMagnus (the NCCL pin in the vLLM launcher, clock-cap unit), and the vLLM sibling's credits.

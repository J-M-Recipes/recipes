# J&M Recipes

Hardware-pinned, verified recipes for serving large open-weight LLMs. **Every number has a method.**

A recipe here promises: on *this* hardware (as the machine reports it, not the spec sheet), with *this* container digest and model revision, *this* command produces *these* numbers, measured *this* way — and here is the raw output they were read from. Recipes also keep the ledgers of what didn't work, because that's half the value.

Written by James Meadlock and Milo. Recipes are MIT; models and engines carry their own licenses and their authors' credit.

## Index

<!-- recipe-index:start -->

| Hardware | Recipe | Model | Status | Headline | Updated |
|---|---|---|---|---|---|
| `dgx-spark-8x` | [Full GLM-5.3 (744B, official zai-org FP8) on 8x DGX Spark, vLLM TP8 over 200G RoCE + MTP](recipes/dgx-spark-8x/glm-5.3-fp8-tp8/) | GLM-5.3 (FP8 as shipped by Z.ai (e4m3, 128x128 block scales, activation_scheme dynamic); 141 safetensors, 755,632,050,320 bytes; all 143 LFS files sha256-matched to the Hub on all eight nodes (results/2026-10-16-fp8-tp8-baseline-mtload/raw/weight-fanout-verify.log). Weight shards are sha256-identical to revision e0b07fd2; ba32f8c1 changed chat_template.jinja (tool-result validation) and eval yamls only.) | 🧪 experimental | 29.92 tok/s @ C1, nginx-log-parser + pytest prompt, 800 max tokens, T=0, reasoning low, streamed; 1 discarded warm + 3 reps | 2026-10-16 |
| `dgx-spark-8x` | [Full GLM-5.3 (744B, RadixArk NVFP4) on 8x DGX Spark, vLLM TP8 over 200G RoCE + MTP, 512K context](recipes/dgx-spark-8x/glm-5.3-nvfp4-tp8/) | GLM-5.3 (NVFP4 (ModelOpt 0.47.0.dev, group 16) on routed experts; 231 ignore entries keep self_attn, shared experts, mlp.gate, layers 0-1, embed/lm_head and the MTP layer (model.layers.78*) in bf16; checkpoint kv_cache_scheme fp8 (serve-time fp8_ds_mla used here). 47 safetensors, 464,823,042,096 bytes; all 49 LFS files sha256-matched to the Hub on all eight nodes. Third-party QFS teacher-forced KLD vs GLM-5.3 BF16: 0.0511 (malaiwah/quant-fidelity-suite, comparator 312a2551db89; stored weights reconstructed, not executed through these kernels).) | 🧪 experimental | 26.29 tok/s @ C1, nginx-log-parser + pytest prompt, 800 max tokens, T=0, reasoning low, streamed; 1 discarded warm + 3 reps (same scripts/bench.py as the FP8 sibling) | 2026-10-16 |
| `dgx-station-gb300` | [DeepSeek-V4.1-Flash at 1M context on one DGX Station GB300 (vLLM UVA expert offload + DSpark)](recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/) | DeepSeek-V4.1-Flash (as shipped — routed experts MXFP4 (I8 + E8M0 block scales), Engram and attention FP8. Nothing re-quantized.) | ✅ verified | 182.8 tok/s @ C1, warm, prose (knee.sh r2), T=0, ignore_eos, 192 tok; 0909 image, v18 launch, hook v21, autotune set loaded (189 configs) on all three boots | 2026-10-06 |
| `dgx-station-gb300` | [GLM-5.3-Flash NVFP4 + DFlash2 speculative decoding on one DGX Station GB300](recipes/dgx-station-gb300/glm-5.3-flash-nvfp4-dflash2/) | GLM-5.3-Flash (NVFP4 (NVIDIA ModelOpt, first-party; 204,475,659,764 bytes). Routed experts FP4; attention, shared experts, router, embeddings, lm_head unquantized (132 excluded modules). mean |dlogp| 0.136 vs zai-org/GLM-5.3-Flash FP8 on 5,120 teacher-forced tokens; argmax differs 15.6%. LibertAIDAI/GLM-5.3-Flash-NVFP4 (previous revision of this recipe): 0.147 / 15.9%. nvidia closer on 27/40 prompts, bootstrap 95% CI on the gap [0.004, 0.017]. Most divergence is shared NVFP4 bias: both quants pick the same non-FP8 token 10.3% of the time. Previous revision of this recipe used LibertAIDAI/GLM-5.3-Flash-NVFP4 @ aa28e1f54130286c95fee10d0705c74ce8743734 (run 2026-09-01); speed identical within noise (235.5 vs 238.1 C1); kept as control.) | ✅ verified | 79.3 percent @ BFCL_v4 live_simple + live_multiple, 1,311 cases, frozen, run once; T=0, reasoning_effort low, tool_choice auto, C8, protocol harness/protocol.yaml (sha256 7120a254c43a); 1,040 solved; unsolved 184 wrong-arg-value / 31 missing-arg / 29 wrong-function / 18 no-call / 9 HTTP 400 | 2026-10-01 |
| `dgx-station-gb300` | [Full GLM-5.3 NVFP4 on one DGX Station GB300 (vLLM UVA offload + HBM expert slot cache + MTP)](recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/) | GLM-5.3 (NVFP4 (ModelOpt)) | ✅ verified | 58.2 tok/s @ C2, 8 distinct nonce-suffixed prompts x 512 tokens, T=0, 3 scored reps after 1 discarded warm rep; same argv as daily except --max-num-seqs 2 | 2026-10-01 |
| `dgx-station-gb300` | [MiMo-V2.6-Pro (1.02T MoE) on one DGX Station GB300 (vLLM UVA expert offload + per-expert HBM residency)](recipes/dgx-station-gb300/mimo-v2.6-pro-vllm-uva-hotsplit/) | MiMo-V2.6-Pro-MOPD (as shipped — routed experts MXFP4 (494.9 GiB), fused qkv_proj FP8 e4m3 128x128 blocks (10.9 GiB), o_proj BF16 (13.1 GiB, in ignored_layers). Nothing re-quantized. 573,489,510,171 bytes on disk incl. ViT/audio/DFlash drafter (per-file sizes verified against the HF manifest, 155 files). PRIOR REVISION 2026-09-22..27 — MiMo-V2.6-Pro-RL rev 54b10491 (573,492,132,336 bytes), the v14-v24 measurements; config.json, chat_template.jinja, generation_config.json and model.safetensors.index.json are byte-identical between the two, so every launch line and patch applies unchanged. MOPD = Xiaomi's 2026-09-27 repetition fix (12-step specialised RL teacher merged by multi-teacher on-policy distillation; mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition).) | ✅ verified | 36 of 36 @ scripts/flood_fixture.py (protocol-v2.yaml v2.1): 36 deterministic simulated-env tasks x {history 0, history 1}; families pagination / EACCES-switch / 12-file fan-out / test-fix loop / ENOENT-then-list / multi-hop; T=0, thinking ON, 6 turns, 120 calls per task then 'flooded', C4. history 1 = one prior turn in context with 12 calls of which 8 exact duplicates (Xiaomi's 'history 1'). Metric: exact within-turn duplicates (N-U)/N after JSON canonicalisation, Xiaomi's lower-bound definition | 2026-09-28 |
| `dgx-station-gb300` | [Muse Spark Open 816B-A42B (NVFP4) on one DGX Station GB300 (vLLM UVA expert offload + DSpark k=2)](recipes/dgx-station-gb300/muse-spark-open-816b-vllm-uva-dspark/) | Muse Spark Open 816B-A42B (as shipped, nothing re-quantized: NVFP4 (group 16) experts on layers 1-60, FP8 experts on layers 0 and 61, BF16 attention/embeddings/lm_head. 141 files / 533,043,429,228 bytes byte-verified against the Hub listing; 128 LFS files sha256-verified after staging to local NVMe (results/2026-10-16-muse-s1/model-provenance). Gated repo (manual approval); Muse Model License requires 'Built with Muse Spark Open' attribution.) | 🧪 experimental | 29.44 tok/s @ C1, warm, T=0, 256 tokens forced (min_tokens + ignore_eos), streaming, template-default reasoning effort (xhigh: the forced tokens are reasoning text — see limits); prompt 'history of lighthouses' essay | 2026-10-16 |

<!-- recipe-index:end -->

## How a recipe is laid out

```
recipes/<hardware>/<recipe>/
  README.md          current state: what it runs, hardware, software, launch, verify, results, known limits, rollback
  recipe.yaml        the machine-readable pin (schemas/recipe.schema.json) — validated in CI
  scripts/           launch / health / warmup / bench / rollback — what was actually run
  results/<run-id>/  provenance bundle for every published number
  patches/           any code mounted into the container, sha256-pinned
  research/          ledgers, root causes, failed experiments
```

`hardware/` holds one profile per machine with **observed** values; `scripts/collect_system_snapshot.sh` produces them.

## Contract, in ten lines

1. No `:latest` — image digests, always.
2. Model revisions are commit SHAs.
3. Hardware is what `nvidia-smi` says.
4. Every metric: value, unit, workload shape, run-id, method, raw file.
5. Five gates — schema, digest, health, quality, performance — with pass dates.
6. `verified` means all five passed on a recorded run.
7. Quality gates are per recipe and stated; there is no global bar.
8. Known limits are listed, or you didn't look.
9. Rollback is a script.
10. No credentials in the tree.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contract and [schemas/recipe.schema.json](schemas/recipe.schema.json) for the fields.

## Verify locally

```bash
uv venv .venv && uv pip install -p .venv/bin/python jsonschema pyyaml pytest numpy
.venv/bin/python -m pytest tests -q && .venv/bin/python scripts/check_recipe.py --all
```

# J&M Recipes

Hardware-pinned, verified recipes for serving large open-weight LLMs. **Every number has a method.**

A recipe here promises: on *this* hardware (as the machine reports it, not the spec sheet), with *this* container digest and model revision, *this* command produces *these* numbers, measured *this* way — and here is the raw output they were read from. Recipes also keep the ledgers of what didn't work, because that's half the value.

Written by James Meadlock and Milo. Recipes are MIT; models and engines carry their own licenses and their authors' credit.

## Index

<!-- recipe-index:start -->

| Hardware | Recipe | Model | Status | Headline | Updated |
|---|---|---|---|---|---|
| `dgx-station-gb300` | [DeepSeek-V4.1-Flash at 1M context on one DGX Station GB300 (vLLM UVA expert offload + DSpark)](recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/) | DeepSeek-V4.1-Flash (as shipped — routed experts MXFP4 (I8 + E8M0 block scales), Engram and attention FP8. Nothing re-quantized.) | ✅ verified | 82.1 tok/s @ C1, warm, prose prompt, DSpark k=5, 1M context config | 2026-09-11 |
| `dgx-station-gb300` | [GLM-5.3-Flash NVFP4 + DFlash2 speculative decoding on one DGX Station GB300](recipes/dgx-station-gb300/glm-5.3-flash-nvfp4-dflash2/) | GLM-5.3-Flash (NVFP4 (NVIDIA ModelOpt, first-party; 204,475,659,764 bytes). Routed experts FP4; attention, shared experts, router, embeddings, lm_head unquantized (132 excluded modules). mean |dlogp| 0.136 vs zai-org/GLM-5.3-Flash FP8 on 5,120 teacher-forced tokens; argmax differs 15.6%. LibertAIDAI/GLM-5.3-Flash-NVFP4 (previous revision of this recipe): 0.147 / 15.9%. nvidia closer on 27/40 prompts, bootstrap 95% CI on the gap [0.004, 0.017]. Most divergence is shared NVFP4 bias: both quants pick the same non-FP8 token 10.3% of the time. Previous revision of this recipe used LibertAIDAI/GLM-5.3-Flash-NVFP4 @ aa28e1f54130286c95fee10d0705c74ce8743734 (run 2026-09-01); speed identical within noise (235.5 vs 238.1 C1); kept as control.) | ✅ verified | 252.5 tok/s @ C1, history essay, 512 output tokens, thinking default, DFlash2 block 7, 1M ctx | 2026-09-11 |
| `dgx-station-gb300` | [GLM-5.3-NVFP4-One-GB300](recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/) | GLM-5.3 (NVFP4 (ModelOpt)) | 🧪 experimental | 33.8 tok/s @ C1, prose prompts, 512 generated tokens, n=3, V1 baseline flags: 188 GiB offload, bf16 KV 8 GiB, seq4, 65k | 2026-09-08 |

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

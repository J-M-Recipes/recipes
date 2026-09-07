# J&M Recipes

Hardware-pinned, verified recipes for serving large open-weight LLMs. **Every number has a method.**

A recipe here promises: on *this* hardware (as the machine reports it, not the spec sheet), with *this* container digest and model revision, *this* command produces *these* numbers, measured *this* way — and here is the raw output they were read from. Recipes also keep the ledgers of what didn't work, because that's half the value.

Written by James Meadlock and Milo. Recipes are MIT; models and engines carry their own licenses and their authors' credit.

## Index

<!-- recipe-index:start -->

| Hardware | Recipe | Model | Status | Headline | Updated |
|---|---|---|---|---|---|
| `dgx-station-gb300` | [GLM-5.3-Flash NVFP4 + DFlash2 speculative decoding on one DGX Station GB300](recipes/dgx-station-gb300/glm-5.3-flash-nvfp4-dflash2/) | GLM-5.3-Flash (NVFP4 (ModelOpt)) | ✅ verified | 234.2 tok/s @ C1, warm, prose prompts, DFlash2 | 2026-09-06 |
| `dgx-station-gb300` | [GLM-5.3-NVFP4-One-GB300](recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/) | GLM-5.3 (NVFP4 (ModelOpt)) | 🧪 experimental | 33.8 tok/s @ C1, prose prompts, 512 generated tokens, n=3, V1 baseline flags: 188 GiB offload, bf16 KV 8 GiB, seq4, 65k | 2026-09-07 |

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
uv venv .venv && uv pip install -p .venv/bin/python jsonschema pyyaml pytest
.venv/bin/python -m pytest tests -q && .venv/bin/python scripts/check_recipe.py --all
```

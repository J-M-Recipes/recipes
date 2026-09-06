# Contributing a recipe

A J&M recipe is a promise: *on this exact hardware, with this exact software, this command produces these numbers, measured this way.* Everything in this file exists to keep that promise checkable by a stranger with the same box.

## The contract

Every recipe lives at `recipes/<hardware-slug>/<recipe-slug>/` and contains:

| file | purpose |
|---|---|
| `recipe.yaml` | machine-readable pin: model revision, image digest, hardware as **observed**, launch command, metrics with provenance, gates, limits, rollback. Validated by `schemas/recipe.schema.json` + `scripts/check_recipe.py` in CI. |
| `README.md` | the human page. Current state only — no diary. Required sections: `What this runs`, `Hardware`, `Software`, `Launch`, `Verify`, `Results`, `Known limits`, `Rollback`. |
| `scripts/` | the launch script the README shows, plus health wait, warmup, benchmark, rollback. What you ran, not a tidied version of it. |
| `results/<run-id>/` | provenance bundle for every published number: `system.json` (from `scripts/collect_system_snapshot.sh`), image digest, model revision, exact launch command, the raw benchmark/quality output the number was read from, trimmed logs. |
| `known-limits.md` (optional) | longer form of `limits:` when the tradeoffs need explaining. |
| `research/` (optional) | ledgers, root-cause writeups, failed experiments. Encouraged — what didn't work is half the value. |
| `patches/` (optional) | any code the recipe mounts into the container; each listed in `recipe.yaml` with its sha256. |

## Rules the checker enforces

- **No `:latest`.** Container images are `name:tag` plus a `sha256:` digest.
- **Model revisions are 40-char commit SHAs**, never branch names.
- **Hardware is what the machine reports**, not the spec sheet. `hardware.observed` must point at a `results/<run-id>/system.json` that produced it. (Our GB300 shows 250.7 GiB, not 288 GB. Say so.)
- **Every metric has provenance**: `run_id`, `method`, and the `raw` file the number came from — and that file must exist.
- **`status: verified` requires** every gate (`schema`, `digest`, `health`, `quality`, `performance`) to carry a `last_pass_utc` and `run_id`, and at least one published metric.
- **No credentials** anywhere in the tree. API keys are read from files outside the repo at launch time.
- **Patches match their sha256.**

## Rules humans enforce (PR review)

- A number without a workload shape (concurrency, prompt/output lengths, prompt class) is not a number.
- If a number was measured before a fix, it carries a `label:` saying so and the README says so next to it.
- Quality gates are **per recipe and stated**. One recipe may accept FP8 KV cache; another may forbid it. Neither is "the" bar — the bar is whatever the recipe declares and measures.
- `Known limits` is not optional in spirit: if you didn't find any, you didn't look.
- Rollback must be a script, not a sentence.

## Status meanings

| status | means |
|---|---|
| `draft` | being written; may not launch |
| `experimental` | launches and has numbers, but a gate is unmeasured or a known defect is open |
| `verified` | all five gates passed on a recorded run; a stranger with the box should reproduce the headline within ~5% |
| `deprecated` | superseded or the software has moved on; kept for the record |

## Workflow

```bash
uv venv .venv && uv pip install -p .venv/bin/python jsonschema pyyaml pytest
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/check_recipe.py --all
.venv/bin/python scripts/render_index.py
```

Open a PR; CI runs the same three commands. Reviews focus on provenance, not prose.

## Attribution

Recipes stand on other people's kernels, engines, and checkpoints. Credit the authors of what you used, by name and link, in the recipe README. Measure before arguing with them.

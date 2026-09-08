# Offline instrumentation disposition

Verdict: documentation/manifests reconciled for offline canary-preparation review only. This is not a live release, GPU safety proof, performance proof, or campaign-valid acceptance.

## Provenance

- Instrumentation offline GO report: `/Users/jamesmeadlock/.hermes/profiles/milo/cache/delegation/subagent-summary-0-20260908_181820_970825.txt`
- Launcher offline GO report: `/Users/jamesmeadlock/.hermes/profiles/milo/cache/delegation/subagent-summary-0-20260908_183643_693828.txt`
- Reviewed repo HEAD in both reports: `f84290bed6acea1afc3a9e8a9742c6269cd2ca4c`

## Current reviewed hash envelope

| item | SHA-256 / value |
|---|---|
| Pinned source `gpu_model_runner.py` | `7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa` |
| Generated patched `gpu_model_runner.py` | `2268a6dafda69566d4128bb9b589bdecb22e3e7eb8d0b7e1155f2bb1ce8e3cd4` |
| Instrumentation adapter | `9f0c75b25438c63511a5b2580a4c0a77520f232e2affe109dd0ba3908477e453` |
| Staged generator | `8b4b3ae177618875378154681a43c16bf4cc265c6f073fb1dd6ef2562c45106b` |
| Slot-cache hook | `27241f2ba66736ade5717201f33b3d175860c11de84fb81578ff9b753bedd1b7` |
| Portable launcher | `aebe4fab6272a8ded9d2e871d5b9c536b641634ae9b10232db9fa5c33bcac04d` |
| Local Docker image identity | `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`; local image ID, not a registry manifest digest |

Review invalidation scope: only the portable launcher hash changed from the prior offline-reviewed envelope; instrumentation adapter, staged generator, generated patched runner, slot-cache hook, pinned source, and local image identity hashes are preserved.

## Local validation disposition

- The three existing regression failures were reproduced RED: disabled/default launcher tests expected image tag `vllm-glm53-uva:v0.28.0-2cf0a691`, but the launcher defaulted globally to the pinned local image ID.
- Affected launcher regression tests are GREEN after restoring disabled/default image-tag behavior and keeping enabled instrumentation pinned-ID behavior: 7 passed.
- Full serialized pytest was run with the requested venv Python, `PYTHONDONTWRITEBYTECODE=1`, and a unique basetemp. Result: 211 tests collected; 211 passed.
- Historical result manifests were not regenerated.

## Remaining canary gates

Generated raw snapshot metadata remains `valid_for_campaign=false` until Station canaries prove stream/capture behavior, graph replay correlation, async D2H race safety, and target-vs-draft counter scope. The local package does not prove GPU safety, campaign validity, or live release readiness.

# 2026-10-17-v22-checkpoint-1017: the V20 release on the DeepSeek-V4.1-Flash-1017 checkpoint

**Status: experimental candidate.** It boots and passes T0. Performance and quality are **not measured** (see below).
The V20 release, its numbers, and its gates are unchanged. This bundle does not promote anything.

## What changed vs V20

Only the weights. The image, flags, hook, rowmap, sitecustomize, and FlashInfer autotune set are the V20 release's.

| | V20 (release) | v22 (this bundle) |
|---|---|---|
| model | `deepseek-ai/DeepSeek-V4.1-Flash@df42c109f1defefcbfcedbe7d905718a12266e40` | `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170` |
| image | `vllm/vllm-openai@sha256:91c875065792e9b79dfec5019e5017f96b1511eed292e7ec15182444b23efe66` (nightly-2671fedf) | same |
| hook / sitecustomize / rowmap | v15: `723066dd…` / `a860036c…` / `550c3202…` ([`2026-09-17-e2b-pin-hot-experts-v15`](../2026-09-17-e2b-pin-hot-experts-v15/)) | same bytes (sha256 checked at launch) |
| flags | `--cpu-offload-gb 54`, `fp8_ds_mla`, seqs 24, DSpark k-schedule `[[1,4,5],[5,24,1]]`, 1M, util 0.97, capture list 1…128 | same ([`launch-v22.sh`](launch-v22.sh); host paths scrubbed to `<lane-root>` / `<hf-home>`) |

The 1017 release ships byte-identical `config.json`, `model.safetensors.index.json` (96,085 tensors, DSpark drafter included), `tokenizer.json`, and `tokenizer_config.json`.
Every one of the 48 weight shards has new bytes, so this is a full 475.3 GiB download. Neither release has `chat_template.jinja` or `generation_config.json`.

Launch differences from the recipe's `launch-dsv41-vllm.sh`:
- the served model name is `current` on `:8000`;
- the container is wrapped in `numactl --membind=0` with `CAP_SYS_NICE`;
- `--gpus all` is used, which exposes only the GB300 on this runtime (probed).

None of these change the FlashInfer config hash.

## Integrity

`receipts/shards.sha256`: **48/48** shard sha256 equal the HF LFS oids for revision `1bae0e10` (`receipts/hf-oids.tsv`, from `/api/models/…/revision/1bae0e10…?blobs=true`).
The four config and tokenizer files were compared with `cmp` against `df42c109`: identical.

## Boot (2026-10-17, 14:06–14:12 UTC; `receipts/boot-trimmed.log`, full-log sha256 in `receipts/boot-full.sha256`)

| check | v22 / 1017 | V20 / df42c109 on the same host |
|---|---|---|
| PIN_HOT `split` installed, rowmap `rowmap-static-v1.json` | yes | yes |
| re-home `HBM_expert / pinned_expert` | **206.61 / 62.33 GiB** | 206.61 / 62.33 |
| after re-home: `hbm_free / host_avail` | 21.73 / 30.06 GiB | — |
| FlashInfer hash `e5a13633…` | **hit, `Loaded 189 configs`**, file sha256 unchanged after boot (0 new) | hit |
| `No tuned config` warnings during capture | 8 | 8 |
| graphs | PIECEWISE 15 + FULL; `Graph capturing finished in 36 secs, took 1.28 GiB` | same |
| KV | `Available KV cache memory: 4.03 GiB`, `GPU KV cache size: 1,918,727 tokens` (1.83× 1M) | 1.92–2.02M |
| load | `Model loading took 233.1 GiB memory and 144.8 seconds`; ready in ~6 min | ~6 min |

The host-memory margin at the hook's layer-9 floor is not a concern on this host and image. `host_avail` was still 30 GiB after all 40 layers, against the 10 GiB abort floor.

### Autotune set

`autotune-e5a13633/autotune_configs.json` (sha256 `a6aa6218…`, 189 tactics + `_generation` + `_metadata`) is the set this boot loaded.
It was tuned on this host on 2026-09-24 for the V20 release flags, because the release publishes only the checksum of its own set (`a34f9ad4…`), not the JSON. It is published here so a V20/v22 boot can seed it instead of running a 16–17 min live tune.
This set is **not** byte-identical to the release's `a34f9ad4…`. The engine-config hash on this host is `e5a13633…`.

## T0 gate: pass

| check | result | raw |
|---|---|---|
| `/v1/models` | `current`, `max_model_len 1048576` | `receipts/t0-models.json` |
| exact-word chat | `pineapple` | `receipts/t0-chat.json` |
| parsed `tool_calls[]` | `get_weather {"city": "Paris"}` | `receipts/t0-toolcall.json` |
| arithmetic smoke | `323` | `receipts/smoke.txt` |

Outside this recipe, the same lane also passed OpenAI Responses-API checks through a local shim:
- text;
- reasoning;
- a function call plus `function_call_output` continuation;
- `input_image` with `detail`;
- a Pi agent text, bash-tool and image round trip.

## Performance: NOT MEASURED

The lane was serving other clients during both perf runs. `shared-window/t1-perf.engine-load.txt` shows `Running: 9 reqs` against the bench's top concurrency of 8. `shared-window/knee.json` (knee, 192 tok, `ignore_eos`) ran on the same shared lane.
By [docs/benchmarking.md](../../../../../docs/benchmarking.md) rule 2, these cells are **relabelled, not published**. They are kept only so a reviewer can see what was discarded. **Do not quote them.**

To measure, run on a quiet lane, CTL→CAND→CTL against V20 (df42c109) in one window:
- `scripts/knee.sh`;
- T1 `--concurrency 1,4,8 --contexts 0,32768,131072 --duration 30`.

## Quality: NOT MEASURED

The weights changed, so T2 is required before any promotion:
- `needle-checksum` must stay EXACT N/N;
- GSM8K-200 runs with `--compare-baseline ../2026-10-05-lib-t1-v21/t2-gsm8k200.json`, and a paired McNemar p < 0.05 loss fails;
- held-out BFCL runs against the V20 / v18 sibling under the recipe's tool-call promotion rule.

The vendor's 1017 model card claims, for example, Terminal-Bench 42.3 vs 31.2, ProgramBench 29.1 vs 20.3, and HLE 42.6 vs 36.8 over the previous release. **These are vendor numbers, not measured here.** The card recommends temperature 1.0 and top_p 1.0.

## Known limits of this candidate

- **The rowmap is stale.** `rowmap-static-v1.json` is the decode routing profile of `df42c109` (`counts-20260917-183654.npz`). Placement is identical by construction: a fixed 295 hot / 89 cold rows per layer. But the cold-expert share of 1017's decode selections is unmeasured; it was 1.55% on df42c109, against a ≤3% bar. Re-deriving it needs a `PIN_MODE=count` boot (≈ −3.7% while counting) on the E2a workload classes.
- **No perf or quality numbers** (above).
- The new release does not change the tokenizer or index. So any regression versus V20 is in the weights, or in the stale rowmap's routing fit, and not in the serving stack.

## Rollback

[`rollback-v22.sh`](rollback-v22.sh) stops the v22 container, drops caches, resets the GB300, then starts the kept V20 container and waits for readiness. It is equivalent to:

```bash
docker stop -t 60 <v22-container> && sync && echo 3 | sudo tee /proc/sys/vm/drop_caches \
  && sudo nvidia-smi -i <gb300-index> --gpu-reset && docker start <v20-container>
```

## Credits

DeepSeek for the 1017 checkpoint. The vLLM team for the nightly image. This recipe's v15 pin-hot-experts hook and v20 release by the recipe owners.

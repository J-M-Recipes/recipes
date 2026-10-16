# Full GLM-5.3 (744B, RadixArk NVFP4) on 8× DGX Spark — vLLM TP8 over 200G RoCE + MTP, 512K context

**Status: experimental** (2026-10-16). **GhettoRak's daily GLM-5.3 lane** (James's pick, 2026-10-16). Serves, passes health,
tool and needle gates to 514K, measured. Served teacher-forced quality gate pending.

Sibling: [`dgx-spark-8x/glm-5.3-fp8-tp8`](../glm-5.3-fp8-tp8/) — same cluster, same image, official FP8 weights.

| | NVFP4 (this recipe) | FP8 sibling |
|---|---:|---:|
| Weights + MTP per node | 57.13 GiB | 90.96 GiB |
| KV pool / context | **720,384 / 524,288** | 150–182K / 131,072 |
| C1 code / prose / explanation (tok/s) | 26.3 / 24.9 / **23.8** | **29.9 / 26.9** / 23.1 |
| C4 aggregate (tok/s) | **59.1** (16.5–18.4 per user) | 40.1 (11.5–14.2) |
| Deepest needle passed | **514K** (~1.2K tok/s prefill) | 70K |
| QFS KLD vs BF16 (third party) | 0.0511 | **0.0223** |
| Cold boot | 17.5 min | 13.2 min |

## What this runs

`RadixArk/GLM-5.3-NVFP4` @ `6e389189`: ModelOpt NVFP4 (W4A4, group 16) on the routed experts, with attention, shared
experts, routers, layers 0–1, embeddings/lm_head and the MTP layer left in bf16 — which is why every GLM-5.3 NVFP4 build is
~433 GiB. Of the three NVFP4 builds with a QFS measurement it has the best fidelity (QFS 0.0511 vs incoai 0.0594, Inferact 0.0754).

Tensor parallel 8, one rank per Spark, MTP one token ahead, `fp8_ds_mla` KV, B12X sparse attention. Launcher is
0xMagnus's NVFP4 TP8 config (gmu 0.80, 512K, MTP k=1) on the same base as the FP8 sibling, plus the 8-thread loader.

## Hardware

Same as the FP8 sibling — see `hardware/dgx-spark-8x.yaml`. Kernel 6.17.0-1031 held, both PCIe functions of the cabled
QSFP cage in NCCL, 8-node all_reduce ≥ 20 GB/s (measured 20.13 GB/s, `raw/nccl-8node-allreduce.log`), `vm.swappiness=10`.
Do not raise `--gpu-memory-utilization` above 0.80: the head keeps ~8.5–8.9 GB of host memory free while serving, and
0xMagnus saw a global OOM at 0.86 during FP4 autotune.

## Software

- Image `ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision` @ `sha256:7ac0031c…` (loaded Id `sha256:ba79eedf…`), NCCL 2.30.4.
- Weights fanned out from one node over the fabric (NFS export, 8-way parallel copy, ~8 minutes for seven copies), then
  49/49 LFS files sha256-checked against the Hub on every node (`raw/weight-fanout-verify.log`, `raw/hub-lfs.sha256`).

## Launch

```bash
export SPARK_HOSTS="spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8"   # rank order; rank 0 = head
# head only: ~/glm53-tp8/.env.glm53 with GLM53_API_KEY=<random>, mode 600
bash scripts/cluster.sh push
bash scripts/cluster.sh up        # NAME=glm53_nvfp4_tp8 LAUNCHER=launch-nvfp4-tp8.sh are the defaults here
```

Overrides: `MODEL_HOST`, `GMU` (0.80), `MAXLEN` (524288), `SPEC_K` (1). API on the head at `:8890`, bearer key.

Boot timeline (head): weight load 558 s (57.13 GiB/rank) → torch.compile 77 s cold → KV profile (37.82 GiB/rank,
720,384 tokens) → FlashInfer trtllm FP4 MoE autotune ~104 s → CUDA graphs → ready at 17.5 min.

## Verify

```bash
BASE_URL=http://<head>:8890/v1 GLM53_API_KEY=... python3 scripts/gates.py
BASE_URL=... GLM53_API_KEY=... python3 scripts/bench.py <tag>
BASE_URL=... GLM53_API_KEY=... python3 scripts/needle.py 256000 0.5
```

Passed 2026-10-16: keyless 401; `glm-5.3-nvfp4` / 524288; exact `ROUTE_OK`; `get_weather {"city":"Telluride"}` then a
correct answer from the tool result; needles at 35,055 / 69,979 / 140,034 / 280,261 / 513,814 tokens.

**Quality gate: pending.** The third-party QFS figure measures stored weights, not these kernels. A served teacher-forced
comparison against the FP8 sibling on this cluster is the next measurement.

## Results

Run `results/2026-10-16-nvfp4-radixark-tp8/` (`throughput.csv`, raw jsonl, boot log excerpt, all eight system snapshots).
Same instrument as the FP8 sibling: streamed, T=0, `reasoning_effort: low`, 800 max tokens, 1 warm + 3 reps.

| | value |
|---|---|
| C1 code | **26.29** tok/s (26.24–26.42) |
| C1 prose | **24.89** (24.71–25.28) |
| C1 technical explanation | **23.82** (23.25–23.96) |
| C4 aggregate | **59.1** (per user 16.45–18.36) |
| MTP | k=1, mean acceptance length 1.77–1.98 |
| needle 35,055 / 69,979 / 140,034 / 280,261 / 513,814 tokens | PASS 37.6 / 50.4 / 102.2 / 214.8 / 413.9 s (~930 / 1,390 / 1,370 / 1,305 / 1,241 tok/s prefill) |
| reasoning high, 1,500 / 16,000 max tokens | length with a partial answer / complete in 2,716 tokens |
| KV pool | 720,384 tokens (37.82 GiB/rank) |
| cold boot | 17.5 min |

## Known limits

- **Fidelity is the price.** QFS stored-weight KLD 0.0511 vs 0.0223 for the official FP8. Use the FP8 sibling when the
  exact model matters more than context or concurrency.
- **Not faster single-stream.** 7–12% slower than FP8 on code and prose, ~3% faster on the explanation prompt.
- **reasoning `high` needs budget.** 1,500 tokens ran out mid-answer (partial content, `finish_reason: length`); 16K
  finished in 2.7K. Better than FP8, which returned nothing at 1,500. Only `low`/`high` are honoured.
- **512K configured, 514K-token needle passes** (one needle per length, depth 0.5 — retrieval, not long-context reasoning). 1M needs `nvfp4_ds_mla` KV or DCP; neither tested.
- Boot 17.5 min; single-file loader on the head is still the long pole.
- Decode is latency/weight-read bound; Light Foundry's RoCE all-reduce is not in this image.

## Rollback

`bash scripts/cluster.sh down`. To serve the FP8 sibling instead, use that recipe's launcher with the same cluster
script; both checkpoints stay on every node (~1.2 TB each).

## Credits

- **RadixArk** — the GLM-5.3 NVFP4 build.
- **0xMagnus** — `im0xMagnus/glm-5.3-uncensored-8x-dgx-spark`: the NVFP4 TP8 config (gmu 0.80, 512K, MTP k=1) and the
  traps documented there.
- **ciprianveg** — `gb10-vllm` and the v19 image. **wrldsuksgo2mars / b12x** — SM121 sparse-MLA kernels.
  **local-inference-lab** — the vLLM fork. **CosmicRaisins** — MTP controller. **eugr** — spark-vllm-docker.
- **malaiwah** — quant-fidelity-suite (the KLD figures). **Z.ai** — GLM-5.3.

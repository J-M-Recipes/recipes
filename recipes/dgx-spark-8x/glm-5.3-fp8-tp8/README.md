# Full GLM-5.3 (744B, official FP8) on 8× DGX Spark — vLLM TP8 over 200G RoCE + MTP

**Status: experimental** (2026-10-16). Serves, passes health/tool/needle gates, measured. Teacher-forced quality gate pending.

Sibling: [`dgx-spark-8x/glm-5.3-nvfp4-tp8`](../glm-5.3-nvfp4-tp8/) — RadixArk NVFP4 on the same cluster: 4× the context, +54% at C4, and with MTP k=3 faster single-stream too (35.7 / 29.9 / 25.8 tok/s), at KLD 0.0511 vs 0.0223. NVFP4 is the cluster's daily lane; this FP8 recipe is the fidelity reference.

| | |
|---|---|
| Model | `zai-org/GLM-5.3` @ `ba32f8c1`, official FP8 as shipped, 703.7 GiB, no re-quantization |
| Hardware | 8× DGX Spark (GB10, 121.69 GiB unified each) on one MikroTik CRS804-4DDQ, 200G per node, lossy RoCEv2 |
| Runtime | vLLM 0.11.2 fork (ciprianveg `gb10-glm-5.2:v19-vision`, B12X sparse-MLA for SM121), TP8, mp executor, MTP adaptive 2/4/5 |
| Context | 131,072 (KV pool 149,824–181,632 tokens depending on boot) |
| C1 decode | **29.9 code · 26.9 prose · 23.1 explanation tok/s** (T=0, reasoning `low`, 800 tokens) |
| C4 | 40.1 tok/s aggregate (11.5–14.2 per user) |
| Prefill | ~1.2K tok/s at 70K tokens (needle PASS at 35K and 70K) |
| Boot | 13.2 min with 8-thread loading (27.5 min stock) |

## What this runs

The unquantized-trained GLM-5.3 checkpoint — Z.ai's own FP8 release, the best-fidelity build of the 744B model that exists
(QFS teacher-forced KLD 0.0223 vs BF16, vs 0.045–0.075 for the 4-bit builds; `malaiwah/quant-fidelity-suite`, comparator
`312a2551db89`). At 88 GiB/rank it does not fit four Sparks; it fits eight with ~9–11 GiB/rank left for KV.

Tensor parallel 8, one rank per Spark, the in-checkpoint MTP layer as the speculative drafter, `fp8_ds_mla` KV, DSA sparse
attention via the B12X kernels. Single seat in practice (`--max-num-seqs 4`).

This is 0xMagnus's FP8 TP8 launcher (`im0xMagnus/glm-5.3-uncensored-8x-dgx-spark` @ `dd5fff7a`) pointed at the **official**
weights, with MTP re-enabled using ciprianveg's v19 spec config, and an 8-thread weight loader. Every delta is marked
`MILO:` in `scripts/launch-fp8-tp8.sh`.

## Hardware

See `hardware/dgx-spark-8x.yaml`. Required before boot:

- Kernel `6.17.0-1031-nvidia` (apt-held) on all eight. `7.0.0-1019` breaks NCCL RoCE registration at this memory load.
- Per node: the cabled QSFP cage's two PCIe functions on separate /24s (`enp1s0f0np0` 10.0.0.N, `enP2p1s0f0np0` 10.0.1.N),
  MTU 9000. They are one 200G wire, not two rails; list both in `NCCL_IB_HCA` or you get ~100G.
- Switch: one bridge with hardware offload on the 8 lane-master sub-ports only, forced `200G-baseCR4`, FEC91, autoneg off,
  QoS trust off, PFC off. A sub-port in a second bridge drops you to switch-CPU forwarding (~10 Gb/s).
- Fabric gate: `scripts/run_bench.sh "1 2 3 4 5 6 7 8"` → all_reduce busbw ≥ 20 GB/s at 1 GiB. Measured **20.13 GB/s**
  (23.49 at 32 MiB), zero switch error/drop/pause counters, switch CPU 1%.
- `vm.swappiness=10`. Power: ~1.5 kW for the rack under prefill at stock clocks — one 15 A strip will not carry all eight.

## Software

- Image `ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision` @ `sha256:7ac0031c…` (loaded Id on overlay2 hosts
  `sha256:ba79eedf…`; the launcher refuses anything else). arm64/sm_121 only.
- vLLM `0.11.2.dev280+gilded.gnosis.v18`; NCCL 2.30.4 preloaded by the image's own `LD_PRELOAD`.
- Weights: one Hub pull, then fan out over the fabric (here: NFS export from one node, 8-way parallel copy, ~8 GB/s
  aggregate, ~13 min for seven copies), then sha256 every LFS file against the Hub manifest on every node. 143/143 on all
  eight (`results/…/raw/weight-fanout-verify.log`, `raw/hub-lfs.sha256`).

## Launch

```bash
export SPARK_HOSTS="spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8"   # ssh hosts in rank order; rank 0 = head
# on the head only: ~/glm53-tp8/.env.glm53 containing GLM53_API_KEY=<random>, mode 600
bash scripts/cluster.sh push    # copies launch-fp8-tp8.sh + bootflush.sh to ~/glm53-tp8/ on every node
bash scripts/cluster.sh up      # drop caches, start the 50-min page-cache flusher, launch ranks 7..1 then 0
bash scripts/cluster.sh status
```

Per-node: `MODEL_HOST` (default `$HOME/models/GLM-5.3-FP8-zai-ba32f8c1`), `GMU` (0.86), `MAXLEN` (131072). The API listens
on the head at `:8890` with a bearer key. Fabric IPs are `10.0.0.(rank+1)`; edit `HEAD_IP`/`HOST_IP` if yours differ.

Boot timeline (boot 2, head): weights 486.9 s + MTP head 10.9 s (model load 566.9 s total, 90.96 GiB/rank) → torch.compile
9 s (cache warm) → KV profile → CUDA graphs (6 sizes, ~65 s) → ready at 13.2 min. FlashInfer autotune is skipped on this
stack (no FlashInfer compute kernels active). Workers finish loading ~2× faster than the head; the head is the long pole.

## Verify

```bash
BASE_URL=http://<head>:8890/v1 GLM53_API_KEY=... python3 scripts/gates.py     # keyless 401, models, smoke, tool 2-round
BASE_URL=... GLM53_API_KEY=... python3 scripts/bench.py <tag>                 # C1 3x3 + C4
BASE_URL=... GLM53_API_KEY=... python3 scripts/needle.py 64000 0.5            # retrieval
```

Gates passed 2026-10-16 on both boots: keyless `/v1/models` 401; `glm-5.3-fp8` / 131072; exact `ROUTE_OK`; `get_weather
{"city":"Telluride"}` with `finish_reason: tool_calls`, then a correct answer from the tool result; needles at 35,055 and
69,979 tokens.

**Quality gate: pending.** The weights are byte-identical to the official release and MTP is target-verified, so nothing
in this recipe is expected to change the model function — but that claim has not been measured on this stack with a
teacher-forced MTP-on vs MTP-off comparison. Until it is, the status stays `experimental`.

## Results

Run `results/2026-10-16-fp8-tp8-baseline-mtload/` (`throughput.csv`, raw jsonl, boot log excerpts, wall power, all eight
system snapshots).

C1 decode — streamed, T=0, `reasoning_effort: low`, 800 max tokens, 1 discarded warm + 3 reps, decode = (tokens−1) /
(last-token − first-token time):

| prompt class | boot 1 (stock loader) | boot 2 (as shipped) |
|---|---:|---:|
| code (nginx parser + pytest) | 30.32 (29.27–30.36) | **29.92** (29.41–30.08) |
| technical explanation | 22.91 (22.16–23.52) | **23.10** (22.90–23.62) |
| prose (email) | 25.75 (25.04–26.24) | **26.87** (25.73–26.90) |
| C4 aggregate | 42.4 | 40.1 |

The loader change did not move decode (all within boot-to-boot noise). MTP's adaptive controller sits at depth 2 on
prose (mean acceptance length 2.0–2.3) and 4–5 on code (4.1–4.9 accepted per step).

| other | value |
|---|---|
| prefill, 35,055-token needle | PASS, 40.3 s (~870 tok/s) |
| prefill, 69,979-token needle | PASS, 57.5 s (~1,217 tok/s) |
| KV pool | 149,824 tokens (boot 1, 8.82 GiB/rank) · 181,632 (boot 2, 10.85 GiB/rank) |
| cold boot | 27.5 min stock → 13.2 min (8-thread loader + warm compile cache) |
| wall power peak, decode + 70K prefill | 169–189 W per Spark; Sparks 1–6 1,069 W; rack ~1.5 kW |

For comparison (third-party, same image class): 0xMagnus measured 22.8 tok/s with MTP / 15.2–15.6 without on the
dealignai uncensored FP8 build at TP8 on GB10.

## Known limits

- **reasoning `high` on a small budget returns nothing.** 1,500 max tokens all went to thinking — `finish_reason: length`,
  empty content. Reproduced on the official weights, so it is not an abliteration artifact. Use `low` (the server default
  here) or budget ≥16K for `high`. The template honours only `low`/`high`; anything else is `max`. Reasoning arrives in
  `message.reasoning`.
- **Context is ~131K, not 1M.** Weights + MTP head take 90.96 GiB/rank. Retrieval was tested to 70K; 128K is untested —
  the head node has ~6.8 GB of host headroom while serving.
- **Decode is latency- and weight-read-bound, not bandwidth-bound.** Light Foundry's `VLLM_ENABLE_ROCE_ALLREDUCE`
  (their biggest single-stream lever, on Int4-Int8Mix) does not exist in this image. A small-message all_reduce probe
  (Tree/LL) read 81 µs at 12 KB; at ~156 all-reduces per target step that is an *estimated* ~13–14 ms of an ~80–150 ms
  step. The default/Ring arms of the probe did not complete, so no NCCL algorithm comparison is claimed.
- **Boot is CPU-bound on the head.** Every rank reads all 704 GiB to keep its 88; the loader is one thread at 100%.
  8 threads cut the head load 957 → 567 s. Pre-sharding (`--load-format sharded_state`) is the next lever and is untested
  with FP8 + MTP on this fork.
- KV pool varies 10–20% boot to boot.
- Effectively one seat: C4 is 40–42 tok/s aggregate.

## Rollback

`bash scripts/cluster.sh down` removes the container on all eight ranks (head first). The recipe changes nothing else on
the nodes; the page-cache flusher exits on its own after 50 minutes.

## Credits

- **0xMagnus** — `im0xMagnus/glm-5.3-uncensored-8x-dgx-spark`: the FP8 TP8 launcher this recipe is a delta of, the
  measured gmu ceiling and KV arithmetic, and the reasoning-effort / empty-answer traps.
- **ciprianveg** — `gb10-vllm` and the `gb10-glm-5.2:v19-vision` image; the v19 spec-decode config.
- **wrldsuksgo2mars** and the b12x stack — `B12X_MLA_SPARSE` kernels that make DSA/MLA run on SM121.
- **local-inference-lab** — the vLLM fork (DCP1 query split, PR #175) inside v19.
- **CosmicRaisins** — adaptive MTP 2/4/5 controller.
- **Light Foundry** — public 8× GB10 TP8 GLM-5.3 numbers and the RoCE all-reduce finding.
- **eugr** — `spark-vllm-docker`.
- **Z.ai** — GLM-5.3 and its official FP8 release.
- Fabric rules from itadmin.blog, the NVIDIA DGX Spark forum builds, and MikroTik's CRS8xx breakout docs.

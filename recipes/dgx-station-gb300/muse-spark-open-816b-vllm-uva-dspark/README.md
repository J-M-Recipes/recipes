# Muse Spark Open 816B-A42B (NVFP4) on one DGX Station GB300

**Status: experimental** (first measurement round, October 15–16, 2026) · **29.4 tok/s** single-stream (prose; JSON 29.7, agent 29.3, code 26.2) · **49 tok/s aggregate at C4** · 128K context · **one GPU**, 310 GiB of experts in Grace memory · boot to `/health` in **~6 min** · quality A/B vs no speculation: **BFCL 563 vs 561 /600**, **GSM8K-200 193 vs 192**

*Built with Muse Spark Open.* (Attribution required by the Muse Model License.)

## What this runs

Meta's **Muse Spark Open** (816B total, 42B active, 256 routed experts top-8, 62 layers, vision + audio towers) in the **first-party NVFP4 checkpoint** (`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`, 496 GiB of weights) on **one GB300** with 250.7 GiB of usable HBM. It does not fit. vLLM's UVA offload backend keeps **310 GiB of expert weights in Grace LPDDR5X** and the GPU reads them over NVLink-C2C at decode time. The Inferact **DSpark** drafter guesses **2 tokens** per step.

Engine: the vLLM day-0 image `vllm/vllm-openai:muse-spark-open` (vLLM `0.1.dev22954+ga34ca64c1`, PR #61879, unmerged at test time), pulled by digest. Nothing re-quantized, no source patches.

## Hardware

Profile: [`hardware/dgx-station-gb300.yaml`](../../../hardware/dgx-station-gb300.yaml) · snapshot: [`results/2026-10-16-muse-s1/system.json`](results/2026-10-16-muse-s1/system.json)

| | observed |
|---|---|
| GPU | NVIDIA GB300, 250.7 GiB visible (nvidia-smi 256,703 MiB) |
| Host | Grace, 494 GiB LPDDR5X, NVLink-C2C |
| OS / kernel / driver | Ubuntu 24.04.5 · 7.0.0-1019-nvidia-64k · 595.91.07 |
| Storage | checkpoint staged to local RAID0 NVMe (`/models`), sha256-checked; never served from the NAS share |
| Do not | set `CUDA_VISIBLE_DEVICES` — DGX OS's docker config already hides the display GPU |

## Software

| | pin |
|---|---|
| Image | `vllm/vllm-openai:muse-spark-open` @ `sha256:c91025b5be1039a93ca975d1d44f352b0e770bb6c8b142b5056cf92204ca137c` (created 2026-10-14), pulled by digest |
| vLLM | `0.1.dev22954+ga34ca64c1` = commit `a34ca64c138ad0360c472ada748b750435b607ae` (Muse support, PR #61879) |
| Libraries | torch 2.13.0+cu130 · transformers 5.19.0 · FlashInfer 0.7.0.post1 · triton 3.7.1 |
| Target | [`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) @ `4239e4db19376904f2e82c167e78f70abb5455fe` (gated, manual approval; 141 files, 533,043,429,228 B) |
| Draft | [`Inferact/Muse-Spark-Open-DSpark`](https://huggingface.co/Inferact/Muse-Spark-Open-DSpark) @ `ffc3eb50c8ae7a5a391ddedc43085d2e5249bc35` (10,604,912,459 B) |
| Hook | [`scripts/sitecustomize.py`](scripts/sitecustomize.py) was bind-mounted in every measured run; it only redirects the autotune cache directory and is inert with autotune off |

## Launch

```bash
MODEL=/models/Muse-Spark-Open-816B-A42B-NVFP4/4239e4db19376904f2e82c167e78f70abb5455fe/original \
DSPARK=/models/Muse-Spark-Open-816B-A42B-NVFP4/dspark/ffc3eb50 \
bash scripts/launch-muse.sh
```

The four settings that matter:

| Setting | Why |
|---|---|
| `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` | torch `pin_memory` rounds each buffer to a power of two, so 300+ GiB of experts would pin ~2× that. The pre-boot sizing said **park** (≈446 GiB load peak of 494). With pinning off, the lowest free host memory during the 10 boots that recorded it was **116–142 GiB**. |
| `--offload-backend uva --cpu-offload-gb 310 --cpu-offload-params experts` | 310 makes room for the 11.3 GiB drafter in HBM. Each extra 10 GiB over C2C cost ~3% C1 (no-spec 26.6 at 300 → 25.8 at 310). |
| `--kernel-config '{"enable_flashinfer_autotune":false}'` | FlashInfer MoE autotune took **1 h 48 min per boot** (6,449–6,510 s) and bought nothing: no-spec C1 26.58 off vs 26.56 on. See below. |
| `--speculative-config '{"method":"dspark",…,"num_speculative_tokens":2}'` | k=2 beats k=3 and k=7 on this box. See below. |

CUDA graphs on (no `--enforce-eager`): **10.6 → 26.6 tok/s** single-stream.

## Results

Run `2026-10-16-muse-s1`.

All rows: T=0, C1 = 3 reps × 256 forced tokens (streaming, `min_tokens` + `ignore_eos`), C4 = 4 concurrent prose streams × 2 reps. Raw per-rep numbers: [`results/2026-10-16-muse-s1/throughput.csv`](results/2026-10-16-muse-s1/throughput.csv).

| Boot | Offload | Graphs | Autotune | Spec | C1 prose / code / JSON / agent | C4 agg | Ready |
|---|---|---|---|---|---|---|---|
| s1-boot3-at | 300 | eager | on | — | 10.65 / 10.64 / 10.66 / 10.63 | 39.9 | 6,510 s |
| s1-boot4-graphs | 300 | on | on | — | 26.56 / 26.55 / 26.57 / 26.57 | 43.4 | 6,449 s |
| s1-boot6-graphs-noat | 300 | on | **off** | — | 26.58 / 26.58 / 26.57 / 26.58 | 42.6 | 345 s |
| s1-boot5-dspark | 310 | on | on | DSpark k=7 | 22.91 / 21.22 / 20.45 / 21.82 | 37.8 | 6,490 s |
| s1-boot7-dspark-k3 | 310 | on | off | DSpark k=3 | 28.67 / 24.99 / 29.91 / 28.74 | 48.0 | 366 s |
| s1-boot8-dspark-k2 | 310 | on | off | DSpark k=2 | 29.41 / 26.16 / 29.64 / 29.29 | 49.1 | 370 s |
| **q-k0** (control) | 310 | on | off | — | 25.83 / 25.84 / 25.82 / 25.83 | 41.4 | 346 s |
| **q-k2** (**recipe**) | 310 | on | off | **DSpark k=2** | **29.44 / 26.18 / 29.67 / 29.31** | **49.2** | 366 s |

DSpark acceptance (vLLM `/metrics`, accepted tokens per step incl. the bonus token): k=7 **3.17** (per position 0.75 / 0.50 / 0.34 / 0.24 / 0.16 / 0.11 / 0.07), k=3 **2.85** (0.79 / 0.60 / 0.46), k=2 **2.42** (0.80 / 0.62) on the bench prompts, **2.53** (0.84 / 0.69) over BFCL + GSM8K traffic.

### Finding 1: on an expert-offloaded MoE, short drafts win

Divide accepted tokens per step by tok/s and you get the cost of one verify step: **~38 ms** with no speculation, **~82 ms** at k=2, **~99 ms** at k=3, **~138 ms** at k=7. k=7 accepts the most tokens per step and is still **11% slower** than no speculation. Our explanation, not yet profiled: checking k+1 tokens routes them to more distinct experts per layer, and most experts are read from Grace over C2C, so the verify cost grows with the draft much faster than it does on an HBM-resident model. On DeepSeek-V4.1-Flash on this same box, a k=5 draft window was measured fetching 3.7× more unique experts per step, which is the same mechanism.

### Finding 2: FlashInfer autotune is noise here

With autotune on, every boot shape re-profiled all 43 MoE entries (~44 min BF16 + ~59 min FP4), even when the previous cache file loaded and its keys matched. Comparing the cache after two boots: same 43 keys, identical metadata, **32 of 43 picked a different tactic**. When the timings are dominated by reads from Grace there is no stable winner, and the default tactics are as fast. Turning it off cut boot from ~1 h 48 min to ~6 min with no speed change.

### Quality A/B (same window, only the speculator differs)

| Test | No spec (q-k0) | DSpark k=2 (q-k2) | Bar |
|---|---|---|---|
| BFCL dev 600 (simple_python 400 + multiple 200), our grader, low effort, C4 | 561 (93.5%) | **563 (93.8%)** · 5 fixed / 3 broken, exact McNemar p ≈ 0.73 | ±2 pts ✅ |
| GSM8K first 200, low effort, C4 | 192 (96.0%) | **193 (96.5%)** · 2 fixed / 1 broken | ±2 pts ✅ |
| Greedy agreement, 20 prompts × 600 tokens | — | 13/20 identical; 7 diverge at token 6–134 | reported |
| Gates (models, chat, tool call, 4-turn replay, greedy repeat) | 6/6 | 5/6 — `reasoning_split` reasons past the gate's 2,000-token cap | — |
| BFCL wall time | 3,521 s | **2,852 s (−19%)** | — |

## Verify

1. `curl -fs http://127.0.0.1:18100/health` returns 200 in ~6 min. The log shows `Total CPU offloaded parameters: 311.5`, `Model loading took 207.88 GiB`, `GPU KV cache size: 179,747 tokens` and `Skipping FlashInfer autotune because it is disabled`.
2. `python3 scripts/muse_gates.py http://127.0.0.1:18100/v1 <outdir>` passes models, chat, the tool call, the 4-turn replay and the greedy repeat; C1 prose is within 3% of 29.4 and C4 within 5% of 49.2.
3. For a quality change, run `scripts/quality-chain.sh` (control and candidate in one window): BFCL and GSM8K within ±2 points of the control.

## Known limits

See `limits` in [`recipe.yaml`](recipe.yaml). The short version:

- C1/C4 decode the template's **default reasoning effort (`xhigh`)**. The 256 forced tokens are reasoning text, which is why every no-spec class reads the same.
- DSpark is target-verified, **not byte-identical** to no-spec decoding. Quality rests on graded evals; no teacher-forced logprob gate was run.
- Code gets no speedup at k=2. The drafter is text-only on this image. Image and audio input were not gated.
- k=7 + adaptive verification failed at boot (not enough KV for one 128K request at offload 310).
- Preview image. Re-gate on the first tagged vLLM release with Muse.

## Rollback

`docker stop muse`, then relaunch without `--speculative-config` (no-spec, C1 25.8 tok/s at offload 310). Same ~6 min boot.

## Next

**Longer context.** The checkpoint allows 1,048,576 tokens. The current KV pool is 179,747 tokens with DSpark (1.37× one 128K request), 342,996 without.

## Files

- [`scripts/launch-muse.sh`](scripts/launch-muse.sh) — the recipe launch.
- [`scripts/muse-boot.sh`](scripts/muse-boot.sh) — the measured runner: refuses if the GPU is busy, host-memory watchdog, gates, stop-and-keep.
- [`scripts/muse_gates.py`](scripts/muse_gates.py), [`scripts/bfcl_gate.py`](scripts/bfcl_gate.py), [`scripts/gsm_eval.py`](scripts/gsm_eval.py), [`scripts/s2_greedy.py`](scripts/s2_greedy.py), [`scripts/quality-chain.sh`](scripts/quality-chain.sh), [`scripts/chain-dspark.sh`](scripts/chain-dspark.sh).
- [`results/2026-10-16-muse-s1/`](results/2026-10-16-muse-s1/) — every boot's runner log, argv, gates, `/metrics`, memory watch, gzipped container log; quality jsonl; autotune cache before/after; model provenance; `SHA256SUMS`.

Blog: https://al-engr.com/gb300-muse-spark-open.html

# Failure ledger — DeepSeek-V4.1-Flash on one GB300

What was tried, what it did, and why it isn't in the recipe. Dates 2026-09-09 → 09-11, CDT.

## Engine

| attempt | result | why |
|---|---|---|
| SGLang day-0, expert offload | **3.3 tok/s** | copies whole expert layers host→GPU every step; not a fit for offloaded MoE on this box |
| vLLM UVA v1 | crash | `--cpu-offload-params` matcher didn't hit the expert tensors |
| vLLM UVA v2 | boot >1 h, abandoned | weights on CIFS thrashing under 4 parallel readers |
| vLLM UVA v3 | host OOM | 486 GB shmem: offload + Engram + shard staging all pinned at once |
| vLLM UVA v4 (offload 40) | `Available KV cache memory: -6.96 GiB` | 40 GiB off the GPU is not enough at 131K |
| vLLM UVA v5 (offload 60, 131K, no spec) | **first bind, 85 tok/s** after a 74-min autotune | this became the baseline |

## Offload bracket (`--cpu-offload-gb`)

| GiB | 131K no-spec | 131K + DSpark | 1M + DSpark |
|---|---|---|---|
| 40 | −6.96 | — | — |
| 60 | 12.03 ✅ | — | **−2.61 ❌** |
| 70 | — | 15.56 ✅ | **10.05 ✅** (recipe) |

60→70 cost ~8% at 131K without spec. 65 was never worth a boot: nothing else freed 2+ GiB.

## Autotune hash misses

New engine-config hash → empty dir → 74-min retune. Missed on v7 (`1e679143`), v8 (`79e8fd7f`), v10 (`08c89d94`), e3 (`b24a0a93`). Each cost a kill + relaunch. A watcher script landed 4 s late once. Fix: seed before launch, or disable autotune in `--kernel-config`. Receipt for success: `Loaded 210 configs`.

## Speculative decoding

| config | prose C1 | agent weighted accept | note |
|---|---|---|---|
| no spec (v6) | 93.2 | — | best prose, C16 336 |
| DSpark k=5 (v7/v10) | 78.9–82.1 | 59.6% | shell 150 / code 146 / tool-json 131 |
| DSpark k=3 (v8) | 90 | 70.1% | fewer tok/step; k=5 wins agent text ~10% |

k=5 adopted for agent traffic. Prose users would take k=3 or no spec.

## Mixed-traffic starvation

Without `--long-prefill-token-threshold`: a 480K prefill delays four short requests by 18.2 / 13.2 / 8.2 / 3.2 s. `--max-num-partial-prefills` / `--max-long-partial-prefills` → `unrecognized arguments` in this vLLM. `--long-prefill-token-threshold 6144` → 0.89–1.09 s. Long prompt +16%. Adopted.

## Overnight decode experiments (2026-09-10/11)

Win bar was set on `replay.py` (+5%), which turned out to have ±17 tok/s spread. Verdicts re-read on the knee:

| exp | change | mechanism check | knee C1 vs 82.1 | verdict |
|---|---|---|---|---|
| E5 | prefix cache (measure) | hits 0→34k | — | 8.8K system prompt: 0.665→0.360 s warm |
| E1 | THP `always` | AnonHugePages 0.5→**2.5 GB** (needed ≥50) | 78.4 | pinned `cudaHostRegister` buffers are not THP-eligible; sysfs alone does nothing |
| E2 | `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` | host used 450→**384 GB** | **76.3 (−7%)** | pageable+ATS is slower; 66 GB host RAM back isn't needed |
| E3 | + `--language-model-only` | KV 10.05→**11.14** | 76.2 | +1.1 GiB, not the 3 needed for offload 65 |
| E4 | + `VLLM_USE_RUST_FRONTEND=1` | tool parser survived | 76.0 | no decode gain; 6.5K TTFT 0.37→0.52 s |
| E1b | THP `always` on unpinned path | AnonHugePages still **2.5 GB**; weights are in **Shmem** (354 GB), `shmem_enabled=[never]` | **77.3** | THP `enabled` was never the relevant knob; the untested lever is `transparent_hugepage/shmem_enabled` |

Lane rolled back to v11 at 05:56 CDT. Report and correction: the overnight operator followed the plan exactly; the wrong instrument was in the plan.

## Not pursued, with reasons

- **Re-quantizing.** Experts are already MXFP4, Engram FP8. Every "V4-Flash NVFP4/AWQ/GGUF" on the Hub is for the FP8-expert 0731 predecessor. llama.cpp has no Engram/CED support. No lever.
- **Engram on NVMe.** Community did it with 64 GB RAM. We have 494 GB; it would only add latency.
- **C4 bimodality** (two clusters in C4 knee runs). Interesting, not actionable.
- **REAP expert pruning 384→256.** The one structural lever left — would put all experts in HBM and remove the C2C term. Multi-day; not started.

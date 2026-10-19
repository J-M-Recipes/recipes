# 2026-10-18 — v23 "Pin Exact": exact-size pinned host allocations (PROMOTED 10:21 CDT)

One DGX Station GB300 (250.7 GiB HBM, 494.5 GiB Grace LPDDR5X). v23 is the v22 launch plus one environment variable:

```
-e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024
```

torch 2.13's caching host allocator rounds every pinned (`cudaHostAlloc`) allocation up to a power of two. With this
setting, allocations above 1024 MB are pinned at their exact size. The two Engram tables (94.42 GiB each, plus scales)
had been pinned into 2 × 128 GiB + 2 × 4 GiB blocks; on v23 they are 2 × 91.55 GiB + 2 × 2.86 GiB. The hook's
cold-expert maps are ≤ 1 GiB, so they are still rounded and cached, unchanged. Kernels, autotune hash, KV pool and the
HBM/Grace expert split are unchanged.

## Provenance

| | pin |
|---|---|
| image | `vllm/vllm-openai:deepseekv41-flash-0909` @ `sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d` (engine `0.1.dev20904+g179dd0fa9`; `docker image inspect` on the box, 2026-10-18) |
| model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, 253 files, 510,311,114,235 bytes on local NVMe (stat sum, 2026-10-18) |
| hook | pin-hot-experts v21, `pin_hot_experts_hook.py` sha256 `b5e4f4f852b9dd3b510b7acda191f30255f105aa0cfda41fa222f48bc37729f4`, `rowmap-static-v1.json` sha256 `550c3202b7a0b6c78865efdc0a246490b2810f8e60ac50d617e0cc72aaa9faf9` (both `docker exec … sha256sum` inside the serving container) |
| env (in-container) | `PIN_MODE=split PIN_HOOK=/w/pin_hot_experts_hook.py PIN_ROWMAP=/w/rowmap-static-v1.json PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024` (`docker inspect`) |
| autotune | FlashInfer hash `62426808…`, `Loaded 189 configs` on every boot (hash hit) |
| KV | 2,504,891 tokens at 1,048,576 context (image-default KV dtype) |
| exact launch | [`scripts/launch_v23.sh`](scripts/launch_v23.sh); one-line diff vs v22: [`launch_v23-vs-v22.diff`](launch_v23-vs-v22.diff) |

## Host memory, before and after

Pinned = sum of `Size:` over the EngineCore `/dev/zero (deleted)` maps in `/proc/<pid>/smaps` (equals cgroup shmem to 0.01 GiB).

| | v22 | **v23** |
|---|---|---|
| pinned host total | 327.96 GiB | **252.79 GiB (−75.17)** |
| MemAvailable, steady (after drop_caches) | 118–124 GiB (131–133) | 193–196 GiB (206) |
| hook guard `host_avail` at layer-9 rehome (floor 10 GiB) | 11.3–12.1 GiB | **197.7 GiB** |
| minimum `host_avail` during rehome | 11.3–11.5 GiB | 152.7–153.6 GiB |
| HBM experts / pinned experts | 206.61 / 62.33 GiB | 206.61 / 62.33 GiB |

The boot guard margin was 1–2 GiB above the 10 GiB floor (the hook aborted three boots on 2026-10-05/06) and is now
~140 GiB above it. The guard gain (+186 GiB) is larger than the 75 GiB steady saving; the likely reason (freed > 1 GiB
pinned blocks held by the rounding allocator during load/rehome) was not measured.

## Card S0 — bracketed CTL1 → S0 → CTL2 (08:56–09:48 CDT)

Real-text harness = RonanLabs `kbench_chat_ds.py` (CNN/DM ~8.2K in, 1,024 forced out, thinking off, T=0, nonce, 0% prefix hits).

| arm | RT C1 (accept) | C4 | C16 | C32 out | TTFT 8K | knee r2 C1/C8/C16 | replay r2 | gauntlet | KV |
|---|---|---|---|---|---|---|---|---|---|
| CTL1 (live v22) | 186.5 (2.97) | 318.7 | 522.9 | 504.4 | 0.66 | 192/618/937 | 441 | 12/12 | 2,504,891 |
| **S0 = v23** | **190.2 (3.00)** | 325.1 | 523.5 | 506.5 | 0.65 | 192/590/942 | 438 | 12/12 | 2,504,891 |
| CTL2 (v22 restart) | 193.5 (3.02) | 335.9 | 540.2 | 515.8 | 0.63 | 193/620/947 | 437 | 12/12 | 2,504,891 |

Pre-registered bars (S0 vs mean of controls): pinned ≥ 40 GiB lower **PASS (−75.17)** · RT C1 within −2% **PASS (+0.1%)** ·
replay r2 within −2% **PASS (−0.2%)** · gauntlet 12/12 **PASS** · greedy non-tool parity **PASS — 16/16 vs CTL1 and 16/16
vs CTL2** (e2c_parity, 18 prompts; every pair incl. CTL1↔CTL2 is 17/18 overall, the miss is the known tool-call jitter
prompt) · KV unchanged **PASS**. Fail bars (decode −3% on C1/C4/C16/replay r2: +0.1 / −0.7 / −1.5 / −0.2%; boot death)
clear. Error grep 0 on all arms. Supplement S0b (09:29–09:48, `receipts/S0/supp/`): N=3 cold-64K probes show the first
request after any boot is ~1 s slow on every arm (S0 3.10/2.16/2.17 s, CTL2 3.05/2.15/2.15, live 2.13/2.12/2.11), so the
main card's cold-64K gap was a first-request artifact; replay r1 is ~25% low on every fresh boot (compare r2 only).

## The C8 question — Card S0c confirm (10:10–10:22 CDT)

S0 read prose knee C8 590/591 vs 599–620 on the controls (−3.4%), outside the bar set. Confirm: A = live v22, B = v23
restart, same cells in the same order. Bar (pre-registered): knee6 r2 C8 ≥ −2%, knee6 r2 C6 ≥ −2%, RT C1 ≥ −2%.

| gated cell | v22 | v23 | Δ | bar |
|---|---|---|---|---|
| knee6 r2 C8 | 597.0 | 604.8 | +1.3% | PASS |
| knee6 r2 C6 | 535.8 | 529.6 | −1.2% | PASS |
| RT C1 (mean of 3) | 187.8 (188.3/185.6/189.4) | 188.2 (190.8/189.1/184.8) | +0.2% | PASS |

**Lesson: prose-knee C8 is bimodal.** Every C8 cell is a mean of two reps, and the reps fall into a fast mode
(~620–632) or a slow mode (~556–583) on every arm (v22 2 slow of 8, v23 3 slow of 8). Means over all four C8 cells:
611.9 vs 604.3 (−1.2%); fast-mode reps only 628.3 vs 625.1 (−0.5%). knee.sh C8 flipped sign between r1 and r2 on both
arms. The S0 −3.4% was the slow mode landing in r2. **Gate C8 on ≥ 4 reps per arm** (Card P2f does).

**Verdict PASS → promoted 10:21 CDT** as `dsv41-vllm-1017-v23-pinexact-BOUND-REF` on `:30006`. Gauntlet 12/12 on the
promoted container; the lane's chat path (`reasoning_effort: low`) and the guest gateway route answered 200.

## Later same-day v23 controls (for the card)

| window | RT C1 (accept) | TTFT 8K | knee6 C8 (4 reps) | knee r2 C16 | cold ~53K prefill | BFCL dev / held-out | GSM8K-200 |
|---|---|---|---|---|---|---|---|
| Card P2f A (live) | 186.1 (2.91) | 0.643 s | 596.5 | 944.4 | 24,094 tok/s | 92.33 / 77.65 | 97.0 |
| Card P2f D (restart) | 186.8 (2.94) | 0.647 s | 614.0 | 944.5 | 24,006 tok/s | 92.67 / 77.88 | 96.5 |

Receipts in [`../2026-10-18-p2-nightly-39983a5e-gate/receipts/P2f/`](../2026-10-18-p2-nightly-39983a5e-gate/receipts/P2f/).

## Rollback

`docker stop dsv41-vllm-1017-v23-pinexact-BOUND-REF && docker start dsv41-vllm-1017-v22-ksched1to8-RETIRED-REF` (~5 min, hash hit).

## Files

- `scripts/launch_v23.sh` — exact launch. `scripts/cardS0_runner.sh`, `cardS0b_runner.sh`, `cardS0c_runner.sh` — runners; `cardS0_verdict.py`, `cardS0c_verdict.py` — verdicts (pre-registered bars in the runner headers).
- `receipts/S0/` — VERDICT.md, SUMMARY.txt (arm table), `hostmem-*` (smaps/meminfo/cgroup per arm: bound, steady, steady after drop_caches), `facts-*` (boot lines: KV, rehome, autotune), `parity/`, `rt-*/` (kbench JSON), knee/knee6/replay JSON, agent fixture, gauntlet summaries, `supp/` (S0b). `receipts/S0c/` — the confirm.
- `throughput.csv` — every kbench and knee cell from both cards.
- Mirrored with a path scrub: box home → `~` (receipts) / `$HOME` (scripts), LAN addresses → host names. Container boot/post logs, window logs and gauntlet task sandboxes stay on the box.

Runs: claude-opus-5-5 workers from Milo's briefs. Write-up: Milo (Hermes, claude-fable-5-1).

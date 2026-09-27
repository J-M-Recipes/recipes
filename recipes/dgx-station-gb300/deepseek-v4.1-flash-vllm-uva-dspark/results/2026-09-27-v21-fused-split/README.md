# 2026-09-27 — v21 "Fused Split": one Triton kernel replaces six torch ops per MoE layer in the pin-hot-experts hook

One morning window on `:30006`, 0909 image, v18 launch (hook, off60, util 0.97, 24 seats, k-schedule `[[1,4,5],[5,24,1]]`, token-sized capture list). Three fresh boots, each loading the same saved FlashInfer autotune set (a Python-only hook change does not alter kernel shapes): **CTL1 (v18 hook) → CAND (v21 hook) → CTL2 (v18 hook)**, so any drift is bracketed. Receipts in [`receipts/`](receipts/); runner [`v19_ab_runner.sh`](v19_ab_runner.sh) (the arm was called `v19ab` on the box before the release number was assigned — the container is `dsv41-vllm-v21-fused-split-BOUND-REF`).

## What changed

`split_ids()` in the hook maps each routed expert id to its hot-tensor row (HBM) or cold-tensor row (Grace) before the two `do_finalize=False` routed-kernel calls. In v15–v18 that was six torch ops per layer (`>=`, `clamp_min`+index, `&` ×2, `full_like`, `where` ×2) — ~400 graph nodes per decode step across 40 layers, each a real ~2 µs kernel in the captured CUDA graph, serialized by data dependency. v21 does the same integer remap in one Triton kernel ([`v19_patch_split.py`](v19_patch_split.py) is the exact edit; [`hook/pin_hot_experts_hook.py`](hook/pin_hot_experts_hook.py) is the file the release mounts, sha256 `b5e4f4f8…`). `PIN_SPLIT_FUSED=0` selects the old path for an in-place A/B. Integer-only; no floating-point path is touched, so parity is expected and was gated anyway.

Bit-exactness against the torch reference ([`v19_test_split.py`](v19_test_split.py)): identical at T = 1, 6, 16, 96, 8192 with −1 padding; eager 157 → 19.5 µs per call at T=6.

## Why this was the lever

A torch-profiler capture of the exact v18 config ([`profile/`](profile/), [`campaign_profile_v18.sh`](campaign_profile_v18.sh), analysis [`prof_step.py`](prof_step.py)) put a C1 prose decode step at **14.6 ms p50, 101% GPU-busy, 0.7 ms inter-step gap** — no CPU bubble. Roles: MoE expert GEMM 5.0 ms (34%), dense GEMM 4.2 (29%), **small elementwise/routing 2.7 ms over ~1,300 launches (18%)**, attention 1.2 (8%), mHC 1.0 (7%), Engram 0.04. The hook's own `split_ids` ops were the largest attributable slice of that swarm. The same session closed the huge-page question first ([`t4_hugepage_bench.py`](t4_hugepage_bench.py)): GPU reads of pinned Grace memory run ~355 GB/s on 64 K pages and on 100 % hugetlb 512 MiB pages alike (random 256 B gathers 118 GB/s both ways) — C2C is bandwidth-bound, not translation-bound, and `cudaHostAlloc` regions ignore THP sysfs entirely.

## Result (knee.sh, prose, warm, r2 of two runs per boot; r1 within 0.4 % on every arm)

| | C1 | C2 | C4 | C8 | C12 | C16 |
|---|--:|--:|--:|--:|--:|--:|
| CTL1 (v18 hook) | 171.8 | 274.3 | 414.9 | 656.0 | 809.3 | 957.0 |
| **CAND (v21 hook)** | **182.8** | **287.0** | **433.7** | **686.8** | **845.0** | **992.8** |
| CTL2 (v18 hook) | 172.5 | 272.8 | 415.8 | 657.1 | 813.4 | 952.4 |
| CAND vs mean(CTL1, CTL2) | **+6.2 %** | +4.9 % | +4.4 % | +4.6 % | +4.1 % | +4.0 % |

Controls agree within 0.5 % at every concurrency. C16 r1 is the usual warm-cache artifact on all three arms (756 / 776 / 753) and is excluded as always; pair r2/r2.

Replay (`replay_c.py`, n=4 workers, two rounds, real Hermes transcripts): 342.7 / 450.4 (CTL1), 335.4 / 462.6 (CAND), 332.1 / 469.6 (CTL2) — a wash inside the instrument's ±20 %/run. Acceptance 2.53–2.71 accepted/step on every arm; agent fixture `weighted_accept` 0.6 on all three.

## Fidelity

Greedy capture on the 18-prompt E5 set, T=0, seed 42 (`e2c_parity.py`, [`receipts/parity/`](receipts/parity/)):

| pair | non-tool prompts identical | all prompts |
|---|--:|--:|
| CAND vs CTL1 | **16/16** | 17/18 |
| CAND vs CTL2 | **16/16** | 17/18 |
| CTL1 vs CTL2 (control-vs-control) | 16/16 | 18/18 |

The one differing prompt is `agent-tool-1`, a tool-call prompt — the class documented since v15 as jittering run-to-run on this lane (preamble/arguments). The two controls happened to agree on it in this window, so the honest statement is: consistent with the known tool-call jitter, not proven to be it by this window alone. The change is an integer index remap with no arithmetic path, and the fixture's acceptance rates and `weighted_accept` are unchanged. Held-out BFCL sibling on v21: **pending** (not run in this window).

## Verdict

Promoted 2026-09-27 08:12 CDT as `dsv41-vllm-v21-fused-split-BOUND-REF`; `dsv41-vllm-v18-cgsizes-BOUND-REF` renamed `-RETIRED-REF` and kept stopped as the rollback (`docker stop <v21> && docker start <v18>`). Bar was > 1.5 % same-window with non-tool greedy parity; measured +6.2 % C1 / +4–5 % at C2–C16 with 16/16.

**Base note.** When this session started (2026-09-27 05:00 CDT) the lane was `dsv41-vllm-v18-cgsizes-BOUND-REF` (0909 image), and the v20 container had been stopped since 2026-09-21 20:57 CDT under the name `…v20-clean-nightly-2671fedf-RETIRED-REF-mimo-campaign-20260921`; the ledger has no entry for that swap, so the reason is not recorded here. This window therefore measured v21 against v18 on the 0909 image. The hook file is the same one v20 mounts, so v20 + fused split is expected to compose, but it is **unmeasured** — see `limits`.

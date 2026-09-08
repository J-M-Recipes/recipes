# Window plan — E0 telemetry · E1 nsys decomposition · E5 MTP K=2 probe

Status: **PREPARED; launch authorized; nothing launched yet.** The separate explicit window go has
been received; live mutation remains blocked until the frozen Station-side `CONTROL=RUN` and exact
matching `RELEASE` gate are established.
Written 2026-09-07 ~22:20 CDT after commit `df28147` (offline E2/E3-A STOP published).

## 0. What changed while writing this plan (read first)

The live 512K lane was inspected read-only. It is running with `SLOT_CACHE_STATS_SEC=20`, so the
stats thread **is already emitting per-window hit telemetry** on this container (343 windows so far).
The earlier "stats hook breaks graph capture" incident was on the staged DMA repo copy; the
production lane launched fine with stats on. So E0 does **not** need a relaunch — it needs an
offline denominator correction applied to the existing log plus a controlled load.

Aggregated from the live log (128,319 layer-steps across 331 busy windows):

| Assumption | Implied hit rate |
|---|---:|
| Hook divisor = 8 uses/step (M=1, pre-MTP assumption baked into the hook) | **0.4075** |
| MTP(1) verify = 2 tokens/step → 16 uses/step | **0.7038** |
| Offline replay of the same slot map (E2 control) | 0.7097 |
| Planning figure in `slots-5792-ctx512k.json` | 0.6166 |

The hook hard-codes `/8`. Under MTP(1) every decode step verifies 2 tokens, so the printed
`HIT=` lines understate reality by ~30 points. The M=2 reading (0.70) agrees with the offline
replay (0.71) to within a point — that is the first live/offline cross-check we have, and it
retires the 0.6166 planning number. **E0 becomes a measurement, no relaunch.** The corrected hook
now counts actual routed expert uses for future launches; the current container remains untouched.

Per-layer signal from the same log, worth carrying into E1: layer 4 is consistently the worst
(negative windowed "hit" = thrash), layers 35/39 the best (0.79–0.88).

## 1. Frozen gates (freeze before the window opens; do not edit after)

| Exp | Gate | Pass → | Fail → |
|---|---|---|---|
| **E0** | Live hit rate under standard prose+code bench and a ≥20-min real Hermes session, corrected using observed MTP draft/accept metrics | publish aggregate + logged extrema; feed E1 | n/a — measurement |
| **E1** | Valid nsys graph-node capture covering ≥50 K=1 verification steps, with adjacent unprofiled control. Report aggregate GPU work, CUDA launch API time, wall residual, and stream overlap separately; never force kernel sums to equal wall time | if directly measured bookkeeping + scalar-gather + attributable launch API work is ≥ **2.0 ms/step**, and profiler slowdown is ≤20%, authorize E4 later | if <2.0 ms with valid low-distortion evidence → campaign closed; if profiler slowdown >20%, mark E1 INCONCLUSIVE rather than claiming a ceiling |
| **E5** | MTP K=2 weighted accepted length on the fixed 4-prompt probe ≥ **K=1 measured + 0.15** | proceed to E5-bench: C1 ≥ +5% over matched K=1, and primary code/math gate (190/200-class) does not regress | stop; one line in README |

Global rules (unchanged from DFlash2/DMA contracts): one candidate container at a time, the
preserved lane `glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907` is stopped-not-removed
during candidates and restarted at the end, `AT_KEY=slotcache-S112` for every launch, restore
verified by `/v1/models` (`glm-5.3-big`, 524288) + exact-string completion, receipts + SHA256SUMS,
publish wins and losses alike. **A stop is not a restore authorization; the DSF `:30003` restore is
a separate James instruction after the window.**

## 2. Timeline (≈4 h wall; three relaunches max)

| T+ | Step | Relaunch? | Notes |
|---|---|---|---|
| 0:00 | Freeze this file; snapshot `docker ps -a`, `nvidia-smi`, current lane inspect | no | preflight |
| 0:05 | **E0-a** run standard bench (prose ×3, code ×1 at C1/C4/C8) against the *live* lane; capture stats windows | no | same `bench_big.py` harness as DMA |
| 0:35 | **E0-b** 20-min real Hermes session (`hermes -m glm-big ...` tool loop, ≥4 replayed tool turns) against live lane; capture windows | no | routing on tool traffic ≠ bench |
| 0:55 | **E0-c** run the fixed 4-prompt/512-token acceptance probe on incumbent K=1 and read `/metrics`; this is an E0 telemetry slice, not the E5 comparator | no | |
| 1:00 | Stop live lane (preserve) | — | window proper begins |
| 1:05 | **E1** launch `glm53-big-e1-nsys` = exact 512K K=1 profile with opt-in in-container `nsys profile`; while armed but not collecting, freeze the patched-tree K=1 512-token acceptance and prose ×3/code ×1 benchmark comparator | **1** | Nsight 2025.6.3 full install tree is bind-mounted read-only; only this arm receives `SYS_ADMIN`/unconfined seccomp |
| 1:40 | Run one adjacent unprofiled 64-token probe, then trigger profiler START/STOP through the engine-core control file around the same fixed probe | — | capture must cover ≥50 verification steps |
| 2:00 | Export `cuda_gpu_kern_sum`, `cuda_kern_exec_sum`, and traces. Bucket routed MoE · row copy · bookkeeping · scalar gather · MLA · MTP · other; disclose overlap and profiler distortion | — | deliver the table |
| 2:20 | **E5** launch `glm53-big-e5-mtp2` = exact 512K profile with `num_speculative_tokens: 2`; run `dflash2_acceptance_probe.py` (4 prompts, 512 tok each, T=0); compare to the patched-tree E1 K=1 receipt | **2** | reuses existing probe verbatim |
| 2:45 | If E5 gate 1 passes: matched C1/C4/C8 bench (prose ×3, code ×1) on the same container | — | only if gate 1 passed |
| 3:10 | If E5 gate 2 passes: frozen 100-task primary quality fixture ×2 | — | historical K=1 lane completed this in under 8 min; budget 30 min |
| 3:40 | Stop candidate; **restart preserved lane**; `/v1/models` + `WINDOW_RESTORE_OK` completion | **3 (restore)** | |
| 3:50 | Copy receipts to `results/2026-09-0X-window-e0-e1-e5/`, SHA256SUMS, README with verdicts | — | |
| 4:00 | Report to James. Then, and only on his separate instruction, DSF restore on `:30003` | — | |

## 3. Pre-window prep (offline, no Station impact — can be done tonight)

1. **DONE — stats denominator fix:** `patches/slot_cache_hook.py` now increments a device-side routed-use counter by actual `N` and reports `1 - misses/routes`; no M/top-k constant enters hit rate. M=1/M=2 and reset-window tests pass. It ships only with a candidate relaunch.
2. **DONE — nsys wrapper/control:** `NSYS=1` invokes Nsight inside the container, with the complete 2025.6.3 install tree mounted at its original path and a required symlink entry point. A file-controlled engine-core thread calls `cudaProfilerStart/Stop`; the controller is run-id bound, atomic, and tested. Direct target-binary invocation was tested and rejected; the corrected symlink layout passed a network-disabled disposable-container smoke on .9 without changing the live service.
3. **DONE — bucket script:** `scripts/nsys_bucket.py` parses `cuda_gpu_kern_sum` CSV and maps seven GPU-work buckets. It reports overlap and wall residual explicitly instead of fabricating a serial sum. CLI and unit tests pass.
4. **DONE — E5 launch parity:** a fake-Docker contract test reproduces the live 512K K=1 args and proves K=2 changes only `num_speculative_tokens: 1 → 2`.
5. **DONE:** froze `CONTRACT.md`, exact commands/receipt names, package verifier, preparation
   manifest, and full repository checks for publication. Nothing is staged on .9 before those
   checks pass.

## 4. Expected outcomes, honestly

- **E0:** most likely confirms ~0.70 live. Value: retires the 0.6166 figure publicly. The unchanged live hook logs only aggregate/best/worst/median, so this window promises those values—not a fabricated full per-layer table. Layer 4's repeated worst-window signal remains worth reporting.
- **E1:** the decisive one. ~5–7 ms/step is unaccounted for. My prior: ~50% it's attention at deep context + MTP verify (nothing to do), ~35% it's launch/bookkeeping overhead worth a focused E4, ~15% something surprising.
- **E5:** coin flip. K=1 acceptance is structurally ≤2.0; K=2 adds a second draft the MTP head wasn't trained to chain. If accepted length gains <0.15 it cannot pay for the extra verify tokens.

If E1 says <2 ms recoverable and E5 fails, the recipe is done at ~55 tok/s C1 on graphs and we say so in one teaching post. That is a legitimate finish, not a failure.

## 5. Not in this window

- Any E2/E3 deployment (both STOPPED with evidence).
- Any DMA variant (no prediction → nothing to overlap).
- Context/slot curve E6 (decision aid only; separate half-window if James wants it).
- DSF restore (separate instruction).

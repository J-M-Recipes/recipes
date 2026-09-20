# T3b — nightly `dee37d89` + v15 hook — OPEN: +7–14% C1 on live-autotune boots, 0% on cache-loaded boots

Worker: Milo (Hermes `milo` profile, anthropic/claude-fable-5.1 via Nous). Daytime 07:43–13:15 CDT 2026-09-19, after the overnight runner wrongly skipped T3b (class-check `docker run` without `-i`; redone with `-i`: `TrtLlmMxfp4ExpertsModular._invoke_kernel` signature byte-identical 0909 vs nightly). Runners: `../t3b-runner-2026-09-19.sh` (died at hook guard), `../t3b-off54-runner-2026-09-19.sh` (T3b → v18ctl5 → T3b2), `../t3b3-runner-2026-09-19.sh`, `../t3b4-runner-2026-09-19.sh`. Receipts: `receipts/` (window logs, knee/knee6/agentfix/replayc json, metrics, parity captures under `parity/`, both autotune cache files).

Launch = `launch-t3-nightly.sh HOOK=1 OFFGB=54` (v18 flags on the nightly image, hook mounted, `PIN_MODE=split`, `rowmap-static-v1.json`, `--cpu-offload-gb 54`). Same rowmap and same post-rehome residency as v18 on every boot: `HBM_expert=206.61GiB pinned_expert=62.33GiB`, KV **4,530,275 tok (4.32× at 1M)** vs v18's 2.50M — the nightly's KV accounting is different from 0909 (hook-off T3 saw the same jump); not investigated.

## 1. Boot fail at 07:48 — the hook's host guard, not the nightly
First attempt (off60, as v18) died at rehome layer 9: `PIN_HOT abort layer 9 after: MemAvailable 9.68 GiB < 10 GiB`. Traced layer-for-layer against the 0909 T4 boot: **identical to 0.1 GiB** the whole way (layer 8: 11.36 vs 11.29 GiB; layer 39: 23.09 vs 23.24). 0909 then landed layer 9 at 12.02 GiB; the nightly at 9.68. The `MIN_HOST_AVAIL = 10 GiB` guard has been clearing by ~1–2 GiB on every v15/v18 boot. (An earlier read of "45 GiB less headroom on the nightly" compared different layers and was wrong.)

Fix: `--cpu-offload-gb 54` → 9 positional UVA layers instead of 10 (`rehome order … hbm_layers=31 uva_layers=9`), ~6 GiB more host at the tail (18.8–19.5 GiB on three boots), identical post-rehome residency. Offload GiB does not enter the kernel-shape hash. `launch-t3-nightly.sh` now takes `OFFGB` (default 60; `.bak-pre-offgb` kept).

## 2. Windows (same instrument as overnight; C1 = knee r1/r2/knee6; C8, C16 = r1/r2)

| boot | autotune path | C1 | C4 | C8 | C12 | C16 | replay n=4 | Δ C1 vs v18ctl5 |
|---|---|---|---|---|---|---|---|---|
| **T3b** 09:04 fresh `docker run`, no cache for this hash | **live tune, 17 min** (`ed692e15…` created, Saved 189) | **183.6 / 184.3 / 184.4** | 454 | 701 / 708 | 857 | 1014 / 1029 | 355 / 490 | **+6.9%** |
| v18ctl5 09:12 | — | 172.0 / 172.2 / 172.2 | 413 | 642 / 659 | 803 | 746 / 942 | 321 / 416 | — |
| T3b2 09:20 `docker start` of T3b's container | Loaded 189 from `ed692e15…` | 172.1 / 172.4 / 172.4 | **346** | 678 / 676 | **672** | 851 / 988 | 285 / 389 | +0.1% |
| T3b3 10:05 fresh `docker run`, cache present | Loaded 189 | 172.3 / 172.4 / 172.5 | **350** | 677 / 680 | **669** | 974 / 986 | 277 / 392 | +0.1% |
| **T3b4** 12:43 fresh `docker run`, `ed692e15…` moved aside | **live tune, 15 min** (new save, kept as `.from-T3b4`) | **195.8 / 195.6 / 195.4** | 442 | 700 / 733 | 891 | 1011 / 1031 | 374 / 468 | **+13.7%** |

Two-window pairs vs v18ctl5: T3b C4 +10% / C8 +8.4% / C16 +21% (paired r1, r2) / replay +15%. T3b4 C4 +7% / C8 +10% / C16 +21% / replay +14%. Cache-loaded boots: C4 **−15%**, C12 −17%, replay −9%, C8 +4%, C1 0%.

Fixture (tok/s · accept%): v18ctl5 prose 203·34.9 / structured 233·52.5 / code 232·66.1 / shell 265·90.8 / tool_json 229·87.8. T3b 199·34.3 / 246·52.9 / 235·63.0 / 266·82.9 / 213·79.7. T3b4 187·28.8 / 223·44.2 / 250·68.8 / 271·82.9 / 222·81.0. T3b2 and T3b3 have exactly T3b's acceptance rates and ~5% lower tok/s. **The nightly accepts fewer drafts on shell/tool_json (83/80% vs 91/88%)** on every boot; acceptance differs again between the two live tunes (prose 34.3 vs 28.8).

## 3. Greedy parity (`e2c_parity.py`, 18 prompts: 2 tool, 16 text; seed 42, 256 tok)

| pair | exact | text-prompt mismatches (of 16) |
|---|---|---|
| T3b vs v18ctl5 | 2/18 | 15 |
| T3b4 vs v18ctl5 | 2/18 | 15 |
| **T3b vs T3b3** (live tune vs its own saved cache) | 16/18 | **0** — only the 2 tool prompts, which jitter on the lane itself |
| T3b vs T3b4 (two live tunes) | 3/18 | 13 |

Reading: (a) the nightly image is not token-identical to 0909 — merged kernels, different numerics; the plan's 18/18 bar was written for hook-vs-no-hook on one image and cannot be met across images; (b) two live autotunes on `0.6.18.post1` select different kernel configs (**27/42 MoE and 58/147 GEMM entries differ** between `receipts/autotune_configs-slow-ed692e.json` and `-fast-T3b4.json`) and those selections change greedy tokens; (c) loading T3b's saved configs (T3b3) reproduces T3b's tokens exactly — **but not its speed**. So config *selection* alone does not explain the 184-vs-172 gap; there is a performance tax specific to the cache-load path (a lookup miss falling back to a default tactic that happens to be numerically identical, or some state only a live tune leaves behind). Not resolved.

## 4. What is established, what is not
Established (six live-vs-loaded windows, no overlap): the nightly+hook lane is **+7% to +14% C1, +8–10% C8, +14% replay over v18 when FlashInfer tunes live in-process**, and **exactly v18 (with a C4/C12 hole) when it loads a saved cache**. Bit-exactness with 0909 is not available on the nightly by construction.
Not established: why the loaded cache is slow; whether the `.from-T3b4` set (196) reproduces when loaded; whether a live tune is repeatable enough to serve (two samples: 184, 196).

## 5. Next (one boot each, ~25 min, no promotion without a same-window pair)
1. **Boot with `ed692e15….from-T3b4` in place** (swap dir names, fresh `docker run`). 196 → the original save was a bad set and pinning a good cache is the recipe; 172 → the load path itself is the tax and the recipe is "serve from a live tune" (17-min boot, non-deterministic numerics → parity gate per boot).
2. If (1) gives 172: dump the tactic actually chosen at runtime for the M=1/topk=6 MoE key on a loaded boot vs a live boot (FlashInfer `autotuner.py` debug log) and file upstream with both json files.
3. Only then: `enable_adaptive_verification` on the nightly (the plan's stated first follow-up if T3b passed) — its 6.5 GiB graph tax may fit now that KV is 4.5M.

VERDICT T3b: OPEN — live-autotune boots 183.9 / 195.7 C1 vs v18 172.1 (+6.9% / +13.7%, bar +5%) · cache-loaded boots 172.2 / 172.4 (0%, C4 −15%) · parity vs 0909 2/18 (image numerics; text 0/16 between a live tune and its own loaded cache) · rehome 206.61/62.33 identical to v18, host tail 19 GiB at off54 · v18 restored 13:15 · stopped-and-kept `…-hook-EXP-BOOT-FAIL`, `…-hook-off54-EXP`, `T3b3-…-FRESHCACHE-EXP`, `T3b4-…-NOCACHE-EXP` · both cache dirs kept.

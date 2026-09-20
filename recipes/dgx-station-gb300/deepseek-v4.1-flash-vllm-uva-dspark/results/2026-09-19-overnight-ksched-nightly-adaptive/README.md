# Overnight campaign 2026-09-18 → 19 + T3b daytime follow-up — DSV4.1-Flash / GB300 `:30006`

Plan: `../../OVERNIGHT-PLAN-2026-09-18.md`. Reference throughout: `dsv41-vllm-v18-cgsizes-BOUND-REF` (v15 hook + seqs 24 + `[[1,4,5],[5,24,1]]` + lpt 6144 + token capture sizes, image `deepseekv41-flash-0909`). v18 is live on `:30006` at the end; nothing promoted.

Overnight runner: Miloh (default profile), `overnight-runner-2026-09-18.sh`, detached nohup on the box, 21:53–07:04 CDT. Daytime T3b series: Milo (this profile, anthropic/claude-fable-5.1 via Nous), `t3b*-runner-2026-09-19.sh`, 07:43–13:15 CDT. All windows = `run_window_T1T2.sh` (knee.sh ×2, knee6.sh, agent_fixture_o.sh, replay_c.py n=4 ×2). Full campaign log: `overnight-campaign-2026-09-18.log`. Receipts per test under `T*/receipts/`. Parsed window table: `windows.json`.

## One table (tok/s; C1 = knee r1/r2/knee6, C8 and C16 = r1/r2, replay = r1/r2 agg)

| tag | what | C1 | C4 | C8 | C12 | C16 | replay n=4 |
|---|---|---|---|---|---|---|---|
| v18ctl | control | 171.7/172.0/172.1 | 413 | 657/658 | 804 | 752/944 | 332/454 |
| T1 | ksched `[5,12,3],[13,24,1]` | 170.6/171.2/171.1 | 412 | 616/619 | 770 | 746/952 | 334/427 |
| T2 | ksched `[9,24,3]` | 170.5/171.2/171.1 | 411 | 574/550 | 734 | 699/875 | 321/435 |
| v18ctl2 | control | 171.8/172.3/172.3 | 414 | 658/655 | 803 | 747/945 | 320/475 |
| T1b | T1 second window | 170.3/170.7/170.6 | 410 | 615/614 | 763 | 747/945 | 324/436 |
| T2b | T2 second window | 171.7/172.0/172.0 | 416 | 581/584 | 748 | 715/894 | 325/457 |
| T3 | nightly `dee37d89`, **hook off** | 97.1/97.1/97.1 | 179 | 312/313 | 330 | 352/426 | 166/230 |
| v18ctl3 | control | 170.8/171.3/171.3 | 412 | 655/654 | 812 | 751/947 | 341/438 |
| T4 | E4b adaptive d1 unfrozen | 163.4/166.8/168.5 | 408 | 637/634 | 804 | 936/944 | 430/440 |
| v18ctl4 | control | 172.5/172.7/172.8 | 417 | 661/666 | 809 | 756/954 | 335/441 |
| T4b | T4 second window (restart) | 168.5/167.9/167.6 | 405 | 644/645 | 800 | 920/939 | 417/436 |
| **T3b** | nightly + hook, off54, **fresh boot, live autotune** | **183.6/184.3/184.4** | 454 | 701/708 | 857 | 1014/1029 | 355/490 |
| v18ctl5 | control | 172.0/172.2/172.2 | 413 | 642/659 | 803 | 746/942 | 321/416 |
| T3b2 | same container, `docker start` (cache loaded) | 172.1/172.4/172.4 | 346 | 678/676 | 672 | 851/988 | 285/389 |
| T3b3 | fresh `docker run`, cache present (loaded) | 172.3/172.4/172.5 | 350 | 677/680 | 669 | 974/986 | 277/392 |
| **T3b4** | fresh `docker run`, cache moved aside (live autotune) | **195.8/195.6/195.4** | 442 | 700/733 | 891 | 1011/1031 | 374/468 |

C16 r1 is ~25% below r2 on every candidate and control (warm-cache effect inside `knee.sh`); quote C16 only paired r1-vs-r1 / r2-vs-r2.

## Verdicts

- **T1 FAIL** — C8 −6%, C16 0%, replay −4% (bars +5% / +8%). `T1T2/README.md`.
- **T2 FAIL** — C8 −12%, C16 −5%, replay −4%. k=3 band is a verify tax on this lane; lever closed. `T1T2/README.md`.
- **T3 FAIL (hook off)** — C1 97 (+9.5% over v14's 88.7 positional), C8/C16 less than half of v18. Nightly binds clean (`v0.29.1rc1.dev347`, FlashInfer `0.6.18.post1`, 81-min retune, KV 4.4M hook-off). `T3/README.md`.
- **T4 FAIL** — C1 −3%, C8 −3.4% (bar −1.5%); 12 swaps total, all in the first drain, none during held-out; held-out instrument never ran (runner called `e2c_heldout.py` without its tag argv). Adaptive-on-this-hook closed. `T4/README.md`.
- **T3b OPEN — real upside, not a recipe yet.** Live-autotuned boots beat v18 by **+7% and +14% C1** (three C1 runs within 0.8 in each); any boot that *loads* the FlashInfer cache lands at exactly v18's 172 with holes at C4/C12. Autotune on `0.6.18.post1` is non-deterministic (86/189 configs differ between two live tunes) and the saved caches reproduce a slow set. Greedy parity vs 0909 is 2–4/18 (image numerics, not the hook). `T3b/README.md`.

## Bugs found in the overnight runner (Miloh's), fixed or worked around in the daytime series
1. T3b class-check ran `docker run … python3 - <<EOF` without `-i` → empty stdin → `NEW=""` → "hook target changed" → T3b wrongly skipped. Redone with `-i`: `TrtLlmMxfp4ExpertsModular._invoke_kernel` signature is byte-identical 0909 vs nightly.
2. `e2c_heldout.py` called without its `<tag>` argv → `IndexError` ×5 → T4 held-out never measured.
3. Runner's `pgrep -f runner` matched the checking shell itself → "runner ALIVE" after it had exited (cosmetic).
4. (Hook, not runner) `MIN_HOST_AVAIL = 10 GiB` guard has ~1–2 GiB margin at rehome layer 9 on **every** v15/v18 boot (0909: 11.36 → 12.02 GiB). The nightly landed at 9.68 and died. Worked around with `--cpu-offload-gb 54` (9 UVA layers instead of 10; identical post-rehome residency 206.61/62.33; tail 18.8–19.5 GiB). Launcher `launch-t3-nightly.sh` now takes `OFFGB` (default 60).

## Box state at end (13:15 CDT 2026-09-19)
`:30006` = `dsv41-vllm-v18-cgsizes-BOUND-REF`, serving `dsv41-flash-uva`. Stopped-and-kept: `T1-ksched-5-3-1-EXP`, `T2-ksched-5-3-EXP`, `T3-nightly-dee37d89-nohook-EXP`, `T3b-nightly-dee37d89-hook-EXP-BOOT-FAIL`, `T3b-…-hook-off54-EXP`, `T3b3-…-FRESHCACHE-EXP`, `T3b4-…-NOCACHE-EXP`, `T4-adaptive-d1-unfrozen-C8gate-EXP`. Nothing removed. Autotune cache dirs under `vllm-cache/flashinfer_autotune_cache/0.6.18.post1/103a/`: `bba7410c…` (T3 hook-off), `ed692e15…` (hook shapes; the slow set), `ed692e15….from-T3b4` (the 196 tok/s set — untested as a loaded cache).

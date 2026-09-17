# E2b — static pin boot (F4 split)

Worker: **grok-4.6 / xai-oauth**. Same-window pair vs v14 REF. GLM big stayed stopped-and-kept. `:30003` untouched. Candidate left **up** on `:30006`.

CAT_ORDER (both boots): `prose structured code shell_ops tool_json`.

## Dry-run (off-lane, before any boot)

Container `pin-e2b-dry` (`--rm`). GPU quiet. Cap 0.25. Real geometry E=384, rowmap layer-0 295/89.

Triton fmaf finalize vs kernel finalize: 100% bit-id at T=1/6/24/96. F4 split (hot HBM + cold UVA) vs full-E: 100% at all T. CUDA-graph T=6: 100%. `results/e2b/dryrun.json`.

## Hook

- `PIN_MODE=split` (count/off kept). Count is **not** on the hot path.
- Hook point: `GPUModelRunner.load_model` end (class lives in `vllm.v1.worker.gpu.model_runner`, not `gpu_model_runner`). Also wraps `profile_run`. Scanner finds the class in `sys.modules`.
- Re-home: all 30 HBM layers first, then 10 UVA (interleave overflows: UVA needs +5.2 GiB HBM, HBM only frees 1.6). Bind via hot `w1` data_ptr — Modular experts are not `nn.Module`s.
- Row map: `int32[384]`, ≥0 hot row, `-(cold+1)` cold. F4: two `do_finalize=False` calls + Triton `tl.math.fma` finalize in k order.
- Failed boots kept (not rm'd): `…-NOREHOME-FAILED`, `…-NOMODULAR-FAILED`, `…-HBMBUDGET-FAILED`.

## Boots

| | container | bind | autotune |
|---|---|---|---|
| control | `dsv41-vllm-v14-1M-ksched-agent-BOUND-REF` (`docker start`) | 244 s | 9ac7b387 / 210 |
| candidate first | `dsv41-vllm-E2B-pin-EXP` | ~25 min (rehome + 21 new configs) | 9ac7b387 Loaded 210 → Saved 231 (21 new) |
| candidate restart | same | 258 s | Loaded 231 / 0 new |

Rehome: HBM_expert **206.61 GiB**, pinned_expert **62.33 GiB** (expect 206.6 / 62.4). KV **4.75 GiB** (v14 4.89, within 0.5).

## Parity (greedy T=0, agent-fixture prompts, C1)

Control vs itself: 8/8 content+tool-args exact (tool_call **ids** differ, also on v14 vs itself). Cand vs ctrl: **8/8** same. Gate PASS.

## Knee (verdict instrument)

| | C1 | C8 | C16 |
|---|--:|--:|--:|
| control | 89.3 | 307.7 | 414.3 |
| candidate | 153.1 | 701.4 | 809.0 |

C1 **+71.4%** (bar +15%). C16 cand runs 666.8/951.2 — spread is real; both still ≫ control 413/415.

## Other instruments

| | control | cand |
|---|--:|--:|
| agent prose tok/s | 98.3 | 169.0 |
| structured | 123.0 | 207.6 |
| code | 153.9 | 213.6 |
| shell_ops | 160.2 | 250.3 |
| tool_json | 180.4 | 182.1 |
| weighted accept | 59.9% | 59.6% |
| T5 prose fixture | 91.4 | 152.5 |
| replay n=4 r1/r2 | 170.6 / 222.7 | 247.5 / 431.3 |

Accept rates bit-identical on prose/structured/code/shell (29.8/49.5/63.8/90.8). tool_json 89.7→87.8. No category −3%. Replay swing is the known instrument; direction matches the knee.

## Left running

`dsv41-vllm-E2B-pin-EXP` on `:30006` (served `dsv41-flash-uva`, 1M). v14 REF stopped-and-kept.

## E2b VERDICT: parity 8/8 prompts exact · C1 control 89.3 → cand 153.1 (+71.4%) · C16 414.3 → 809.0 · replay_c 196.6 → 339.4 · KV 4.75 GiB · host pinned 62.33 GiB · autotune 12 min (21 new)

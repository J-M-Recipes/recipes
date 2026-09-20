# T3 — nightly image `vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910`, v18 flags, hook OFF — FAIL (as a lane), PASS (as a compat check)

Runner: Miloh's overnight runner. Launched 04:57 CDT, bound 06:18 CDT (81-min wall: FlashInfer `0.6.18.post1` rejected the seeded 0909 cache → full retune, new hash `bba7410c…`). Window T3 06:18–06:24, control v18ctl3 after. Receipts: `receipts/boot-T3.txt`, `smoke-T3.txt`, `coldprefill-T3.txt`, `class-check.txt` (the overnight one — see bug), `window-T3.log`, `knee-T3-*.json`.

## Boot facts
- Image binds on driver 595.84 / CUDA 13.0 (`_metadata` in the autotune file: flashinfer 0.6.18.post1, cuda 13.0, cublas 13.1.1, cudnn 92000). vLLM `v0.29.1rc1.dev347`. No driver mismatch; the cu13.0.1 fallback was not needed.
- Engram host placement resolved as `EngramConfig(cpu_offload=True, embedding_across_dp=False, dp_shared_memory=False)` — two new fields, our `--engram-config` still valid.
- Graphs 1.21 + 1.03 GiB. **KV 4,405,597 tok (4.20× at 1M)** vs 2.5M on 0909 — hook off, so 62 GiB of pinned experts are not competing; not comparable to v18's KV.
- Smoke 5/5 (arith, count, prose, tool call parsed, think parsed). Idle cold prefill 8k 15.8k / 32k 14.0k / 128k 21.0k tok/s (v18: 15.7k / 21.3k / 22.7k).

## Knee (hook off, positional off60) vs v18ctl3 and vs v14 positional baseline

| | C1 | C4 | C8 | C16 | replay |
|---|---|---|---|---|---|
| T3 nightly hook-off | **97.1 / 97.1 / 97.1** | 179 | 312 / 313 | 352 / 426 | 166 / 230 |
| v14 (0909, positional, 2026-09-15) | 88.7 | — | 305 | 386–417 | — |
| v18ctl3 (0909 + v15 hook) | 170.8 / 171.3 | 412 | 655 / 654 | 751 / 947 | 341 / 438 |

Bar was C1 ≥ 93.1 (v14 × 1.05) + smoke 5/5 → technically **PASS (+9.5% over v14)**: the nightly's merged DSV4.1 kernels are worth ~+9% at C1 with everything positional. But as a lane it is less than half of v18 at C8/C16 — the hook is the bigger lever, and T3 hook-off cannot replace it. This is the number T3b was designed to build on.

## Bug: T3b was skipped for a wrong reason
The runner's class-signature check ran `docker run … python3 - <<EOF` **without `-i`**, so the nightly python got an empty stdin, printed nothing, `NEW=""`, and the runner concluded "hook target changed → skip T3b". Redone at 07:30 CDT with `-i`: `TrtLlmMxfp4ExpertsModular._invoke_kernel` signature is byte-identical on 0909 and the nightly (`class-check.txt` has the 0909 half; the nightly half is in the T3b README). T3b then ran in daytime — see `../T3b/README.md`.

VERDICT T3: hook-off nightly binds clean · C1 97.1 (+9.5% vs v14 88.7, bar 93.1 PASS) · C8 312 / C16 352–426 (≈½ of v18) · KV 4.41M hook-off · 81-min retune (`bba7410c…`) · tools 5/5 · stopped-and-kept `dsv41-vllm-T3-nightly-dee37d89-nohook-EXP`. Not a lane candidate by itself.

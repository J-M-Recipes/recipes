# 2026-10-18 — Card M3G: el8 "M3" MegaMoE hook, Grace-only (no sidecar), same-window vs v22; plus V22S32 (32 seats) and its confirm

Question (James, 2026-10-17 22:38 CDT): improve the one-Station recipe *without* the RTX PRO 6000 sidecar, tonight. Three Opus 5.5
research briefs preceded the card (el8/Ronan Grace-only mechanism; upstream vLLM state; 0xSero recipes + SGLang) — see the blog/skill;
not vendored here.

## Arms (one window, 1017 weights `1bae0e10`, lane `:30006`, stop-and-keep)

| arm | stack |
|---|---|
| CTL1 / CTL2 | v22 reference: `deepseekv41-flash-0909` + pin-hot hook v21 (rowmap-static-v1 295/89), off60, util 0.97, 1M, seqs 24, `[[1,8,5],[9,24,1]]` |
| M3G-mix | `vllm/vllm-openai@sha256:3af6a45b…` (= nightly-af7f9488; tag rotated off Hub) + original-el8 M3 hook @87af662, `MEGA_PEER=0 MEGA_COLD_TRT=1`, `rowmap-mix-v1` 285/99, vllm#58132 overlay, `deep_gemm_mega_moe --enable-expert-parallel`, seqs 32, util 0.97, 1M, k `[[1,4,5],[5,8,3],[9,32,3]]` probabilistic, `MEGA_LL_GEMM=1` |
| M3G-ours | same with `rowmap-static-v1` 295/89 |
| V22S32 | v22 launch with `--max-num-seqs 32` and `[[1,8,5],[9,32,1]]` (new autotune hash) |

Instruments: RonanLabs `kbench_chat_ds.py` real text (CNN/DM ~8.2K in, 1,024 forced, thinking off, T=0, nonce per run, prefix hit 0.00% every run;
C1 = mean of 3 pools, C4 head-8, C16/C32 32 prompts; metric decode_agg = C×1000/mean TPOT), our `run_window_T1T2.sh` (knee r1/r2, knee6,
agent fixture, replay n=4 ×2), tool gauntlet 12 tasks. `throughput.csv` collects every harness line from both cards.

## Results (controls = CTL1/CTL2 mean)

| | CTL | M3G-mix | M3G-ours | V22S32 (first boot) | V22S32 (confirm, own controls) |
|---|--:|--:|--:|--:|--:|
| RT C1 | 188.5 | 147.2 | 176.4 (−6%) | 194.3 | 185.8 vs 190.4 (−2.4%) |
| RT C4 | 333 | 268 | 341 (+2%) | 346 | 321.7 vs 331.8 (−3.0%) |
| RT C16 | 538 | 459 | **654 (+22%)** | 550 | 542 vs 539 |
| RT C32 output tok/s | 511 (24 seqs, 8 queued) | 535 | **739** | 654 | 640 vs 516 (+24%) |
| TTFT 8K C1 | 0.65 s | 0.33 s | **0.29 s** | 0.64 s | 0.64 s |
| knee C1 / C8 / C16 | 192 / 619 / 938 | 164 / 512 / 691 | 180 / 672 / 934 | 211 / 633 / 995 | 200 / — / — |
| replay n=4 r2 | 415 | 329 | 405 | 446 | 418 vs 425 |
| fixture prose/code/shell/tool_json | 180/249/259/237 | 159/278/230/195 | 164/240/256/207 | 186/234/271/211 | — |
| KV at 1M | 2.50M | 6.05M | 2.49M | — | — |
| gauntlet | 12/12 | 12/12 | 12/12 | 12/12 | — |
| boot | — | 1,977 s (first, JIT) | 674 s | 1,410 s (live tune) | 317 s (cached) |

## Verdicts (bars pre-registered in `scripts/cardM3G_runner.sh` header)

- **M3G-ours: not promoted.** Fails C1 (−6% vs −3% bar); passes C16 (+22%), TTFT (0.29 s), gauntlet, replay within noise. A prefill/TTFT and C16+ lever,
  not a C1 lever. Kept stopped as the RTX PRO 6000 sidecar base (same image/hook; `MEGA_PEER=2` + el8 `peer_server2.py` + b12x `2cc7f66a`).
- **M3G-mix: fails every decode bar** (−13…−28%); only TTFT. **The map, not the kernel, is the gap:** 285→295 hot experts per layer on the same hook
  = +20% C1, +42% C16. el8 ships our `rowmap-static-v1` as his hook default.
- **V22S32: passes its bar at the edge, not promoted.** The first boot's knee C1 +10% did not survive a bracketed confirm (−2.4% C1, −3.0% C4, +24% C32 output,
  worse C16 TTFT). Lane traffic is C1–C8; the seat count moves with the sidecar.

## Mechanism notes

- el8's hook constructs MegaMoE with `num_experts=n_hot`, so DeepGEMM only holds the hot rows (that is how it avoids the all-384 HBM copy stock
  `deep_gemm_mega_moe` makes); cold rows are converted to TRT-LLM layout, pinned, and read via `get_accelerator_view_from_cpu_tensor` — this
  recipe's cold path. Host: ~264 GiB pinned Engram + ~62 GiB cold copy ≈ 326 of 494 GiB.
- vllm#58132 (decoder SWA bounded replay) is **not** in af7f9488 or v0.31.0; it is an overlay here and upstream in the v0.32.0 tag / nightly-39983a5e.
- Acceptance at C16 was 2.66 on M3G (k=3 band, probabilistic) vs 1.80 on v22 (k=1 band) and M3G-mix was still slower: step cost dominates.
- Known runner bugs: the RT summary line reads `decode_agg` from the per-run file (absent there) — the numbers are in the stdout JSON lines; the gauntlet
  completes in ~15 s/arm (verify cache-salt semantics before relying on it as a gate).

Scripts in `scripts/` have box paths scrubbed (`$HOME`, `<station-ip>`). Raw receipts under `receipts/` and `cardS32/receipts/`.

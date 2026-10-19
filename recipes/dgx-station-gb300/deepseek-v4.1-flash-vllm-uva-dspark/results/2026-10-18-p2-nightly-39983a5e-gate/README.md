# 2026-10-18 — vLLM nightly-39983a5e + pin-hot hook, "config C": **PASSED GATES — NOT PROMOTED (pending James)**

Candidate, not the lane. The lane stays v23 on the 0909 image ([`../2026-10-18-v23-pinexact-host/`](../2026-10-18-v23-pinexact-host/README.md)).

**Config C** = the v23 launch translated to vLLM nightly `39983a5e` (`0.32.0rc1.dev329+g39983a5e7`, FlashInfer 0.7.2),
pulled by digest `vllm/vllm-openai@sha256:a3bc7f09c1e2e72e068021feb75118c92e98c538706caa0c5d1acc0bb9252623` (the tag
rotates), same 1017 weights (`1bae0e1088993b9b82de08f59de5be5a34e42170`), same hook (sha256 `b5e4f4f8…`), same rowmap,
`--cpu-offload-gb 60`, `PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024`, plus:

- `--kv-cache-dtype fp8_ds_mla` (required: MegaAttn otherwise resolves to `nvfp4_ds_mla`, a second axis)
- `--attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41"}'` (the SM100 default on this image; the drafter inherits it)
- `--no-swa-bounded-replay` (turns off vllm#58132 decoder SWA bounded replay)
- `--tokenizer-mode deepseek_v41`, the 15 capture sizes as `--compilation-config` JSON
- FlashInfer managed autotune store `-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm/fi-autotune-v2` on a fresh cache mount `vllm-cache-p2-39983a5e` (210 entries after the first boot; later boots `Preloaded 210/210`, no live tune)

Exact launcher: [`scripts/launch_p2b_noreplay.sh`](scripts/launch_p2b_noreplay.sh) (= `launch_p2_nightly39983a5e.sh` + `--no-swa-bounded-replay`).
Kept stopped on the box as `dsv41-vllm-1017-P2b-n39983a5e-noreplay-EXP-FAIL-P2f-PASS`.

## Promotion gate — Card P2f (19:09–20:18 CDT), A = v23 live → C = config C → D = v23 restart

Bars fixed in the `cardP2f_runner.sh` header before launch; verdict script run unmodified.

| gate | bar | result | verdict |
|---|---|---|---|
| per-class decode wall-clock, 5 classes × {C1, C8} × 2 controls | candidate mean ≥ control mean × (1 − max(2%, 1.5 × control same-config spread)) in every cell | 20/20 cells; margins +0.5% … +12.6% | **PASS** |
| GSM8K-200 (lil-bench v0.7.7 `c71ec1f2`, T=0, C30) | ≥ A − 1.5 pt | 95.5 vs 97.0 (−1.5 pt, 3 items, McNemar p = 0.25) | **PASS — exactly at the bar** |
| BFCL dev (simple_python 400 + multiple 200) | ≥ A − 1 pt | 92.50 vs 92.33 | PASS |
| BFCL held-out (live_simple 258 + live_multiple 1,053) | ≥ A − 1 pt | 77.65 vs 77.65 (1018/1311 both) | PASS |
| gauntlet | 12/12 | 12/12 | PASS |
| RT C1 (kbench real text, mean of 3) | ≥ CTL − 2% | 197.8 vs 186.4 (+6.1%) | PASS |
| knee6 C8 (4 reps per arm) | ≥ CTL − 2% | 656.5 vs 605.2 (+8.5%) | PASS |
| validity | rehome 40/40, no replay boot line, MegaAttn line, prompts sha1 `618898aa` on all arms | all present | OK |

CTL = mean(A, D). Also measured on this image (Card P2, replay on; teacher-forced capture is cold prefill):
margin-weighted top-1 flips W = 3.110 per 1k positions vs the v23 capture, bar ≤ 3.85 (1.25 × calibration 3.078),
0 confident agent/tool flips — PASS. Not re-run on config C.

### GSM8K caveat (read before promoting)

- The −1.5 pt sits exactly on the pre-registered line; the 3 flips all go one way (correct only on A).
- **Config C writes +48% completion tokens on GSM8K** (mean 293.5 vs 198.3; p50 177 vs 158; p90 483 vs 330). D vs A,
  same image two boots, is +6% and −0.5 pt. Two of the three flipped items are long C outputs (3,229 vs 407 tokens;
  2,256 vs 327). Length inflation is an early damage signal in lil-bench's own guidance.
- n = 200 cannot separate a 1.5-pt effect from noise. The full 1,319-item GSM8K per arm (~10 min each) would; it has not been run.

### Per-class decode wall-clock (FL 256, 40 prompts per class, mean of r1/r2, tok/s)

| class | cell | config C | v23 18:29 (spread) | bar | Δ | v23post (spread) | bar | Δ |
|---|---|---|---|---|---|---|---|---|
| prose | C1 | 241.6 | 232.3 (0.3%) | 227.7 | +4.0% | 231.4 (0.4%) | 226.7 | +4.4% |
| prose | C8 | 564.8 | 532.0 (2.5%) | 511.9 | +6.2% | 519.3 (1.9%) | 504.8 | +8.8% |
| structured | C1 | 255.5 | 250.3 (0.8%) | 245.3 | +2.1% | 254.3 (1.6%) | 248.3 | +0.5% |
| structured | C8 | 573.8 | 509.4 (2.7%) | 488.6 | +12.6% | 522.2 (0.8%) | 511.8 | +9.9% |
| code | C1 | 312.6 | 293.4 (3.3%) | 278.8 | +6.6% | 296.1 (1.6%) | 289.2 | +5.6% |
| code | C8 | 673.2 | 642.1 (1.2%) | 629.2 | +4.8% | 642.8 (4.6%) | 598.7 | +4.7% |
| shell_ops | C1 | 331.6 | 316.9 (5.0%) | 292.9 | +4.6% | 317.0 (2.9%) | 303.4 | +4.6% |
| shell_ops | C8 | 739.1 | 681.4 (3.3%) | 647.5 | +8.5% | 657.7 (0.7%) | 644.6 | +12.4% |
| tool_json | C1 | 316.9 | 300.0 (1.6%) | 293.0 | +5.6% | 286.2 (0.1%) | 280.5 | +10.7% |
| tool_json | C8 | 767.1 | 684.0 (3.1%) | 652.0 | +12.1% | 721.8 (1.7%) | 703.1 | +6.3% |

Acceptance (% of drafted tokens, `/metrics` deltas, token-weighted), config C vs v23 / v23post:

| class | C1 | C8 | step-speed ratio C1 / C8 (decode ratio ÷ tokens-per-step ratio) |
|---|---|---|---|
| prose | 40.5 vs 41.8 / 41.5 | 43.1 vs 42.2 / 41.0 | 1.061 / 1.048 |
| structured | **42.9 vs 45.5 / 46.5** | 43.2 vs 42.8 / 43.5 | 1.062 / 1.111 |
| code | 66.5 vs 66.6 / 66.9 | 67.7 vs 67.6 / 69.3 | 1.064 / 1.057 |
| shell_ops | 67.1 vs 68.4 / 67.9 | 67.2 vs 66.6 / 66.5 | 1.059 / 1.096 |
| tool_json | 59.6 vs 59.6 / 56.3 | 59.5 vs 57.4 / 59.1 | 1.058 / 1.074 |

- The engine step is ~6% faster at equal acceptance in every class at C1 and 5–11% at C8.
- **Investigation trigger (> 2 pts):** structured C1 −2.6 / −3.6 pts is the one downward move; the cell still decodes
  faster (+2.1% / +0.5%). The explanation under the new rule is owed before promotion; the candidate cause is a drafter-input
  numerics shift upstream of layer 37 (see the bisect), **not verified**. Upward moves: prose C8 +2.1, tool_json C1 +3.4, tool_json C8 +2.1.
- **Window drift:** v23post differs from the 18:29 baseline by more than the spreads on tool_json C1 (−4.6%), tool_json C8
  (+5.5%) and shell_ops C8 (−3.5%). Config C clears both controls, so the verdict does not depend on which one is used.
- The 8-prompt acceptance fixture (`agent_fixture_x3.py`) had flagged code −4.7 pts on every nightly variant (P2b/P2c/P2e).
  The 40-real-prompt instrument does not reproduce it: code acceptance is within 0.4 pt at C1 and 1.6 pt at C8.

### Fidelity

| cell | A v23 live | C config C | D v23 restart |
|---|---|---|---|
| GSM8K-200 | 97.0 (194/200) | 95.5 (191/200) | 96.5 (193/200) |
| GSM8K mean completion tokens | 198.3 | **293.5** | (+6% vs A) |
| BFCL dev | 92.33 (554/600) | 92.50 (555/600) | 92.67 (556/600) |
| BFCL held-out | 77.65 (1018/1311) | 77.65 (1018/1311) | 77.88 (1021/1311) |
| gauntlet | 12/12 | 12/12 | 12/12 |

BFCL = our AST grader against the possible_answer files: a same-grader A/B, not a leaderboard number.

### Speed

| cell | A v23 | C | D v23 | C vs CTL |
|---|---|---|---|---|
| RT C1 decode (3 runs) | 186.1 | 197.8 (192.5/201.3/199.7) | 186.8 | +6.1% |
| RT C1 median TTFT (8.2K prompt) | 0.643 s | **0.430 s** | 0.647 s | −33% |
| RT C1 acceptance_len | 2.910 | 2.940 | 2.942 | — |
| knee6 C1 / C6 (4 reps) | 192.1 / 530.0 | 200.4 / 557.5 | 192.2 / 525.3 | +4.3% / +5.7% |
| knee6 C8 raw reps | 615.7 / 569.2 / 622.8 / 578.2 | 645.9 / 647.3 / 649.5 / 683.4 | 622.7 / 603.8 / 625.9 / 603.7 | +8.5% |
| knee r2 C1 / C8 / C16 | 192.2 / 618.5 / 944.4 | 200.3 / 647.8 / 1,074.8 | 192.4 / 600.6 / 944.5 | +4.2% / +6.3% / +13.8% |
| cold ~53K prefill (N=1) | 24,094 tok/s | 30,518 tok/s | 24,006 tok/s | +26.9% |
| KV tokens at 1M | 2,504,891 | **3,878,024** | 2,504,891 | +55% |

v23's C8 reps are bimodal again; config C's four sit at 646–683. Boot facts (C): bind 335 s, `Preloaded 210/210`,
9-s autotune window, `rehome done layers=40`, host guard at layer 9 196.9 GiB, pinned 252.79 GiB, errcount 0.

## How config C was reached (P2 → P2e, same day)

| card | window (CDT) | arms | outcome |
|---|---|---|---|
| **P2** | 10:42–11:33 | v23 → nightly + hook, SWA replay **on** → v23 | FAIL on two pre-registered bars: warm replay r2 **−10.9%** and 8-prompt-fixture per-class acceptance (±5–7 pts). Every speed bar passed: TTFT 8K 0.283 s, cold 64K ×1.56, RT C1 +6.0%, KV 3,455,569; flips W 3.110 PASS; BFCL dev 92.3 vs 92.4; gauntlet 12/12. First boot = 11.4-min live tune into the new managed store. |
| **P2b** | 12:12–12:45 | v23 → B (P2 container, store-loaded) → C (`--no-swa-bounded-replay`) → v23 | B reproduces the replay loss with no live tune (std r2 −10.3%, forced-length r2 −6.4%): **#58132 bounded replay on prefix hits** makes a warm-hit request cost a cold one (TTFT 0.208 vs 0.212 s; v23 0.10–0.13). C restores warm TTFT and passes forced-length replay (+10.7%) at the price of TTFT 8K 0.28 → 0.43 s and cold 64K 46.3K → 30.4K tok/s; KV rises to 3,878,024. Fixture code acceptance −4.7 on both B and C → FAIL under the then-current 2-pt rule. |
| **P2c** | 13:05–13:45 | v23 → C restart → C1 (#56893 MXFP8 KV record forced off) → C2 (drafter on `FLASHMLA_SPARSE_DSV41`) → v23 | Neither override recovers fixture code acceptance (C1 −4.8, C2 −4.2; C restart reproduces C to the decimal). **Ruled out: the #56893 KV record** (also costs 305K KV tokens) **and the drafter's attention kernel / MegaAttn fused cast.** Override published as a patch: [`patches/p2c-C1-norecord-on-39983a5e.patch`](patches/p2c-C1-norecord-on-39983a5e.patch). |
| **P2e** | 17:04–17:31 | v23 → E (target and drafter on `FLASHMLA_SPARSE_DSV41`) → v23 | Code −3.7, structured −2.5: **target MegaAttn is not the cause** either (at most a partial contributor on one prompt). SPARSE: TTFT 8K 0.46 s, cold 64K 27.5K tok/s, RT C1 +8.1%. Attention backend is not in the FlashInfer hash. |
| AUXCAPTURE (desk) | afternoon | 1017 reference vs 0909 vs nightly source | **Ruled out the aux-capture path:** all three take the plain unweighted mean over the 4 hyper-connection streams entering target layers 37–39 — the same tensor up to fp32 summation order. Nothing to file upstream. |
| ACTQUANT (desk) | afternoon | 1017 reference `act_quant(o)` before `wo_a` | vLLM did MXFP8 `o` before an FP8 `wo_a` GEMM from the start; 1017 brought the BF16 reference in line. No mismatch on either image. |
| PERCLASS | 18:24–18:46 | v23 baseline on the new instrument | 40 real prompts per class, forced length; replaces the 8-prompt fixture as the drafter-side gate (rule below). |
| **P2f** | 19:09–20:18 | the promotion gate above | **all 7 gates PASS**; not promoted |

Remaining unexplained: a drafter-proposal shift at fixed target tokens on the 8-prompt fixture (code prompt 4: 0.943 on
v23 vs 0.800 on every MegaAttn nightly arm, 0.867 on SPARSE). Remaining candidates are shared compiled paths
(FlashInfer 0.7.2, DeepGEMM, TileLang mHC) and indexer BF16/MX logits on SM100 — none revertible from Python.

**SWA bounded replay ON variant — parked.** It is the faster prefill (TTFT 8K 0.28 s; cold ~53K 46.3K vs 24.3K tok/s
store-loaded) but warm-prefix replay r2 is −6 … −11% (P2 std −10.9%, P2b std −10.3% / forced-length −6.4%), and the
lane's traffic is warm-prefix agent turns.

## Rule amendment (accepted by James, 2026-10-18)

- **Per-class DSpark acceptance is an investigation trigger, not a gate.** Any class moving > 2 pts (the 8-prompt fixture
  or this instrument's `/metrics` acceptance) requires a written explanation before promotion; it does not block by itself.
- **The gate for drafter-side changes is per-class decode wall-clock:** `perclass_run.sh`, prompts sha1 `618898aa`,
  FL 256, C1 + C8, r1/r2. For every class × cell, candidate mean ≥ control mean − max(2%, 1.5 × the control's
  same-config spread). The control is a same-window v23 run, bracketed when a boot is involved. All 10 cells must pass.
- **Fidelity gates unchanged:** BFCL (≥ control − 1 pt), GSM8K, margin-weighted teacher-forced flips, gauntlet 12/12.
  Speed bars outside the drafter question (RT harness, replay, knee) unchanged.

## Per-class instrument

`scripts/perclass/perclass_decode.py` (stdlib), `perclass_run.sh` (nat warm pass → r1 C1, C8 → r2 C1, C8 → x3 fixture → table/gate), `perclass_table.py` (`--control` applies the gate).
Box sha1 before the path scrub: `perclass_decode.py 09c66432`, `perclass_run.sh 682114ad`, `perclass_table.py 6e6d3ff8` (unchanged by the scrub).

**The prompt set is private and not published.** `perclass_prompts.jsonl` (sha1 `618898aac432bd3f50eefa7dde2dabb6f8849b31`,
200 prompts, 40 per class) is built from real Hermes transcript turns (8 fixture + up to 32 harvested per class, capped at
8,192 prompt tokens). The published per-run JSONs keep every aggregate and drop the per-request `rows` (they carry
60-character output heads from those prompts); tables are verbatim. Labels describe the context, not the output (v23
emitted tool calls on 30% of prose and 72.5% of shell/tool prompts); forced 256 tokens run past natural EOS on both arms.

## Files

- `scripts/` — launchers `launch_p2_nightly39983a5e.sh` (P2), `launch_p2b_noreplay.sh` (**config C**), `launch_p2c.sh`, `launch_p2e_sparse.sh`; runners + verdicts `cardP2{,b,c,e,f}_{runner.sh,verdict.py}`; `replay_fl.py` (forced-length replay), `agent_fixture_x3.py` (8/class fixture), `flips_margin_p2.py`, `p2b_flagcheck.py`; `perclass/`.
- `receipts/P2`, `P2b`, `P2c`, `P2e`, `P2f` — VERDICT.md, SUMMARY.txt, runner.log, per-arm facts/errcount/hostmem/metrics/cold, rt-*/ kbench JSON, knee/knee6/replay/fixture JSON, flips (P2), BFCL summaries, GSM8K-200 JSON (P2f), gauntlet summaries, autotune store keys.
- `perclass/v23-baseline`, `perclass/candC`, `perclass/v23post` — tables (md/json) and row-stripped run JSONs.
- `throughput.csv` — every kbench cell across P2–P2f and the P2f knee cells.
- Mirrored with a path scrub (box home → `~`/`$HOME`, LAN → host names). Boot/post logs, teacher-forced captures and raw BFCL jsonl stay on the box.

Runs: claude-opus-5-5 workers from Milo's briefs. Write-up: Milo (Hermes, claude-fable-5-1). Rule amendment drafted
after an Opus 5.5 review; vLLM upstream for the nightly, #58132, #56893 and the managed autotune store.

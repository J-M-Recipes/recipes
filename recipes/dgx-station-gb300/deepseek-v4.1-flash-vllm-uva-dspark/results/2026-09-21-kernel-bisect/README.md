# 2026-09-21 — kernel bisect: where the tool-shaped logit drift enters the vLLM nightlies

Hook ON, `--cpu-offload-gb 54`, `--kv-cache-dtype fp8_ds_mla` (the v19 flags) on three nightlies between the 0909 image and the reverted v19 (`dee37d89`). Instrument: teacher-forced Δlogprob vs the v14 no-hook capture (`tf_logprob.py` + `tf_split.py`, cold by construction), `agent_fixture_o` acceptance, `knee.sh` r1. Runner `cardC-kernel-bisect-2026-09-21.sh`; receipts in `receipts/`. r1 and r2 were fresh FlashInfer hashes (live tunes 16 min each); r3 hashed to `ddf01704…` — the v19 pinned set — and loaded it (7 s), i.e. `0bfc7a15` and `dee37d89` share MoE kernel shapes (#56266 Mega-Gate does not change them). v18 restored on `:30006` after the ladder.

## Ladder

| rung | nightly | main date | kernel PRs added vs previous | agent_tool flips | heldout | prose | accept tool_json / shell / prose | C1 r1 |
|---|---|---|---|--:|--:|--:|---|--:|
| v18 | `deepseekv41-flash-0909` | 09-10 | — (day-0 build) | **0.56%** | 1.08% | 1.06% | 0.878 / 0.908 / 0.349 | 172 |
| **r1** | `2671fedf` (v0.29.1rc1.dev9) | 09-13 | #56214 model support on main; **no perf kernels** | **1.77%** | **0.00%** | 1.04% | **0.860 / 0.908 / 0.349** | **189.2** |
| r2 | `af1c0149` | 09-16 | + #56464 DeepSelect top-k, #56962 Mega-mHC, #56512 Engram async prefetch | **11.73%** | 8.72% | 1.82% | 0.757 / 0.760 / **0.260** | 194.3 |
| r3 | `0bfc7a15` | 09-17 | + #56935 FlashMLA mega-attn + NVFP4 KV, #56568/#57204 MegaMoE | 10.99% | 8.89% | 1.82% | 0.838 / 0.829 / 0.336 | 183.3 |
| v19 | `dee37d89` | 09-18 | + #56266 Mega-Gate | 11.45% | 8.89% | 1.80% | 0.838 / 0.829 / 0.336 | 183 |

## Reading

**The drift enters between `2671fedf` and `af1c0149`** — the rung that adds DeepSelect top-k (#56464), Mega-mHC (#56962) and Engram async prefetch (#56512). Agent-doc flips go 1.77% → 11.73%; heldout 0.00% → 8.72%; prose acceptance falls from 0.349 to 0.260 and shell from 0.908 to 0.760. r3 then *recovers* part of the acceptance (Mega-MoE / mega-attn land; shell 0.829, prose 0.336) without moving the flips — so r2 carries two effects: the flip-inducing one that persists to v19, and an extra acceptance loss that r3's kernels cancel. Of the three PRs in r2, **#56464 DeepSelect** (a different top-k selection for the DSA sparse indexer — which latents each query attends over) is the one that changes *what attention reads*, and would move short, high-confidence, structured positions first; #56962 Mega-mHC is a fused hyper-connection kernel; #56512 is a prefetch path. This bundle does not separate them; that needs three more boots (image with one PR reverted each, or a local build), or upstream's own numerics tests.

**r1 is a candidate, not a bisect artefact.** On `2671fedf` the hook keeps v18's acceptance on every class (tool_json 0.860 vs 0.878 is inside the 2-pt bar; shell and prose identical) and reads **189 tok/s C1 (+10% over v18)** with 1.77% agent-doc flips — three times v18's but an order of magnitude under v19's — and 0.00% on held-out math/JP-DE. BFCL and a second knee were run on r1 after the ladder (`docker start`, autotune hit): **BFCL 561/600 = 93.5%** (v18 93.3, v19 93.2; failure kinds value 35 / name 3 / no-call 1 — the same set), C1 **180.2** loaded (189.2 on the live-tune boot; v18 172), C8 685, C16 958. On the promotion rule — BFCL ≥ v18 − 1 pt AND tool_json/shell acceptance within 2 pt — r1 **passes both**, with 1.77% agent-doc flips (v18 0.56%, v19 11.45%). It is the **v20 candidate**: everything the nightly speed bought that was not the drift is already present at `2671fedf`. Not promoted here — promotion needs a same-window v18 control pair, the fund harness at C16/C24, and a pinned autotune set with its sha256, in a scheduled window.

## Not claimed

- Which of the three r2 PRs is responsible. Upstream context: #56625's PR text says `FLASHMLA_MEGA_ATTN_DSV41` (the SM100 default from #56935, i.e. r3 onward) is "currently broken upstream for this checkpoint"; r2 predates it and already has the drift, so mega-attn is not the entry point.
- Whether the flips are a *correctness* regression. BFCL on v19 was 93.2% vs 93.3%; acceptance is the observable that moves.

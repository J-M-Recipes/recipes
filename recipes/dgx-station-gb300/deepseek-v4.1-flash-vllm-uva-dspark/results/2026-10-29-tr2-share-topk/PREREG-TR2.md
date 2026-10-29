# PREREG — Card TR2: L2 attention/indexer top-k table sharing on the GB300-only lane

Written 2026-10-29 ~04:25 CDT (box clock), **before the first bracket boot**. Frozen once the runner starts;
any change is a new card.

## What is under test

One axis: memoise `vllm.models.deepseek_v4_1.nvidia.flashmla.compute_global_topk_indices_and_lens` inside one
decode forward (`/l2/share_topk.py`, sha1 5c6943bd… before the verify-toggle edit), for layers that are not their
group's index source, keyed by `(kv_source_layer_id, index_source_layer_id, compress_ratio)` and validated by a
pointer/identity stamp. Everything else — image, hook, rowmap, k-schedule, `draft_sample_method: probabilistic`,
autotune cache — is byte-identical to the release the lane already documents as `v28-prob`.

`L2_SHARE=0` (arms A/A2): the patch is installed and reports counts, every kernel still runs. Behaviour is the
control's.
`L2_SHARE=2` (arm B): relaxed sharing — the stamp drops only `token_to_req_indices` / `is_valid_token` pointers.

## The claim this card tests

The op is a pure gather/count over identical inputs, so the shared table must equal the recomputed one **exactly**,
and the step should shed ~30 kernel launches (8 share groups over 38 compressed layers). The sidecar sibling
measured −0.117 ms/step (≈ +1%) from the same idea on af7f9488. Expected here: **+0.5 … +1.0 % C1**, zero numerics.

## Cells and bars (pre-registered)

Arms: **A → B → A2** on :30006, GB300 only, production regime (adaptive residency unfrozen, GO + ctrl).

| gate | cell | bar |
|---|---|---|
| **G0 boot** | all three | hash-hit (Loaded N + `Config cache hit` + autotune ≤ 60 s); `GPU KV cache size` **identical** on A, B, A2 (a python-only overlay must allocate nothing); `L2 installed` in the boot log on every arm |
| **G0b lever** | B | per-forward `L2 stats` show `hit > 0` and `src > 0`; A/A2 show `hit == 0` and `hit_noskip == 0` |
| **G1 speed** (primary) | `perclass_run_t.sh`, temp 1.0, 40 real prompts/class, 256 forced tokens, C1 + C8, r1/r2 | per class, B ÷ mean(A,A2) ≥ **0.98** (no class loses more than 2 %); headline = mean over the 5 C1 cells, **≥ 1.005 to call the lever WORKS**, 0.995–1.005 = NULL, < 0.995 = FAIL |
| **G2 identity (in-process)** | B, verify window at temp 1.0 (L2_VERIFY file toggle = 1) | `verify_bad == 0` and `verify_ok ≥ 100`. `verify_ok == 0` with hits > 0 = INSTRUMENT-BLIND (recorded, not a pass) |
| **G3 identity (teacher-forced)** | `tf_logprob.py` on both arms, prefill-only corpus | flips(B→A) ≤ flips(A2→A) + 0.05 pt, margin-weighted flips(B→A) == 0 |
| **G4 parity** | stock temp-0 per-class window | per class, B ÷ mean(A,A2) ≥ 0.97; `accepted`/`drafted` within the documented boot-to-boot band |
| **G5 quality** | gauntlet, temp 0, seed 42 | 12/12 on all three arms |
| **G6 hygiene** | every arm | 0 engine errors, 0 foreign POSTs (lane isolated), watchdog absent, KV unchanged |

A FAIL is any G0/G2/G3/G5 miss, or G1 < 0.995. A NULL (G1 in 0.995–1.005) parks the change: it is neither worth a
launch variant nor a risk.

## Instruments (absolute paths, unchanged)

`$HOME/dsv41/bench/perclass_run.sh`, `perclass_decode.py`, `$HOME/dsv41/m3/br1/perclass_run_t.sh`,
`perclass_decode_t.py`, `$HOME/pin-hot-experts/scripts/tf_logprob.py` + `tf_corpus.jsonl`, gauntlet
`$HOME/pin-hot-experts/round7/gauntlet/run_suite.py`.

## What this card does NOT claim

No 1M-context cell, no C16+, no soak, no BFCL/GSM8K (all temp 0, and the change is not supposed to touch numerics —
the identity gates carry that weight instead). Multi-user: the share key is per forward, so C8+ is exercised by the
C8 half of G1/G4 but not at C16+.
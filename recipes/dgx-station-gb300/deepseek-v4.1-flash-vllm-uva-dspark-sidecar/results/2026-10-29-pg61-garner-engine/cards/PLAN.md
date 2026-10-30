# Garner → sidecar recipe: plan (2026-10-29, desk; nothing booted)

Source: `garnermccloud/dgx-station-recipes` @ `8043a6811a5feda70fdfb6e827eebb2ec25c65d1` (MIT), `deepseek-v4.1-flash/`.
Local checkout `~/hermes/research/garner/dgx-station-recipes`; box copy `$HOME/dsv41/garner/8043a681/`.

## Desk facts (verified today, CPU-only)

| fact | evidence |
|---|---|
| **Same lineage as our sidecar lane.** Both run original-el8's `mega_peer_hook.py` (`MEGA_PEER=2`), a peer server on the RTX, `--moe-backend deep_gemm_mega_moe --enable-expert-parallel`, and a byte-identical `--speculative-config` | `docker inspect dsv41-vllm-1017-h2-REF`; his `serve/serve.sh:324-339` |
| His fast paths are a component layer **on top of** that hook (`DS41_FASTPATH` list, 29 components) | `fastpath/fastpath_hook.py`, `config/recipe.env` |
| His DeepGEMM fork bases (3 headers) are **byte-identical in our production image** (af7f9488) and in v0.31.0 | `serve/compat.sh` hashes vs `sha256sum` in both images |
| All 11 prebuilt `.so` load (ctypes, no GPU) in **both** images | desk_check |
| `vllm/vllm-openai:v0.31.0` already on the box, passes his compat assert | desk_check |
| His hook is 323 lines; ours is 610 (L2 share, split-KV ATT2b, LL GEMM, etc.) | sha1/wc |
| Disk free 465 GiB | `df -h /` |

## What his tree needs that our policy rules out

| component | needs | our policy |
|---|---|---|
| `attn` (W1 decode attention) | `nvfp4_ds_mla` KV: MXFP8 window (528 B) + NVFP4 compressed (288 B) | ruled out (KV quantization). Under `fp8_ds_mla` every call falls to stock → we keep our split-KV ATT2b |
| `preattn*` (W3) | written for `nvfp4_ds_mla` + MXFP4 indexer K; patches fall through when conditions fail | partly applies — **unknown how much until a boot reports its ACTIVE/stock split** |
| `lowb_aux` | MXFP4 indexer K store path | indexer K MXFP4 ruled out |
| `ds41_sampling` / `--override-generation-config` | forces T=1.0 / top_p 0.95 on thinking requests | policy change not agreed — drop |
| `w8oproj`, `lowb_aux`, `tail_drafthead_mxfp8` | change numerics by design (MXFP8 o_proj, K-order, MXFP8 draft-head copy) | each needs a quality gate; draft head is a requantized weight copy → James's "stock weights only" rule |
| `--max-model-len 524288` | — | we keep 1,048,576 |

Everything else (`peer3`, `megamoe` fork, `boundary`, `quant_producer`, the `tail_*` drafter fusions, `topp_fast`,
`engram_side`, `a12_*`, `fastload`, `coldskip`) is claimed bit-identical (or policy-free) and has no KV dependency.

## Hardware difference to keep in view

Our RTX is the **Max-Q Workstation Edition, 300 W**. His edition is not stated in the repo. The peer tier v3 kernel is
SM120 block-scaled QMMA + `cp.async.bulk` streaming; at 300 W the RTX's clocks are lower. G0 answers whether this
matters.

## Cards

**G0 — reproduce his release as shipped on our box (measurement only, nothing adopted).** v0.31.0, his 29
components, `nvfp4_ds_mla`, MXFP4 indexer, 512K, util 0.95, h255 rowmap, his sampling override, port **30006**
(his default 30003 is DSF production here — never). Lane dark + isolated; h2-REF restored after. Cells: his BRRR
fixed coding prompt ×10 (greedy, thinking off; his published 733.6 tok/s, 5.43 tok/step, 135.4 steps/s), his GSM8K
T=1.0 decode window at 1/8/32 streams, our per-class C1 + C8 harness, peer3 stats (`RTX busy`, `GB300 wait`),
boot gate (29 ACTIVE, 0 fallback lines). Same cells on h2-REF in the same window as the control.
Bar for continuing: steps/s ≥ 0.9× his 135.4 on the fixed prompt. Below that, the Max-Q is the story and the plan
narrows to peer3 alone. ~1.5 h dark (first boot JITs ~15 min).

**G1 — his tree under our policy ("bit-identical subset"), bracketed against h2-REF (A → C → A2).** v0.31.0 or our
af7f9488 image (both pass), `fp8_ds_mla`, default indexer, 1M, `--kv-cache-memory 9126805504`, 48 GiB host prefix tier,
no sampling override, components minus {`attn` (no-op here), `ds41_sampling`, `ds41_stop`, `w8oproj`, `lowb_aux`,
`tail_drafthead_mxfp8`}. Gates = the sidecar recipe's own: TF flips ≤ control+1 (0 confident on agent/tool),
gauntlet 12/12, per-class C1/C8 temp 1.0, C32 survives in one process (the ATT2 lesson), 1M needle, watchdog
counters 0. Win bar: C1 ≥ +10% on the per-class mean with no class < 0.98 and C8 ≥ 1.0.

**G2 — numerics-changing components one at a time** (`w8oproj`, then `lowb_aux` if the indexer path engages, then
`tail_drafthead_mxfp8` only if James lifts "stock weights only" for a drafter-only copy). Each: full quality gate
(TF + GSM8K-200 paired + BFCL held-out) before any speed reading counts.

**G3 — the KV question, priced.** The G0-vs-G1 step delta that comes from `attn`/`preattn` disengaging is the speed
cost of the `fp8_ds_mla` ruling. Report it in ms/step; James decides.

## Credit

Every mechanism above is Garner McCloud's (MIT, @8043a681), built on original-el8's hook (Apache-2.0), DeepGEMM
(MIT), b12x (Apache-2.0), vLLM. A recipe that adopts any component names him in `credits` and links the commit.

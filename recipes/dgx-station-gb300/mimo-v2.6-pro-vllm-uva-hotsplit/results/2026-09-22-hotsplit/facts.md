# Receipts: routed-expert histogram + hotsplit (September 22, 2026, 12:26 PM – 4:22 PM CDT)
All numbers from tool output on the GB300 lane. Session model: anthropic/claude-opus-5-5 (Milo), blog drafting via xai/grok-4.7 delegate.

## Histogram 1 (8 agent-shaped prompts, 2,736 generated tokens, full 70 layers, eager)
- First attempt crashed: counting hook used CPU tensor during CUDA-graph capture. Fixed with eager + capture guard.
- Early layers (1–22): ~100–124 of 384 experts carry 50% of traffic; ~0 never routed (layer 8 exception: 128 never routed).
- Late layers (30–69): ~48–66 experts carry 50%; 10–37 never routed per layer.
- Pooled over all (layer, expert) cells: top 10% of cells → 34.2% of traffic; top 20% → 51.6%; top 30% → 64.5%; top 50% → 83.0%.

## Histogram 2 (real Hermes turns — private corpus, never published)
- 900 conversation windows sampled from real Hermes agent sessions (private; content not published, counts are); 630 train / 270 held out.
- Train: 2,516,611 prompt tokens, 62,399 generated, 2,553 s. Holdout: 1,061,333 prompt, 26,357 generated, 1,075 s.
- Counted decode (≤16 tokens/step) and prefill separately.
- Stock placement with --cpu-offload-gb 320: layers 1–48 routed experts in Grace (UVA), layers 49–69 in HBM (verified in v18 log).
- Holdout decode traffic served from HBM:
  - stock (layers 49–69 whole; 8,064 cells ≈ 30% of cells): 30.4%
  - same budget, hot list ranked on TRAIN decode counts: 62.2%
  - oracle (holdout ranks itself) at same budget: 62.5%
  - ranking by prefill counts only: 54.9% (decode-ranked is better; mixing in prefill at 0.25 weight: 62.0%)
  - other budgets (train decode ranking): 20% cells → 49.3%, 40% → 71.6%, 50% → 79.8%

## hotsplit (experiment code, ~/hermes/research/mimo26_gb300/hotsplit.py; NOT upstream)
- After process_weights_after_loading, each MXFP4/Marlin routed-expert layer is split into a hot set (compact HBM tensors) and a cold set (compact exact-size pinned host tensors via UVA). Two Marlin launches per layer, each with an expert_map (global→local, −1 = absent), outputs summed. Uses vLLM's existing EP expert_map path; no kernel changes.
- Default budget = exactly the HBM stock gave routed experts, so KV is not reduced.
- 16-layer parity (stock vs hotsplit, same boot recipe, 40 GiB offload): 6/6 greedy outputs token-identical; residual relative diff grows smoothly to 5.0e-4 by layer 15 (max abs 0.0020); max |Δ logprob| on chosen tokens 0.067. Consistent with fp summation-order change from summing two partial outputs.
- First full boot (v19) OOM-killed during split: launched with the old lane's cudaHostRegister-patched uva.py, whose freed buffers didn't return host memory (~3 GiB/layer leak; host avail 82 → 36 → 2.6 GiB at 10/20/30 layers). Fix: stock uva.py + VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 (exact cudaHostAlloc path), explicit scale free, and a host-memory floor that leaves remaining layers stock instead of dying.
- v19b: 69/69 layers split in 33 s (29 s on restart); plan 8,064/26,496 cells hot, 141.8 GiB budget (= stock HBM experts), train-weighted coverage 60.2%; after split HBM free 45.6 GiB, host avail ~111 GiB; GPU KV 511,074 tokens (stock v20: 490,466). Sanity: "Paris", "Jupiter".

## Matched A/B (both: stock loader, stock uva.py, DISABLE_PIN_MEMORY=1, Marlin, off320, FULL_DECODE_ONLY cudagraphs, same image nightly-d05da62e; only difference is HOTSPLIT on/off). Warm runs only.
Streaming TTFT/decode bench (prompts ~3k/11.5k/46k tokens, 128 output tokens, greedy, 3 reps, median):
| prompt tokens | v20 control TTFT / prefill tok/s / decode tok/s | v19b hotsplit TTFT / prefill / decode |
| 2,952 | 2.76 s / 1,071 / 31.3 | 2.52 s / 1,170 / 35.3 |
| 11,521 | 9.15 s / 1,259 / 31.2 | 8.55 s / 1,348 / 35.1 |
| 45,826 | 35.40 s / 1,295 / 31.0 | 34.01 s / 1,347 / 34.7 |
Decode +13% (31.2 → 35.1 at 11.5k), prefill +7%.

Agent fixture (tokens/s, warm runs r2 and r3; they agreed within 0.1):
| task | v20 control | v19b hotsplit |
| tool_json | 29.5 | 35.9 |
| code | 30.3 | 31.1 |
| shell_ops | 28.6 | 35.7 |
| structured | 30.0 | 34.1 |
| prose | 30.3 | 30.1 |
Gains are largest on tool/shell/structured (agent-shaped, matching the corpus the hot list was built from); prose unchanged.

Knee sweep (192-token prose "paragraph about the number N" prompts, ignore_eos, short prompts):
- v19b C1: first sweep 20.4 (runs 27.8 / 12.9 — first-run warmup outlier); 6 clean C1 samples 28.2–28.5. C2 36.3, C4 45.8, C8 57.8, C12 53.3, C16 57.4.
- v20 control C1: 30.0 ×6.
- So on that synthetic prose prompt hotsplit is ~5% SLOWER at C1; on agent-shaped and long-context streaming it is 13–25% faster. The hot list is workload-specific; prose about numbers is outside the corpus it was ranked on.
- Earlier references: v14 (patched uva/mxfp4, default loader) C1 30.2, C8 56.2. v20 control reproduces v14 without either patch.

## Methodology corrections recorded
- A cold first run of the agent fixture reads low (v20 tool_json 23.8 cold vs 29.5 warm); only warm runs compared.
- My pre-control reading ("hotsplit neutral/negative, fetches not the bottleneck") was wrong; it came from a synthetic prose knee sweep plus a cold fixture. The matched warm A/B overturned it.

## Lane state at 4:22 PM CDT
- :30007 = v19b hotsplit (experiment lane), serving. Fallbacks kept: v20 control container (stock loader, no patches), v18 (patched).

## Addendum 5:05-5:50 PM CDT (v21, v22, v23)
- v21 mixed ranking (agent decode + 8-prompt histogram x0.25): held-out agent decode coverage 61.0% (agent-only 62.2%). Warm: decode 36.4/36.3/35.7 at 3k/11.5k/46k; tool_json 35.7, shell 35.4, structured 34.8, code 31.9, prose 31.6; synthetic prose C1 29.6-30.0; C8 60.8. KV 513,612 tokens.
- v21 torch-profiler trace, 64 C1 decode tokens: 28.8 ms/token span, 96% GPU busy. Marlin 17.3 ms/token (hot w13+w2 2.10+1.29 ms, cold 9.16+4.79 ms), dense GEMM 5.8, routing/align/sum 1.6, attention 1.1, norms 0.8.
- v22 hot budget 152.8 GiB (8,692/26,496 cells): KV 302,368 tokens. Decode 37.5/37.4/36.9; tool_json 36.6, shell 36.4, structured 35.9, code 32.6, prose 32.7; synthetic prose C1 29.7-30.1 (knee 30.7); C8 63.7.
- v23 = v22 + live counter: decode 37.1/36.9 at 11.5k over two invocations (36.9/37.1/37.0; 36.9/37.2/36.9); fixture tool_json 36.2, shell 36.0, structured 35.4, code 32.3, prose 32.3 (single warm run). First snapshot 69 layers, 5,052 decode tokens.
- Dense GEMMs per layer: o_proj BF16 nvjet split-K 36.1 us median, qkv FP8 CUTLASS blockwise 35.9 us median; 70 of each per token; 5.0 ms/token together.

# DFlash2-over-UVA experiment contract

Frozen before the first candidate completion was served on September 7, 2026.

## Fixed artifacts

- Target: local `incoai/GLM-5.3-NVFP4` intent at revision `54e52520606f96b3d9fc84088ad22882a61648ac`.
- Draft: `incoai/GLM-5.3-DFlash2` at revision `425aa615ce320caac34400208b30808c8f14f76c`.
- Draft weight SHA-256: `3105f14043bef642baa49a7d533fdf0b8b2895737ec84b6305601da662656161`.
- Engine: `vllm-glm53-uva:v0.28.0-2cf0a691`, local image ID `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`.
- Candidate: sc13g slot-cache map, eager compilation, DFlash K4, 65,536-token server window, 8 GiB bf16 KV, `max_num_seqs=4`.
- Preserved rollback: the exact pre-experiment 512K/MTP container, stopped and renamed rather than rebuilt.

## Gates, in order

1. **Geometry:** draft hidden size, intermediate size, vocabulary, target layer count, and target-layer IDs must match the target contract.
2. **Acceptance stop gate:** weighted acceptance length on the fixed two-prose/two-code battery must be at least `3.0`, computed as completion tokens divided by vLLM's `spec_decode_num_drafts_total` delta.
3. **Short-window throughput:** only after acceptance passes, run the existing warmed `bench3.sh` C1/C4/C8 harness. Promotion requires the prose means to beat the recorded sc13g+MTP(1) rows at all three shapes: 54.7 / 107.8 / 102.9 aggregate tok/s.
4. **Decode-path quality:** capture two no-spec sc13g self-repeat controls and one DFlash run with `tf_decode.py`; require the existing self-repeat-floor gate to pass. Also require the fixed 20-prompt greedy output battery to match the no-spec sc13g control 20/20.
5. **Daily-profile promotion:** if gates 1-4 pass, relaunch the DFlash candidate with the 512K/48 GiB-KV daily profile. It must boot, pass authenticated model/text/tool health checks, and beat the measured MTP decode-at-occupancy baseline of 46.144 tok/s on the same 477,675-token cached-prefix probe before it can be left up.

If any gate fails, stop the candidate, restart the preserved pre-experiment 512K/MTP container, and verify it before publishing the result.

## Pre-acceptance startup amendment

Attempt 1 loaded the target and draft, built the slot cache, and then failed before API bind while vLLM captured full CUDA graphs. The slot-cache statistics hook attempted an unsupported CUDA operation during stream capture, invalidating the graph. This is an integration failure, not an acceptance result.

Before attempt 2, the candidate contract is amended to add explicit `--enforce-eager`. The existing `--compilation-config '{"mode":3,"backend":"eager"}'` controls the compile backend but did not disable CUDA-graph capture. `--enforce-eager` is also the measured launch posture of the earlier GLM-5.3-Flash DFlash2 recipe. All numerical gates remain unchanged; the mode difference will be disclosed in comparisons.

## Interpretation limit

This is one GB300 with target experts offloaded through UVA. It does not revise the HBM-resident DFlash2 results published by inco.ai or catid. A failed transfer here is a memory-path result, not a model-quality verdict.

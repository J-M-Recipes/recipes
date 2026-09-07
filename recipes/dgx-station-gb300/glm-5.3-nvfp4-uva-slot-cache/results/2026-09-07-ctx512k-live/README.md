# 2026-09-07 — Long-context serving profile: 512K / 48 GiB KV (live receipts)

On September 7, 2026 we tested three context profiles for the sc13g slot-cache + MTP(1) build on one GB300 and settled the daily serving profile at **512K context (524,288 tokens) with a 48.0 GiB bf16 KV cache**, chosen to balance decode speed against context headroom. This directory records the decision facts and the live verification receipts. It does not change any quality verdict: all quality and promotion statements elsewhere in this recipe still stand, and the MTP/structured-output caveats are unaffected by context sizing.

## Why 512K

Every extra GiB of KV comes directly out of the HBM expert-slot budget, and decode speed follows the slot budget. The three profiles compared:

| profile | context | bf16 KV | expert slots (per-layer range) | mean predicted hit allocation | outcome |
|---|---:|---:|---:|---:|---|
| ctx256k | 262,144 | smaller than 48 GiB (not the constraint) | 7,360 (64–176) | 0.6982 | fastest slot budget, but context was not the pain point; not compelling |
| ctx512k | 524,288 | 48.0 GiB | 5,792 (48–96) | 0.6166 | **selected daily profile** |
| ctx1m | 1,048,576 | 96.0 GiB | 2,672 (32–48) | 0.4016 | loads and serves, but roughly halves the slot budget; aborted as a daily profile and kept only as a special long-context option |

"Mean predicted hit allocation" is the slot-weighted average of the per-layer hit-rate curves used to derive each allocation map (`trace/r1-base/slots-*.json` on the campaign host); it is a planning figure computed from routing traces, not a measured runtime hit rate.

The 1M launch (`glm53-big-sc13g-mtp-ctx1m`, 96 GiB KV, 2,672 slots) was aborted during startup on 2026-09-07 because the profile looked too slow to justify as the everyday configuration; the container was stopped and archived as `glm53-big-sc13g-mtp-ctx1m-superseded-by-512k-20260907-085813`. The 512K container was launched immediately after.

## Live launch flags (receipt `ctx512k-20260907-090009`)

Container `glm53-big-sc13g-mtp-ctx512k`, started 2026-09-07T14:00:10Z, image `vllm-glm53-uva:v0.28.0-2cf0a691`:

- `--cpu-offload-gb 420 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight`
- `--kv-cache-dtype bfloat16 --kv-cache-memory 51539607552` (48.0 GiB) `--max-model-len 524288`
- `--max-num-seqs 1 --max-num-batched-tokens 8192`
- `--compilation-config {"mode":3,"backend":"eager"}`
- `--speculative-config {"method":"mtp","num_speculative_tokens":1}`
- `SLOT_CACHE=112`, `SLOT_CACHE_PER_LAYER=slots-5792-ctx512k.json` (sha256 `73962276498626af16baaa7c8737384960aff8718d254e66ab92610570c9743c`, packaged in this repo as `configs/slots-5792-ctx512k.json`), `SLOT_CACHE_ROUTER=ffi`, `EXACT_PIN=1`

Engine log milestones (2026-09-07 14:21:54 container log):

- Total CPU offloaded parameters: 337.5
- Initial free memory 249.12 GiB; reserved 48.0 GiB for KV cache via `kv_cache_memory_bytes`
- GPU KV cache size: 548,800 tokens; maximum concurrency for 524,288 tokens per request: 1.05x
- Application startup complete

## Verification

- Near-window probe: a 480,011-prompt-token request (1,474,255 bytes) completed successfully in 142.723 s cold (the first attempt with `max_tokens 16` hit the length cap with 16 reasoning tokens — a probe artifact, not a serving failure); an immediate repeat returned the exact string `CTX512K OK` in 1.393 s with the prompt cached.
- Post-test health: container still running, no OOM or traceback in logs; compute GPU showed 238,908 MiB used / 16,902 MiB free after the probes.
- Hermes client wiring: declared `glm53` context was corrected from 65,536 to 524,288 on the M4 Milo profile and on Forge/Echo (provider and nested model entries); both clients then answered `HERMES 512K OK`.

## Receipt files

- `receipt-facts.json` — distilled public-safe facts above (launch flags, log milestones, probe timings, slot-map hashes).
- Source receipt on the campaign host: `/home/milo/big-v1-campaign/receipts/ctx512k-20260907-090009/` (`container-id.txt`, `gpu-before.txt`, `ram-before.txt`, `slots-5792-ctx512k.json`).

## Limits

- The 480k-token probe proves the 512K window is genuinely operational, not a launch-time claim; it does not measure decode throughput at that occupancy, and no quality gate was rerun at 512K.
- The packaged `scripts/launch-slotcache-portable.sh` still defaults to the measured sc13g 8 GiB / 65k flags; the 512K profile is launched from it via the documented `KV_CACHE_MEMORY`, `MAX_MODEL_LEN`, and `MAX_NUM_SEQS` overrides, or via the raw docker command above.
- `--max-num-seqs 1` in the live 512K run means one in-flight request at full window (1.05x concurrency); concurrency-vs-context tradeoffs were not re-benched.
- Predicted hit-allocation figures come from routing traces on one workload; runtime hit rates follow the actual workload.

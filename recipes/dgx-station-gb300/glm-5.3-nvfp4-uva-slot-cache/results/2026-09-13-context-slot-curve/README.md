# 2026-09-13 — context ↔ expert-slot curve (256K / 512K / 1M), matched window

**Verdict: 256K context is the new daily profile.** Same image, same model, same MTP(1), same 8 prompts, same host, one axis changed (KV reservation ↔ HBM expert slots). Decode speed follows the slot budget almost linearly with expert-cache misses; greedy output is byte-identical across all three profiles.

| profile | bf16 KV | slots | live hit rate (mean of 20 s windows) | misses / step / layer | 512-token decode median (n=8) | vs 512K |
|---|---:|---:|---:|---:|---:|---:|
| **ctx256k** | 24 GiB | 7,360 | 0.61 (n=15) | 3.12 | **51.29 tok/s** (49.2–53.3) | **+14.4%** |
| ctx512k | 48 GiB | 5,792 | 0.53 (n=14) | 3.77 | 44.83 tok/s (42.9–46.1) | — |
| ctx1m | 96 GiB | 2,672 | 0.17 (n=19) | 6.82 | 34.14 tok/s (33.3–37.4) | −23.9% |

Quality gates (greedy, 20 fixed prompts, `scripts/greedy_equiv.py`):
- self-repeat ×3 within each lane: **20/20** and **20/20** on all three profiles (K=1 is bit-stable on this build);
- cross-lane 512K vs 256K: **20/20 identical**; 512K vs 1M: **20/20 identical**.

Container identity: all three were the September 7 one-axis clones of the sc13g-mtp incumbent (image `vllm-glm53-uva:v0.28.0-2cf0a691`); `docker inspect` env/cmd/binds diff is exactly `SLOT_CACHE_PER_LAYER`, `--kv-cache-memory`, `--max-model-len`. 1M engine init after weight load: 92.0 s; KV capacity 1,097,600 tokens (1.05× concurrency at 1,048,576).

## Caveats (read before quoting)
- One window, 8 × 512-token requests per lane, short prompts (~50–60 tokens). Not a long-prompt benchmark; 1M was measured at short context to isolate the slot cost, not its long-context prefill.
- **512K baseline probe artifacts:** TTFT (~12 s) and the engine-wide `metric_delta` acceptance figures for the 512K lane are **invalid** — another client hit the endpoint during that probe (draft counts ≈2× the completion tokens). `decode_tok_s` is measured after first token and matches the September 9 historical 45.747 tok/s, so decode stands; do not quote 512K acceptance or TTFT from this window.
- Hit rates are the engine's live `SLOT_CACHE STATS` 20 s windows during the probes, not the planning "predicted hit allocation" numbers (0.698 / 0.617 / 0.402). Live is lower across the board; the ordering matches.
- The September 7 1M abort ("too slow during startup") was cold-load impatience, not a failure; the lane loads and serves.

## Files
- `k1-*/greedy-r{1,2,3}.json`, `selfrepeat-*.txt` — greedy outputs and in-lane equivalence
- `crosslane-512v256.txt`, `crosslane-512v1m.txt` — cross-lane equivalence
- `k1-*/probe-512-{a,b}.json` — `dflash2_acceptance_probe.py --max-tokens 512`, 4 requests each
- `k1-*/slotcache-stats.txt` — engine `SLOT_CACHE STATS` lines captured during the window
- `k1-*/nvidia-smi.txt`, `window.log`, `window_lane.sh` — runner and host state
- `SUMMARY.json` — the table above, computed from the rows

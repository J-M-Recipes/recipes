# 2026-10-24 — Card KV3: a 48 GiB Grace host prefix-KV tier on the sidecar lane

One change to the sidecar lane: two vLLM flags, `--kv-offloading-size 48 --kv-offloading-backend native`, added to
the published launch (which already declares `--kv-cache-memory 9126805504`). vLLM's `OffloadingConnector`
(`CPUOffloadingSpec`) then keeps evicted prefix blocks in a 51.54 GB `/dev/shm` mmap in Grace memory and loads
them back on a prefix hit instead of recomputing them. The GPU KV pool does not shrink: the tier is additive.

Base config for all three arms: the v25m variant (`ROWMAP=rowmap-mixed-ar-v1.json` on both steps), image
`vllm/vllm-openai@sha256:3af6a45b…` (nightly af7f9488), 32 seats, 1M context, DSpark on. Measured 2026-10-24
21:22–22:35 CDT, bracketed **K0 (no tier) → K1 (tier) → K0R (no tier)**, a fresh sidecar+server boot per arm,
fail-closed watchdog armed in enforce mode on every arm.

**Verdict: PASS, 10/10 pre-registered gates** ([`VERDICT.txt`](VERDICT.txt), [`verdict.json`](verdict.json)).

| gate | result | detail |
|---|---|---|
| G1 GPU KV tokens | PASS | 4,400,013 on K0, K1 and K0R |
| G2 C1 (held-out mixed JP/DE + agent, 6 × 120 s) | PASS | K1 219.89 vs controls 219.48 (+0.19%); bar −2.04% = max(2%, 1.5 × pooled spread 1.36%); drift K0→K0R 0.26% |
| G2 cold 64K TTFT proxy | PASS | 1.424 s vs 1.404 s (+1.47%, bar +5%) |
| G2 warm replay r2 | PASS | 177.48 vs min(K0 195.52, K0R 183.72) = 183.72 (−3.40%, bar −5%) |
| G3 tier works | PASS | evict → `/reset_prefix_cache` → replay: host-tier hits, loads, identical outputs, recall, at 7,929 and 31,629 tok (mod 256 = 249, 141); replay ×1.54 / ×3.2 (info); connector present on K1 only; 0 pin warnings |
| G4 watchdog | PASS | 0 timeouts, 0 trips, alive, all arms |
| G4 errors | PASS | 0 server / 0 harness / 0 sidecar error lines, all arms |
| G4 1M-class needle | PASS | 954,028-token prompt recalled on all arms, 32.8–33.0 s |
| G4 host memory | PASS | MemAvailable min 129.8 GiB with the tier (floor 60); 0 new OOM-kill lines |
| G5 gauntlet | PASS | 12/12 on K1 |

| arm | KV tokens | C1 mean | C1 spread | C1 post-60 s | cold 64K s | warm r1 | warm r2 | needle | MemAvailable min boot/cells GiB |
|---|---|---|---|---|---|---|---|---|---|
| K0 | 4,400,013 | 219.76 | 2.09% | 221.88 | 1.401 | 183.86 | 195.52 | 32.9 s | 161.3 / 177.7 |
| **K1 (tier)** | 4,400,013 | **219.89** | 0.96% | 224.29 | 1.424 | 195.83 | 177.48 | 32.8 s | 131.4 / 129.8 |
| K0R | 4,400,013 | 219.20 | 0.63% | 222.51 | 1.406 | 179.38 | 183.72 | 33.0 s | 162.6 / 178.9 |

Warm replay (one 8,172-token prompt, 256 forced tokens, run twice) is a noisy cell: the two controls' r2 span
183.7–195.5, and K1's r1 (195.83) is the highest r1 of the three. The −3.4% r2 sits inside that spread.

## What the tier holds

The evict-replay deltas in [`raw/K1/evict.json`](raw/K1/evict.json): 12,930,048 B loaded for 7,808 hit tokens and
52,356,096 B for 31,616 hit tokens, both **1,656 B per hit token** (fp8_ds_mla, 128-token hit granularity on this
lane). 48 GiB / 1,656 B ≈ **31M prefix tokens** held in Grace on top of the 4.40M GPU tokens. Measured on this lane
and this image only; another model or KV format changes the bytes per token.

## Files

| file | what it is |
|---|---|
| `VERDICT.txt`, `verdict.json` | gate table and per-arm table as written by `scripts/kv3_verdict.py` |
| `SUMMARY.txt`, `runner.log`, `phases.jsonl` | the runner's timeline: boots, facts, every cell's one-line result, restore |
| `instruments.txt` | sha1 of every script as run **on the box** (the published copies here differ only by the path scrub), image digests, per-arm EXTRA |
| `launch-K*.txt`, `health-*.json`, `errcount-*.txt`, `oom-*.txt`, `mem.tsv` | launcher output, chat health, error-line counts, OOM-kill counts, 5 s MemAvailable samples |
| `raw/K0`, `raw/K1`, `raw/K0R` | per-arm boot log, boot facts, C1 rep logs, `mixed-*.json.gz` (per-request timings only — synthetic JP/DE + agent pools, no prompt text), cold-64K, warm, needle, evict / evictctl, watchdog status |
| `raw/postlog-*.log`, `raw/restore-ref.log` | server and sidecar logs after the cells; the restore boot of the lane that was serving before the card |
| `raw/gauntlet-K1/` | gauntlet summary + sandbox isolation check + per-task results (gzipped); fixture inputs omitted |
| `LANE-OWNER.before.txt`, `watchdog-stale-212415.log` | lane ownership/soak ledger at card start (records the cancelled v25m and v26 soaks); a watchdog log left over from the 20:39 smoke |
| `scripts/` | `launch_kv3.sh`, `cardKV3_runner.sh`, `kv3_cells.py`, `kv3_verdict.py` exactly as run, box paths scrubbed |
| `throughput.csv` | every cell above with its receipt |

## Read before reusing

- **`launch_kv3.sh` sets `VLLM_SERVER_DEV_MODE=1`. That was for this card only:** it enables
  `/reset_prefix_cache`, which the evict-replay probe needs. The production variant is the two flags on the
  published `scripts/launch-sidecar.sh` (`EXTRA="--kv-cache-memory 9126805504 --kv-offloading-size 48
  --kv-offloading-backend native"`), no dev mode.
- **Upstream [vllm#47890](https://github.com/vllm-project/vllm/issues/47890)** reports speculative decoding
  zeroing CPU-tier hits. We tested for it: with DSpark on, both evict-replay lengths (not multiples of 256) hit
  and loaded from the host tier. The K1 boot logs `KV offloading: speculative decoding is enabled but no KV-cache
  group is annotated as a drafter group; treating all groups as non-draft for offloading`.
- The `vllm:kv_offload_*` metrics only exist on K1; the controls' `evictctl` cell replays the same documents
  without a reset (recall only). Cross-arm evict outputs are identical to the controls (8,205- and 33,010-target
  prompts).
- Identity here is on short outputs (a 12-character code per prompt). It shows the reloaded KV recalls the
  document; it is not a teacher-forced fidelity run, and none was run on K1.
- Path scrub: the box home directory → `~`, LAN address → `<gb300>`. The `mixed-*.json` and gauntlet `task_results.jsonl`
  are `gzip -n` for size (bytes otherwise as scrubbed).

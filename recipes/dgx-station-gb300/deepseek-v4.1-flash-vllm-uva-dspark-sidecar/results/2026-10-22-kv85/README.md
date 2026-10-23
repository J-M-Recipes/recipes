# 2026-10-22-kv85 — explicit KV sizing on the sidecar lane

One change to the serving lane of `2026-10-22-v25-sidecar`: the KV pool is **declared** with
`--kv-cache-memory 9126805504` (8.5 GiB) instead of being whatever `--gpu-memory-utilization 0.97` leaves after
vLLM's memory profile (4.9 GiB). Same image, same hook, same 295/89 map, same flags otherwise; the FlashInfer
autotune cache hit (`Loaded 159 configs`), so no kernel changed. Measured on the live lane, 2026-10-22 18:43–19:18 CDT.

| | util 0.97 (v25 as first promoted) | `--kv-cache-memory` 8.5 GiB (this bundle) |
|---|--:|--:|
| KV pool | 4.9 GiB | **8.5 GiB** |
| KV tokens at 1M max-model-len | 2,537,081 | **4,400,013 (+73%)** |
| Concurrency for 1,048,576-token requests (boot line) | 2.42× | **4.20×** |
| GB300 memory used at idle / at stress peak | 253,140 MiB / — | 254,786 / **255,308 of 256,703 MiB** |

## Cells

| file | what it shows |
|---|---|
| `raw/kv1/boot-facts.txt` | the serving boot: `reserved 8.5 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling`, `GPU KV cache size: 4,400,013 tokens … 4.20x`, peer mode 2, 40-layer rowmap, replay line, autotune hit |
| `raw/kv1/verdict.txt` | Card KV1, 7 gates PASS: capacity, flag, 998,363-token needle recalled (TTFT 35.45 s), post-probe, 1M-depth decode 256 tok/s, C1 after/before the 1M request 0.984, 0 load errors |
| `raw/kv2/A-speed.txt` + `A-c1-*.json`, `A-c16.json` | speed on the 8.5 GiB lane with RonanLabsAI's `kbench_chat_ds` and CNN/DM prompt pool (8.2K real text in / 1,024 forced out): **C1 275.1 tok/s** (6 runs: 278.4 268.7 276.8 282.6 275.9 268.1, sd 5.2), **C16 1,575.9** (32/32) |
| `raw/kv2/A-s1-*` | S1 mixed load, all at once: ~1M needle (recalled, TTFT 40.1 s) + cold 131K prefill + kbench C16 (32/32) |
| `raw/kv2/A-s2-*` | S2: four ~1M needles submitted together — all recalled, TTFT 38.8 / 65.7 / 93.2 / 86.4 s. **Not a capacity test:** they ran two at a time and the KV pool peaked at 27% |
| `raw/coresident/` | the capacity test: four ~990K-token prompts each decoding 16,000 tokens (`ignore_eos`), so all four are resident at once — **4 running / 0 waiting, KV usage peak 75.3%**, all 16,000-token completions, 0 preemptions |
| `raw/kv2/A-stress-checks.json` | across S1 + S2: 0 engine-error lines, 0 watchdog trips, sidecar timeout counter `words[2]` 0 → 0, both containers up, chat health OK |
| `raw/kv2/A-gpumem.txt`, `raw/coresident/coresident.json` | 1 s GB300 `memory.used` samples: 254,786 MiB idle → **255,308 MiB peak**, no further growth under the co-resident cell |
| `scripts/` | the runners and the co-resident probe exactly as run (box paths scrubbed) |

## Corrections and instrument failures, kept on purpose

- **KV1 run 1 rolled back on an instrument bug, not the change.** The runner filtered the server log with
  `docker logs --since "<local time>"`; the container logs in UTC, the filter returned one line, the boot-facts
  check read "peer mode 2 missing" and the runner restored the 4.9 GiB pair (lane dark 17 min). The boot itself
  had reached 4,400,013 tokens. `raw/kv1/run1-INSTRUMENT-FAIL-runner.log`.
- **KV1's `flag` gate** looked for `kv_cache_memory` in a facts file that truncates the `non-default args` line at
  400 characters. Before the verdict ran, the real log line (`reserved 8.5 GiB … kv_cache_memory_bytes`) was
  appended to `facts.txt` from `docker logs`; the 4.40M-token figure is the independent evidence.
- **KV2 `A-verdict.json` says FAIL; the cell passed.** The runner's trip counter printed `0` twice (`grep -c`
  exits 1 when nothing matches), so `A-stress-checks.json` was invalid JSON and `no_errors` / `alive` read FAIL.
  The file in this bundle was rewritten from the runner's own `STRESS CHECKS` log line and says so in a `note`
  field; the original FAIL verdict is kept beside it. The runner then skipped arm B (9.5 GiB) — see below.

## What was not measured

- **No same-window A/B between 4.9 and 8.5 GiB.** The kernels and the autotune store are identical, so none is
  expected; C1 275.1 here vs 279.5 (mean of two boots, `2026-10-22-v25-sidecar`) is inside the few-percent
  boot-to-boot spread of live FlashInfer tuning.
- **Vision requests and C32 prefill bursts** were not part of the stress.
- **9.5 GiB was not booted.** At 8.5 GiB the GB300 has 1,395 MiB free at the observed peak; +1 GiB of KV would leave
  ~370 MiB, against +0.5M tokens on a pool that peaked at 75% with four 1M contexts live. An OOM is a lane crash
  (fail-stop watchdog), so it was declined.

## Side finding

A cold ~1M-token prefill holds up everything queued behind it. In S1 the concurrent 131K prompt waited 34 s for its
first token (3,857 tok/s effective, vs 2.61 s / 50K tok/s alone in SC5) and the C16 aggregate fell from 1,576 to
718 tok/s while the 1M prompt was being read. This is the scheduler's long-prefill policy
(`--long-prefill-token-threshold 6144`), not KV.

## Credit

Every real-text speed cell here (C1, C16, the S1 C16 burst) was measured with **RonanLabsAI's** `kbench_chat_ds.py`
and prompt pool from [dgx-station-frontier](https://github.com/RonanLabsAI/dgx-station-frontier), the same harness as
the sidecar round; the lane itself is his sidecar recipe on original-el8's hook. The KV sizing, stress cells and
the 1M needle probe are ours.

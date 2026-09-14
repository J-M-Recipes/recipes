# Round 3 — batch-size K-schedule promoted to v13; decode-vs-depth map — 2026-09-14

Three one-axis experiments on the `:30006` lane, each with a same-window control on the v12 reference container
(`OFFGB=60 UTIL=0.97 SEQS=16 CTX=1M dspark k=5 lpt 6144`, image `deepseekv41-flash-0909`). Operator plan:
`TEST-PLAN-round3-2026-09-14.md` (Milo work dir); runner `campaign_round3.sh`, instrument `depth_knee.py` (both here).
All runs 17:28–18:28 CDT, lane otherwise idle, `:30003` untouched.

## E3 — `num_speculative_tokens_per_batch_size=[[1,2,5],[3,16,1]]` → **v13** (win, two pairs)

k=5 while 1–2 sequences are running, k=1 at 3–16. Config-only: the drafter, verifier and weights are untouched
(vLLM `SpeculativeConfig.num_speculative_tokens_per_batch_size`, entries are `(range_start, range_end, k)` inclusive).
Motivation: the 2026-09-12 k-sweep showed static k=1 wins every multi-stream point by 23–33% and loses agent text at
C1 by a third; static k=5 the reverse. The schedule takes each where it wins.

| knee (tok/s) | C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|---|
| pair 1 control (v12) | 90.3 | 126.7 | 174.0 | 257.8 | 256.2 | 317.6 |
| pair 1 **v13** | 91.2 | 127.7 | **224.5** | **320.1** | **339.2** | **430.8** |
| pair 2 control (v12) | 90.9 | 127.6 | 177.7 | 247.4 | 276.8 | 320.0 |
| pair 2 **v13** | 90.2 | 126.7 | **222.2** | **318.0** | **339.3** | **427.1** |
| **mean Δ** | +0.1% | −0.0% | **+27%** | **+26%** | **+27%** | **+35%** |

Candidate-vs-candidate spread across pairs ≤1% at every C. Boot 151 s both times — the schedule does **not** change
the FlashInfer autotune hash (`9ac7b387`, `Loaded 210 configs`), KV 4.87 GiB / 2.21× at 1M, offloaded 61.17 GiB:
the v12 memory picture is unchanged.

**Single-stream fixture (C1, `agent_fixture.sh`)** — as expected for a schedule that is k=5 at C1:

| | prose | shell_ops | code | tool_json | structured | wacc |
|---|---|---|---|---|---|---|
| ctrl p1 / p2 | 99.0 / 99.3 | 165.2 / 163.2 | 155.8 / 156.5 | 157.6 / 158.7 | 124.2 / 124.8 | 0.596 |
| v13 p1 / p2 | 100.0 / 98.8 | 163.2 / 162.3 | 156.8 / 155.8 | **146.8 / 146.5** | 125.2 / 124.2 | 0.596 |

tool_json is −7% in both pairs and is the one number that moved; acceptance is identical, so it is not the drafter.
The fixture runs its classes sequentially and tool_json is 318 tokens with a tool round-trip; today's four v12
controls put it anywhere from 145 to 159, so this is inside the class's own spread but consistent in sign. Flagged, not
explained. Prefill, starvation and harness numbers are not spec-dependent and were not re-run.

Promotion bar (set before the run): C1 within ±1.5% of control **and** C8/C16 ≥ +15%, on two same-window pairs. Met.

## E2 — `--async-scheduling` (wash)

| | C1 | C2 | C4 | C8 | C12 | C16 |
|---|---|---|---|---|---|---|
| control | 88.5 | 123.2 | 168.1 | 248.6 | 265.6 | 305.5 |
| async | 90.1 | 126.4 | 175.6 | 244.7 | 255.4 | 316.0 |
| Δ | +1.8% | +2.6% | +4.5% | −1.6% | −3.8% | +3.4% |

Hash hit, no memory change. Nothing outside the ±3% drift band; not adopted, not harmful. Could be stacked on v13
later if a multi-stream tail-latency reason appears.

## E1 / E1b — decode speed vs prompt depth (map, not a lever)

`depth_knee.py` v1 (`depth-E1-v12-depth.json`) is **invalid**: it assumed a prefix-cache hit on the second call and at
106K and 425K the "decode" number reconstructs exactly as prefill + decode (26 and 7 tok/s). Kept as the record of the
bug. v2 streams and times first→last token, cache-independent:

| prompt tokens | 6.5K | 53K | 106K | 212K | 425K |
|---|---|---|---|---|---|
| decode tok/s (C1, k=5) | 123.8 | 117.3 | 114.3 | 183.6* | 121.8 |
| cold TTFT | — | — | 5.6 s | — | 25.8 s |

**Decode is flat with depth on this lane: −7% from 6K to 425K.** Attention and the sparse indexer are not the
bottleneck at depth; MoE expert fetch over C2C is, all the way out. This is the fact that decides whether to chase the
`dsv41-optimized` attention megakernels (#56344) or the sparse-MQA indexer (#56254): for throughput on this box, no.
(*212K's 183 is a random-word-list artifact — the drafter ran long accepted streaks on a repetitive continuation, 51–60
stream chunks vs ~80 elsewhere. Not a real number. 0.1–0.5 s TTFTs at 6K/53K/212K are prefix-cache hits from the
warm-up call; 106K and 425K show true cold prefill, consistent with the 2026-09-10 numbers.)

## Files

- `knee-R3-*.json`, `agentfix-R3-*.json` — raw knee / fixture per boot (`CTRL-for-E2`, `E2-async`, `CTRL-for-E3`, `E3-ksched`, `CTRL-for-E3-p2`, `E3-ksched-p2`)
- `depth-E1-v12-depth.json` (invalid, see above), `depth-E1b-v12-depth-p2.json` (valid)
- `facts-R3-*.txt` — boot receipts (hash, configs loaded, KV, offload, parsed schedule)
- `campaign-round3-2026-09-14.log`, `ledger-round3.md` — the run as it happened
- `v13-container-args.txt` — `docker inspect` args of the promoted container
- `campaign_round3.sh`, `depth_knee.py` — runner and instrument (v2)
- `throughput.csv` — recipe-schema rows for v13

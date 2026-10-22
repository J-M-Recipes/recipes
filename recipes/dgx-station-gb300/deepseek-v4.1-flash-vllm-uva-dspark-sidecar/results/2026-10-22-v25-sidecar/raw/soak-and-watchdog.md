# Soak and watchdog receipt — v25 sidecar lane

The lane is guarded, not just measured. A sidecar timeout does not raise an error: the GB300 hook skips
that layer's cold-expert add and keeps streaming, so the exposure is wrong tokens. Two independent
mechanisms cover it:

1. **Spin, never skip.** `VLLM_EXP_PEER_SPINS=2000000000` (el8's default is 2e7). On expiry the hook
   increments `words[2]` and drops the cold add. At 2e9 a dead or paused sidecar *hangs* the lane instead
   of corrupting it.
2. **Fail-closed watchdog** (`scripts/sidecar_watchdog.py`, 1 s interval) trips on `words[2]` increasing,
   on the sidecar container dying, on completed-seq frozen while published is ahead for > 2 s, or on an
   unreadable shm buffer — and kills the server first, then the sidecar.

Production policy on this lane is **fail-stop**, not auto-rollback: on a trip the lane is killed and stays
down; the rollback path exists in the script (`--rollback`) and was fault-tested, but nobody auto-starts the
previous lane behind a wrong-token trip.

## Fault test (2026-10-22 05:4x CDT)

Paused the sidecar container mid-serve: watchdog tripped in ~3 s, the in-flight request was cut, the
previous lane (`dsv41-vllm-1017-v23-pinexact-RETIRED-REF`) bound in 260 s when started by hand. 0 timeouts
before and after. Receipts (box): `~/dsv41/m3/logs/pause-test-v25.log`, `respin-v25.log`,
`repromote-v25.log`; scripts `pause_test_v25.sh`, `respin_v25_spins.sh`, `test_watchdog_offline.sh`
(an offline harness that fabricates a frozen counter and asserts the trip).

## Live counter read (2026-10-22, lane up 3 h since the 06:02 CDT restart)

```json
{"server_up": true, "words": [4259400, 4259400, 0, 6751420, 21704576, 11655380096, 2429809],
 "timeouts_base": 0, "wait_ms_per_wait": 0.0048, "mode": "enforce", "trip": null, "stall_s": 2.0}
```

| field | meaning | value |
|---|---|---|
| `words[0]` published / `words[1]` completed | sidecar call sequence handshake | 4,259,400 / 4,259,400 (in lockstep) |
| `words[2]` | **timeouts** (cold contribution dropped ⇒ wrong tokens) | **0** |
| `words[3]` rows sent | cold-expert rows computed on the RTX | 6,751,420 |
| `words[4]` tokens seen | tokens whose routes crossed the sidecar | 21,704,576 |
| `words[5]/[6]` | GB300 wait ns / waits | 0.0048 ms per wait |
| watchdog trips | — | **0** |

## Soak contract (in progress)

- Start: 2026-10-22 06:02 CDT, after the ack-first sidecar fix; window 48 h.
- Sampled every 5 min from a second machine: lane answers `/v1/models`, watchdog status has `trip: null`,
  `words[2]` unchanged, sidecar container up, RTX memory.flat.
- Trip conditions and thresholds are the watchdog's (above), plus `words[2] > 0` at any sample.
- The recipe's `status` moves from `experimental` to `verified` only on a clean 48 h window; this bundle is
  otherwise complete.

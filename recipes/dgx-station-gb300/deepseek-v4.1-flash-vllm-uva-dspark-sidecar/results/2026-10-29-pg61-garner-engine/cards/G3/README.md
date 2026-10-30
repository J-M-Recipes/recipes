# Card G3 / G3b — #61161 port + fail-closed sidecar watchdog on the Garner policy tree (2026-10-29)

**Verdict: both PASS.** Pg61 = Pg + #61161 port + `PEER3_SPINS=1000000000000` + `sidecar_watchdog_p3` is the new
candidate. Not promoted. Lane dark 17:12–17:46 (G3) and 17:48–18:03 (G3b); h2-REF restored both times (smoke OK,
production watchdog armed, isolate off).

## #61161 exactness (GPU, `test_markov_61161.py`, synthetic V = 129,280, rank 256, S = 5, T = 1)
Reference = production's eager L3 `gumbel_sample` on the q-cache row the kernel wrote, `logits_cache_col = s`.

| kernel | patched, matches per step | stock, matches per step |
|---|---|---|
| w6 `libw6mk.so` (R = 1, 30 trials) | 30 / 30 / 30 / 30 / 30 | 30 / 0 / 0 / 0 / 0 |
| w5 Triton (R = 4, 30 trials) | 120 / 120 / 120 / 120 / 120 | 120 / 0 / 1 / 0 / 0 |

T = 0: identical token streams stock vs patched for both kernels. (First harness run read draft rows by request
state instead of batch row; logged as `U/unit-v1bad-*`, rerun in the same card.)

## Pg61 cells (watchdog in ENFORCE throughout: 0 trips)
| check | Pg61 | Pg (G1f/G2) |
|---|---|---|
| in-container hashes | rsu edc8f8f4, gumbel 6eac3e13, libw6mk b072ee09, PEER3_SPINS=1e12 | — |
| fixed steps/s, tok/s | 119.3, 648.0 | 119.4, 632.5 |
| GSM8K T = 1 / greedy (60 s) | 459.5 / 545.1 tok/s | 450.3 / 546.7 |
| 8 / 32 streams | 1,440 / 2,604 | 1,393 / 2,586 |
| C32 window | survived | survived |
| gauntlet | 12/12 | 12/12 |
| GSM8K-200 T = 0 | 192/200 | 193/200 |
| tool-eval hardmode | 89/100, 163/184, safety-capped (TC-51, 89, 91) | 91/100, 167/184, not capped |
| BFCL held-out | 1026/1311 = 78.26 % | 1026/1311 |

Tool-eval is T = 0 and the port cannot change T = 0 tokens (unit test), so 91 → 89 is run-to-run variance at C4 —
one run is not enough to rank Pg/Pg61/v27. Run it ≥ 2× on the promotion card.

## Hang test (G3b)
- G3's first injection (`ctrl=1` alone) was a no-op: his sidecar host uses ctrl=1 as its routine park and relaunched
  the v3 kernel (launch count 1,083 → 2,130 over the card). Streams continued correctly.
- G3b: 4 streams, `docker pause ds41-sidecar` at …461.378, ctrl=1 at …461.483.
  - **last token on all 4 streams …461.481** (0.10 s after the pause, before the ctrl write) — no tokens after the stall;
  - watchdog `TRIP v3 stalled: GB300 published tag 0xa257, RTX done 0x9217 … frozen for >2.0s` at 17:54:24 (~3 s);
  - ds41 killed (exited); GB300 memory back to 22 MiB; h2-REF restore normal (BOUND after 425 s).
- Clients saw the stream end with no `finish_reason` and no HTTP error (server killed mid-stream). Callers must treat a
  missing finish_reason as failure.
- Fix after G3b: `docker stop` blocks on a paused sidecar, so the watchdog's fail-closed line never logged before the
  runner killed it. Now logs the server kill first, unpauses, then stops the sidecar (13/13 tests still pass).

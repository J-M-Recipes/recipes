# PREREG — CARD R1: one dark window (FROZEN 2026-10-31 before the first boot)

Written 2026-10-31 by W3. Freeze this file before the first boot; changes after the first boot need a new card name.

## What the card is for
Release items 3 (final watchdog never fault-injected), 4 (rollback never run end-to-end), 5 (no per-layer cosine
gate on peer tier v3), plus the post-window re-verify. One dark window, Pg61 at both ends.

## Lane, arms, pins
- Lane `:30006`, Pg61 = containers `ds41` + `ds41-sidecar`, launcher `~/dsv41/garner/launcher/launch-pg61.sh`,
  build `~/dsv41/garner/pg61-work` (frozen, `BUILD-SHA256SUMS`).
- Rollback arm = `dsv41-vllm-1017-h2-REF` + `-rtx` (stopped-and-kept, restart policy `no`).
- Watchdog under test: `~/dsv41/garner/pg61-work/sidecar_watchdog_p3.py`
  sha256 `ba8373e9bdd9a0225620142ea93330a99caa597826910399ab4c741ee8eabc9f` (NOT the recipe's `c97d149d…`; the
  difference is two default-path lines — ship the tested bytes in the recipe).
- Tokenizer/corpus for the trust chain: `~/pin-hot-experts/tf_corpus.jsonl` (39 docs).

## Phases and dark time
P0 preflight (0 dark, ~5 min) · P1 peer3 cosine tap (0 dark, ~10 min) · P2 fault injection (3 min) ·
P3 restore boot #1 + bring-up gate repeat (7 min) · P4 rollback drill (8.5 min) · P5 cutover back (8.5 min) ·
P6 close (2 min). **Dark total ~30 min, 3 boots (2 Pg61, 1 h2-REF).** Hard bar 45 min.

## Bars (frozen)
A1 rollback exit 0 with `Pg61 down` → `rtx serving (fresh)` → `BOUND` → `smoke OK`, in order.
A2 both `/dev/shm/vllm_peer_tier{2,3}` absent when `docker start $RTX` is issued (100 ms sampler).
A3 exactly one watchdog: `comm==python3`, argv has `sidecar_watchdog.py` and `--server dsv41-vllm-1017-h2-REF`,
   PPID 1, status age < 5 s, `trip: null`. A WARN-from-the-script outcome is FAIL.
A4 h2-REF lane: BOUND, sidecar running, `words[2]==0`, still true after a 60 s hold.
A5 all four containers `RestartPolicy no`, `RestartCount 0` at every phase end.
A6 cutover back: exit 0, BUILD-SHA256SUMS ok, `gumbel 6eac3e13` / `rsu edc8f8f4` / `libw6mk b54168223847560b` /
   `PEER3_SPINS=1000000000000`, p3 `mode=enforce`, boot-log KV **11,696,582 tokens / 11.15x**, smoke OK,
   MAINTENANCE removed.
A7 exactly one TRIP in `wdp3.log` for the whole card; zero v2 timeouts.
A8 dark ≤ 45 min; Pg61 bind ≤ 760 s; h2-REF bind ≤ 850 s.
B1 TRIP − t_pause ≤ 6.0 s.
B2 `ds41 .State.Running == false` within 3 s of the TRIP line.
B3 `fail-closed: docker kill ds41 rc=` logged BEFORE `fail-closed: sidecar ds41-sidecar stop/kill rc=`, both ≤ 10 s
   after TRIP (the post-G3b fix; G3b never logged either line).
B4 all 4 streams `finish_reason == null`, `error == null`, all chunk timestamps ≤ t_pause + 0.5 s.
B5 zero client chunks after t_pause (the "no wrong tokens after the stall" claim, recorded separately).
B6 GB300 ≤ 4 GiB and RTX idle within 120 s; watchdog exits rc 2 and does not re-arm.
B7 sidecar stopped, not merely paused; no restart policy resurrects it.
C1 40/40 MoE layers; ≥ 4 samples and ≥ 48 cold (token,slot) pairs per layer.
C2 min per-layer cos ≥ 0.9990.
C3 max |delta| ≤ 2e-2 (Garner's v3-vs-fp64 dev assertion).
C4 zero sidecar timeouts during the tap.
C5 tap overhead ≤ 2% on p99 TTFT and median inter-token time vs the same-session untapped baseline.
C6 accepted-sample rate ≥ 30%.

## Abort rules
1. `ABORT` file → finish the atomic step, restore, `CARD R1 ABORTED`, exit 2. Never leave the lane dark or isolated.
2. Wedged GB300 (>= 90% util with no compute apps in 2 of 3 samples, or `nvidia-smi` hangs) → leave DARK, write
   "needs hands", skip restore, exit 3.
3. No TRIP in 60 s at P2 → unpause, wait 30 s, if still no trip: card FAIL, restore via down+up, exit 2.
4. Rollback smoke fails → immediately run the cutover to get back to Pg61; if that fails, MAINTENANCE + "needs hands",
   exit 4.
5. Either boot over 900 s → restore, abort.
6. `BUILD=BAD` at any point → hard abort (the lane is not the config being stamped).


## Amendments before the first boot (2026-10-31, Milo; all made before any R1 phase ran)
1. P1 (live cosine tap) moved OUT of the window (advisor consult-2): it ran on the serving lane before the card,
   receipts `~/dsv41/garner/r1/live-tap2/`. Inside the window only the P3 bring-up tap runs; its fp64 reference pass
   runs in P6 after the lane is restored. Tap stop condition fixed (it stopped at one sample per layer) and the
   coherence check now re-reads every record tag after the copies. The offline reference (`peer3_cos_ref.py`) is new:
   W3's ref pass was a placeholder.
2. P2 fault-injects the DEPLOYED watchdog process (the one `launch-pg61.sh up` armed); the draft armed a duplicate.
3. B4 amended: PASS = every stream ends with `finish_reason == null` (no stream claims a normal completion); client
   exceptions on the killed connection are recorded, not gated (an exception is a louder failure, not a safety loss).
   B5 (no chunk after t_pause + 0.5 s) is unchanged and is the "no wrong tokens after the stall" bar.
4. Isolation (non-loopback :30006 REJECT) from P2 to P6; MAINTENANCE kept through the h2-REF drill.
5. P4 adds the Card QT1 h2-REF reference arm if staged (budget 25 min, its own PREREG). Dark-time bar raised to
   75 min with QT1 (advisor: 3 boots at 380–425 s bind alone is ~20 min; plan 45 min + QT1).
6. Rollback/launcher scripts were hardened BEFORE the card (rollback 2c6c8ce4, launcher ff5573b5; `.orig-20261031`
   kept): fatal watchdog re-arm, shm/containers verified gone after `down`, `up` removes stale segments when nothing
   runs, `up` writes `soak-start.json`. A3/H2 are now script behaviour, and the bars still check them independently.

## Explicit non-goals
No config, flag, hash, KV or weight change. No `PEER3_FOLD` change. No temperature > 0 work (that is CARD L1).
No sidecar-kill fault (freeze only) — a 5-line extra phase if James asks.

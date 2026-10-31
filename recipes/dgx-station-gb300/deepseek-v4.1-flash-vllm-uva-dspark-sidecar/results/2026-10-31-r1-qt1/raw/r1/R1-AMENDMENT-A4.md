# R1 amendment A4 (2026-10-31 12:15 CDT, after R1 run 1 P2, before any re-run data)
Run 1 P2 result: B1 PASS (2.86 s), B2 PASS, B4 PASS (4/4 in flight, no finish_reason), B5 PASS (last chunk 0.004 s after
the pause). B3/B6/B7 FAIL. Cause, from the watchdog log and source (`fail_closed()` then `return 2`): the verdict was
taken 5 s after the TRIP, while the watchdog was inside `docker stop -t 5` of the unpaused sidecar. R1's `down` then
killed the watchdog at 12:06:28 before it logged the sidecar stop.
This is an instrument-timing defect in wd3_fault.sh, not a watchdog defect. It is not proven to be one until the
re-run passes.
Amendment: wd3_fault.sh waits for the watchdog process to exit (≤ 60 s) before collecting B2–B7. Bars are unchanged.
Re-run: r1b_fault_rerun.sh right after R1 closes (~9 min more dark; the soak restarts at its `up`).
The run-1 FAIL stays on file: results-R1-20261031-1205/P2/.

# E1 v2 offline repair package

This directory documents the v2 E1-only retry contract prepared after the v1 live window exposed launcher/environment/readiness/profiler-safety defects.

- `CONTRACT.md` defines the versioned v2 release, controlled environment, K=1-only scope, preflight/restore pinning, and fail-closed profiler evidence rules.
- Operator source is in `scripts/window_e1_v2.py` and existing recipe scripts; the runner refuses output collisions, uses a shared host operation lock, bounds and process-group-terminates subprocesses, launches from the archived source snapshot, preserves real Nsight report bytes, restores before offline analysis, and leaves v1 receipts/checksums historical.
- Behavioral coverage lives in `tests/test_e1_window_v2.py` and uses fake Docker/API/tool executables rather than source-string assertions.

No live Station action was performed by this package preparation.

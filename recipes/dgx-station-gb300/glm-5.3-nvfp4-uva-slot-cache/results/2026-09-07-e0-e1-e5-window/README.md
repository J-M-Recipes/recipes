# E0/E1/E5 window — prepared, not launched

Status: **READY FOR STATIC REVIEW; LIVE WINDOW NOT RELEASED.**

This package freezes the last bounded GLM-5.3 512K slot-cache campaign window:

- **E0:** correct live cache-hit telemetry;
- **E1:** freeze a matched patched-tree MTP K=1 comparator, profile graph-node work, and decide
  whether E4 gather compaction has at least 2 ms/step of directly measured opportunity;
- **E5:** screen MTP K=2 through serial acceptance, throughput, and primary-quality gates.

The already-published E2 cache-policy and E3-A prefetch screens both failed. Nothing here revives
those branches. DMA and DFlash2 also remain stopped.

## Preparation completed

- corrected telemetry uses an observed routed-expert counter instead of a fixed eight-route divisor;
- opt-in in-container Nsight Systems 2025.6.3 launch path;
- run-id-bound, atomic CUDA profiler START/STOP control;
- overlap-aware CUDA kernel bucketing;
- exact K=1 live-argument reproduction and K=2 one-field-delta test;
- fail-closed `CONTROL`/`RELEASE` gate;
- byte-identical prose/code benchmark fixtures copied from the measured DMA campaign.

A disposable, network-disabled, non-GPU container proved the required Nsight installation-tree and
symlink layout on the Station. The first direct-binary attempt failed safely and taught us that
Nsight must be invoked through a symlink while retaining its original absolute install tree. The
corrected smoke returned Nsight 2025.6.3. The incumbent container remained running with the same
start timestamp.

## Important methodology corrections

1. The live hook's printed `HIT=` assumes eight routes per engine step. MTP(1) verification normally
   routes 16. Corrected aggregate historical live hit is about 0.704, consistent with the 0.710
   offline replay; the old 0.6166 planning estimate is not a live measurement.
2. CUDA kernel-summary durations are aggregate GPU work. Concurrent streams overlap, so they are
   not forced to sum to request wall time. E1 separately reports overlap and residual time.
3. The unchanged live hook exposes aggregate/best/worst/median only. E0 does not promise or invent
   a complete per-layer table.
4. `masked_row_copy` is excluded from the E4-recoverable gate because it is the necessary weight
   transfer; E4 targets bookkeeping, scalar gathers, and their launch overhead.

## Release boundary

The Station staging root will start with `CONTROL=HOLD` and no `RELEASE`. No candidate may launch
until James separately authorizes the live window and these exact files exist:

```text
CONTROL: RUN
RELEASE: e0-e1-e5-20260907-v1
```

`CONTRACT.md` is authoritative. A preparation success is not a launch, promotion, or DSF restore.

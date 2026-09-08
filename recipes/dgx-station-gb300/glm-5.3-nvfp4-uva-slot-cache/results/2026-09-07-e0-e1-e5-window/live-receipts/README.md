# E0/E1/E5 live-window receipts

- Source commit: `3ba679371252a2a64a5579899f4d36d539b58ccb`
- E0 corrected live hit rate: `0.747622` across `37` windows
- E0 weighted K=1 acceptance length: `1.826940`
- E1 verdict: **INCONCLUSIVE**
  - candidate exited before readiness because the one-off operator exported COMPILATION_CONFIG into a frozen launcher expansion that appended a trailing brace
  - no matched patched-tree K=1 comparator or Nsight capture was collected
  - the frozen 120 x 5 second readiness budget is shorter than the observed cold-load path, which remained in checkpoint loading beyond 600 seconds
- E5: not run; blocked by E1 INCONCLUSIVE
- Restored container: `glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907`
- Restored image: `vllm-glm53-uva:v0.28.0-2cf0a691`
- Restore completion: `WINDOW_RESTORE_OK`

No Nsight profile or kernel-attribution receipt is claimed: E1 failed before readiness and capture. E0 telemetry uses the corrected speculative-verification denominator recorded in `e0/telemetry-summary.json`.

# E1 v2 masked_row_copy characterization

## Inputs / hashes

- `e1_cuda_gpu_kern_sum.csv`: `cd768d4db40374606b08db01968a5ff796d44467ed4a2ca9cf0c3793c4ddaab7`
- Station `e1_cuda_gpu_trace.csv`: `0ba12baaf22db45d3efa39fe4cf6f791faa6029e7c2f0e0936fc38266017bb6a`
- `executed-source/patches/slot_cache_hook.py`: `2e7fb1e5dd621eb465add5a5766fe01dd0a7bc2872ecdb70027b251a43cfdc46`
- `executed-source/configs/slots-5792-ctx512k.json`: `73962276498626af16baaa7c8737384960aff8718d254e66ab92610570c9743c`

## Summary-row measurement

`e1_cuda_gpu_kern_sum.csv` line 2 (`Name=masked_row_copy`) reports:

- `Total Time (ns)=2,813,695,348` = `20.097824` ms per 140-step probe.
- `Instances=42,900`; `Avg (ns)=65,587.3`; `Min (ns)=992`; `Max (ns)=519,168`.
- Share of aggregate GPU kernel time: `33.539%`.

Instance arithmetic:

- The prompt arithmetic is correct: `42,900 / 140 = 306.43` calls/verification-step, and `306.43 / 92 = 3.331` calls/layer averaged over all 92 model layers.
- The full time-ordered GPU trace contains `42,900 = 143 * 75 * 4` slot-cache copy calls. Treating the last 140 inferred slot-cache steps as the verification window gives `42,000 / 140 = 300.00` calls/step.
- Source/config explain the exact pattern: 75 cached layers (`slots-5792-ctx512k.json` lines 8-83, layers 3..77) × 4 `masked_row_copy` launches per cached layer (`slot_cache_hook.py` lines 305-310) = 300 calls/step. Averaged across all 92 layers, that is `300/92 = 3.261` calls/layer/step.

## Duration distribution from full GPU trace

All 42,900 `masked_row_copy` rows (Station trace):

| percentile | ns | us |
|---:|---:|---:|
| min | 992 | 0.992 |
| p1 | 1,408 | 1.408 |
| p5 | 7,104 | 7.104 |
| p10 | 8,736 | 8.736 |
| p25 | 14,400 | 14.400 |
| p50 | 34,144 | 34.144 |
| p75 | 101,056 | 101.056 |
| p90 | 173,632 | 173.632 |
| p95 | 214,496 | 214.496 |
| p99 | 323,616 | 323.616 |
| p99.9 | 433,152 | 433.152 |
| max | 519,168 | 519.168 |

Last 140 inferred slot-cache steps (`42,000` calls): mean per-call `64,660.984 ns`; total `2,715.761 ms`; mean per-step `19.398 ms`. Step totals: min `10.186 ms`, p50 `19.029 ms`, p95 `28.243 ms`, p99 `35.593 ms`, max `35.750 ms`.

Histogram for the last 140 inferred steps:

| duration bucket | calls | share |
|---|---:|---:|
| [0, 5us) | 846 | 2.01% |
| [5, 10us) | 5,103 | 12.15% |
| [10, 25us) | 12,958 | 30.85% |
| [25, 50us) | 5,665 | 13.49% |
| [50, 100us) | 6,713 | 15.98% |
| [100, 250us) | 9,510 | 22.64% |
| [250, 500us) | 1,201 | 2.86% |
| [500us, 1ms) | 4 | 0.01% |

Bimodality: not a clean two-mode distribution. The aggregate is multi-modal because each cached layer launches the fixed `GrdY` sequence `(768, 384, 96, 48)` with `GrdX=16`; durations scale with row size, while the 0.992us minimum and the large `<25us` population indicate many near-empty/no-copy calls (`mask[k] == 0`).

Per fixed launch in the last 140 inferred steps:

| inferred launch | GrdX | GrdY | n | mean us | p50 us | p95 us | max us |
|---|---:|---:|---:|---:|---:|---:|---:|
| w13 weights | 16 | 768 | 10,500 | 149.936 | 140.064 | 311.712 | 519.168 |
| w2 weights | 16 | 384 | 10,500 | 75.286 | 69.600 | 156.000 | 273.088 |
| w13 scales | 16 | 96 | 10,500 | 21.318 | 19.872 | 40.864 | 70.464 |
| w2 scales | 16 | 48 | 10,500 | 12.104 | 11.424 | 21.696 | 296.448 |

## Per-layer / time-order pattern

Measured/inferred from ordering, not explicit layer IDs: the trace rows have no layer column, so this maps every 75 groups to config layers 3..77 in ascending order. Source launch order is `w13`, `w2`, `w13_scale`, `w2_scale` (`slot_cache_hook.py` lines 305-310).

- Slowest mean layer-invocation sums: L4 `560.5us`, L3 `491.8us`, L5 `484.3us`, L6 `447.6us`, L7 `434.0us`.
- Fastest mean layer-invocation sums: L18 `171.1us`, L19 `193.0us`, L20 `197.0us`, L58 `199.4us`, L41 `200.0us`.
- Correlation (inferred): layer id vs mean time Pearson `-0.521`; configured slots vs mean time Pearson `0.441`. Early cached layers are generally slower; more configured slots weakly/moderately increases time.

## What the kernel does

Per `executed-source/patches/slot_cache_hook.py`:

- Lines 218-226 define `masked_row_copy(src_ptr, dst_ptr, src_idx_ptr, dst_idx_ptr, mask_ptr, row_elems, BLOCK)`.
- It uses a 2-D grid: `k = tl.program_id(0)` selects route/expert candidate; `blk = tl.program_id(1)` selects a row block (line 220).
- It reads `mask[k]`, `src_idx[k]`, and `dst_idx[k]` (lines 221-222), computes offsets `blk * BLOCK + arange(BLOCK)` (line 223), and copies `src[s * row_elems + off]` to `dst[d * row_elems + off]` only when `off < row_elems` and `mask[k] != 0` (lines 224-226).
- `_cache_forward` launches `fused_bookkeeping` first (lines 300-303), sets `BLOCK = 2048` (line 304), then launches `masked_row_copy` for `w13` and `w2` rows (lines 305-307) and resident scale rows (lines 308-310).
- `_rows64` converts each expert tensor row to an `int64` flat row (`slot_cache_hook.py` lines 276-280), so `row_elems` is an `int64` element count.
- The slot-cache path is taken only when `lc` exists, `M <= BYPASS_ABOVE`, and `M * topk_ids.shape[1] <= lc.S`; it sets `N = M * topk_ids.shape[1]`, flattens top-k IDs, calls `_cache_forward`, then invokes the slot-backed MoE (`slot_cache_hook.py` lines 391-398).

## Bytes per call (derivable bound)

The trace reports `GrdX=16` and fixed `GrdY=(768,384,96,48)`. With `BLOCK=2048` and `_rows64` int64 rows, one masked route (`mask[k] != 0`) copies:

| launch | row bytes per masked route | payload MiB | load+store MiB |
|---|---:|---:|---:|
| w13 | `768*2048*8` | 12.00 | 24.00 |
| w2 | `384*2048*8` | 6.00 | 12.00 |
| w13_scale | `96*2048*8` | 1.50 | 3.00 |
| w2_scale | `48*2048*8` | 0.75 | 1.50 |
| all four for one missed expert | — | 20.25 | 40.50 |

Per launch upper bound if all 16 `GrdX` route entries miss: 16× those numbers; all four launches would be 324 MiB payload / 648 MiB load+store per cached-layer invocation. Actual bytes are not derivable from the trace alone because it does not expose `mask` population per call.

# O1-b — single-VA bandwidth rescues (Track 1)

Window 1, 2026-09-17 ~12:35 CDT. Worker: **claude-fable-5.1 via nous**. GPU quiet (no containers, GB300 22 MiB). Container `pin-e1b-w1` (`--rm`, stopped after). Box driver **595.91.07, CUDA 13.2** (`cuDriverGetVersion` = 13020; spec said 595.84 / 13.0), 2 MiB granularity, `HOST_NUMA_ID` = 0. Script: `spikes/e1b_vmm_rescues.py`. Numbers + every `CUresult`: `results/track1/O1b-vmm-rescues.json`.

Reader: `torch.sum(dtype=fp32)` over a **1 GiB** range viewed as fp16 (one coalesced pass), 20 reps, median. Contended: **false**.

Device attributes: `PAGEABLE_MEMORY_ACCESS=1`, `..._USES_HOST_PAGE_TABLES=1` (ATS), `HANDLE_TYPE_FABRIC_SUPPORTED=1`, `POSIX_FD_SUPPORTED=1`, `VMM_SUPPORTED=1`, `HOST_REGISTER_SUPPORTED=1`, `HOST_ALLOC_DMA_BUF_SUPPORTED=0`.

## B0 — controls (reproduce: yes)

| control | GB/s |
|---|--:|
| HBM `torch.empty` | 5303 |
| torch `pin_memory()` read via host ptr / via `cudaHostGetDevicePointer` (same ptr) | 368 |
| `cudaHostAlloc(Mapped\|Portable)` via device ptr | **373** (expect ~340) |
| VMM `cuMemCreate(HOST_NUMA)` → `cuMemMap`, access DEVICE | **90.9** (expect ~91); second pass 90.9 |
| VMM `cuMemCreate(DEVICE)` (wrapper control) | 5320 |

## B1 — `CU_MEM_LOCATION_TYPE_HOST` and variants

| variant | CUresults | GB/s |
|---|---|--:|
| `cuMemCreate(HOST, id=0)` → map, access DEVICE | all SUCCESS | **90.9** |
| `cuMemCreate(HOST_NUMA_CURRENT)` | `cuMemCreate` → `CUDA_ERROR_INVALID_VALUE` | — |
| `HOST`, access DEVICE+HOST, CPU first-touch through the VA, then GPU read | all SUCCESS | 91.0 |
| `HOST_NUMA`, access DEVICE+HOST_NUMA, CPU first-touch, GPU read | all SUCCESS | 91.0 |

## B2 — shareable handles

| variant | CUresults | GB/s |
|---|---|--:|
| `cuMemCreate(HOST, POSIX_FD)` | `cuMemCreate` → `CUDA_ERROR_INVALID_VALUE` | — |
| `cuMemCreate(HOST, FABRIC)` | `cuMemCreate` → `CUDA_ERROR_INVALID_VALUE` | — |
| `cuMemCreate(HOST_NUMA, POSIX_FD)` → export fd → import → map → access DEVICE | all SUCCESS (fd 65) | **91.0** |
| `cuMemCreate(HOST_NUMA, FABRIC)` | `cuMemCreate` → **`CUDA_ERROR_NOT_PERMITTED`** | — |
| `cudaHostAlloc` ptr → `cuMemRetainAllocationHandle` (only route from a cudaHostAlloc buffer to a mappable handle) | `CUDA_ERROR_INVALID_VALUE` | — |
| `cudaHostAlloc` ptr → `cuMemGetHandleForAddressRange(DMA_BUF)` | `CUDA_ERROR_INVALID_VALUE` (device reports DMA_BUF unsupported) | — |

The driver refuses fabric handles on host memory and gives no handle for `cudaHostAlloc` memory, so the 340 GB/s pages cannot be re-mapped into a VMM range at all. The one handle that works (POSIX fd on HOST_NUMA) maps back to the same 91 GB/s pages.

## B3 — `cudaHostRegister` over the VMM-mapped host range

| range | `cudaHostRegister(Mapped\|Portable)` | `cudaHostRegister(Default)` |
|---|---|---|
| VMM HOST, access DEVICE+HOST | `cudaErrorInvalidValue` | `cudaErrorInvalidValue` |
| VMM HOST_NUMA, access DEVICE+HOST_NUMA | `cudaErrorInvalidValue` | `cudaErrorInvalidValue` |
| VMM HOST, access DEVICE only | `cudaErrorInvalidValue` | `cudaErrorInvalidValue` |

No read possible.

## Extra ceiling rows (same run)

| path | GB/s |
|---|--:|
| `malloc` + `cudaHostRegister(Mapped\|Portable)`, dev ptr (== host ptr) | 351 |
| system `malloc`, unregistered, read via ATS | 342 |

Every host-memory path that goes through the **host page tables (ATS)** — pinned, registered, or plain pageable — reads at 340–373 GB/s. Every path that goes through the **GPU page table via VMM `cuMemMap`** reads at 91 GB/s, regardless of location type, access flags, CPU first-touch, or handle export/import. The gap is the mapping mechanism, not the allocation.

## O1-b VERDICT: DEAD (best 91 GB/s)

Best mappable variant: 91.0 GB/s (`HOST` or `HOST_NUMA`, any access set, any handle type the driver accepts). Fabric/host: refused (`CUDA_ERROR_NOT_PERMITTED`). `cudaHostRegister` over VMM ranges: refused (`cudaErrorInvalidValue`). Nothing reached 150, let alone 250. The single-VA mixed-backing design cannot reach C2C speed on this driver; per TRACK1 §O1-b, closed for good. Follow-up owed (not done by this worker: skill writes stage under write-approval): note in skill `dsv41-gb300-serving` that the HOST / FABRIC / hostRegister rescues were tried on 595.91.07 and all land at 91 GB/s or are refused.

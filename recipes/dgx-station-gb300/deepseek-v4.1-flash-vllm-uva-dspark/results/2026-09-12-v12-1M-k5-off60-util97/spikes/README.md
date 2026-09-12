# Residency spikes — why the offload lever stops at "how many GiB", 2026-09-12

Run inside `vllm/vllm-openai:deepseekv41-flash-0909` on the GB300 (driver 595.84, CUDA 13.0) with the v12 reference serving beside them:

    docker run --rm --gpus all --ipc host --ulimit memlock=-1 --cap-add IPC_LOCK -v <workdir>:/w --entrypoint python3 vllm/vllm-openai:deepseekv41-flash-0909 /w/<script>.py

| Script | Question | Answer |
|---|---|---|
| `spike_vmm_egm.py` | Can one VA range hold expert rows backed by HBM *or* Grace (`cuMemCreate HOST_NUMA` + `cuMemMap`), wrapped as a torch tensor, remapped in place? | Yes, mechanically (6 steps). |
| `spike_vmm_step5.py`, `spike_vmm_step5b.py` | How fast does the GPU read the Grace-backed rows? | **~90 GB/s** vs **~350 GB/s** for `cudaHostAlloc` pinned (what the UVA offloader uses) and 4.2 TB/s HBM. |
| `spike_managed.py` | Does `cudaMallocManaged` give an adaptive hot/cold split for free? | No. Plain managed migrates the whole range to HBM on first GPU fault; with CPU-preference advice it reads at 90 GB/s; oversubscribed (working set > free HBM) it settles at **155 GB/s** — worse than pinned. |

Read: two GPU→Grace paths exist — ATS (pinned/registered, ~340 GB/s, no per-page placement) and GPU-page-table-mapped host memory (~90 GB/s, placement possible). Nothing gives cold-in-Grace at full speed under one pointer. The residency axis for this recipe closes at `--cpu-offload-gb 60`.

Raw output: `spike-*.json`.

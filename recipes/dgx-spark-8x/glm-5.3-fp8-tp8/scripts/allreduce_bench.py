"""GhettoRak NCCL all_reduce benchmark (1 rank per GB10 node).

Spec (Fable 5.1 review, PHASE1 runbook §3): float32; sizes 8M..1G bytes;
5 warmup + 20 timed; CUDA-event timing; MAX elapsed across ranks;
busbw = 2(n-1)/n * bytes / t. No NCCL_ALGO/PROTO overrides.
Launch: env:// (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE), one container per node.
"""
import torch
import torch.distributed as dist

SIZES = [8 << 20, 32 << 20, 128 << 20, 512 << 20, 1 << 30]
WARMUP, ITERS = 5, 20


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(0)

    # correctness: sum of ones == world size
    c = torch.ones(1024, dtype=torch.float32, device="cuda")
    dist.all_reduce(c)
    ok = bool(torch.all(c == ws).item())
    if rank == 0:
        v = torch.cuda.nccl.version()
        print(f"world={ws} torch={torch.__version__} nccl={'.'.join(map(str, v)) if isinstance(v, tuple) else v} correctness={'OK' if ok else 'FAIL'}", flush=True)

    for b in SIZES:
        x = torch.zeros(b // 4, dtype=torch.float32, device="cuda")
        for _ in range(WARMUP):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            dist.all_reduce(x)
        e.record()
        torch.cuda.synchronize()
        t = torch.tensor([s.elapsed_time(e) / 1000.0 / ITERS], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        t = t.item()
        alg = b / t / 1e9
        bus = alg * 2 * (ws - 1) / ws
        if rank == 0:
            print(f"size={b / 2**20:6.0f}MiB  time={t * 1e3:8.3f}ms  algbw={alg:6.2f}GB/s  busbw={bus:6.2f}GB/s", flush=True)
        del x
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# small-message all_reduce latency (decode-shaped): bf16, 12K..384K bytes, 50 warm + 500 timed, CUDA graphs off.
import os, torch, torch.distributed as dist
dist.init_process_group("nccl"); r=dist.get_rank(); torch.cuda.set_device(0)
for b in [12288, 36864, 73728, 196608, 393216]:
    x=torch.zeros(b//2,dtype=torch.bfloat16,device="cuda")
    for _ in range(50): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(500): dist.all_reduce(x)
    e.record(); torch.cuda.synchronize()
    t=torch.tensor([s.elapsed_time(e)/500*1000],device="cuda"); dist.all_reduce(t,op=dist.ReduceOp.MAX)
    if r==0: print(f"{os.environ.get('TAG')} bytes={b} us={t.item():.1f}",flush=True)

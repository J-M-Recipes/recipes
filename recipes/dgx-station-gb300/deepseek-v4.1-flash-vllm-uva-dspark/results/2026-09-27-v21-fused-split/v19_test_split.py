"""Bit-exactness test: fused Triton split_ids vs the torch reference, on random ids incl. -1 padding,
plus a launch-count/time comparison at T=6,K=6 (decode) and T=96 (graph max)."""
import os, sys, importlib.util, time, torch
os.environ.setdefault("PIN_MODE", "split")
spec = importlib.util.spec_from_file_location("hook", sys.argv[1])
hook = importlib.util.module_from_spec(spec)
sys.modules["hook"] = hook
# avoid install() side effects: the module runs install() only under sitecustomize; importing is safe
spec.loader.exec_module(hook)
E = hook.E
g = torch.Generator(device="cuda").manual_seed(0)
# row map like the real one: 295 hot (>=0 local idx), 89 cold (-(local+1))
perm = torch.randperm(E, generator=g, device="cuda")
rm = torch.empty(E, dtype=torch.int32, device="cuda")
rm[perm[:295]] = torch.arange(295, dtype=torch.int32, device="cuda")
rm[perm[295:]] = -(torch.arange(89, dtype=torch.int32, device="cuda") + 1)
ok = True
for T in (1, 6, 16, 96, 8192):
    ids = torch.randint(0, E, (T, 6), generator=g, device="cuda", dtype=torch.int32)
    mask = torch.rand(T, 6, generator=g, device="cuda") < 0.1
    ids[mask] = -1
    h1, c1 = hook.split_ids_torch(ids, rm)
    h2, c2 = hook._get_split()(ids, rm)
    same = torch.equal(h1, h2) and torch.equal(c1, c2) and h2.dtype == torch.int32
    ok &= same
    print(f"T={T:5d} equal={same} hot_valid={int((h2>=0).sum())} cold_valid={int((c2>=0).sum())} pad={int(mask.sum())}")
assert ok, "MISMATCH"
# timing
for T in (6, 16):
    ids = torch.randint(0, E, (T, 6), generator=g, device="cuda", dtype=torch.int32)
    for name, fn in (("torch6op", hook.split_ids_torch), ("fused", hook._get_split())):
        for _ in range(20): fn(ids, rm)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(500): fn(ids, rm)
        torch.cuda.synchronize()
        print(f"T={T} {name}: {(time.perf_counter()-t0)/500*1e6:.1f} us/call (eager, incl. launch)")
print("PASS")

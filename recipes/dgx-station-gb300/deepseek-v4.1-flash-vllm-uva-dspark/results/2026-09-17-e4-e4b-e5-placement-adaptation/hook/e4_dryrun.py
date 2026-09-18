#!/usr/bin/env python3
"""E4 off-lane dry-run: F4 split + 5 adaptive swaps vs full-E, plus a
deliberately mis-ordered map flip that MUST diverge.

Runs inside vllm/vllm-openai:deepseekv41-flash-0909. GPU must be quiet.
Cap 0.25; abort if free < 4 GiB. Never boots the model.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback

import torch

MEM_FRACTION = float(os.environ.get("E4_MEM_FRACTION", os.environ.get("E2B_MEM_FRACTION", "0.25")))
MIN_FREE = 4 * (1024**3)
SEED = 20260917
E = 384
N_HOT = 295
N_COLD = 89
HIDDEN = 5120
INTER = 2304
TOPK = 6
TS = (1, 6, 24, 96)
OUT_PATH = os.environ.get("E4_DRY_OUT", "/w/e4/dryrun.json")
ROWMAP = os.environ.get("PIN_ROWMAP", "/w/hook/rowmap-static-v1.json")
HOOK = os.environ.get("PIN_HOOK", "/w/hook/pin_hot_experts_hook.py")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def abort(reason, extra=None):
    rec = {"aborted": True, "reason": reason, "ts": _now()}
    if extra:
        rec.update(extra)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(rec, open(OUT_PATH, "w"), indent=2, default=str)
    sys.stderr.write(f"ABORT {reason}\n")
    sys.exit(2)


# Safety first.
_dev_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
GB300 = next((i for i, n in enumerate(_dev_names) if "GB300" in n), 0)
torch.cuda.set_device(GB300)
torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, GB300)
free_b, total_b = torch.cuda.mem_get_info(GB300)
if free_b < MIN_FREE:
    abort("gpu_free_below_4GiB", {"free_bytes": free_b, "total_bytes": total_b})

device = torch.device("cuda", GB300)
pin = load_mod(HOOK, "pin_hot_experts_hook")

from flashinfer.fused_moe.core import ActivationType  # noqa: E402
from flashinfer.fp4_quantization import nvfp4_block_scale_interleave  # noqa: E402
from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache  # noqa: E402

ACT = ActivationType.Swiglu


def _as_tensor(out):
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def trtllm_shuffle(w13, w2, s13, s2):
    cache: dict = {}
    num_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2
    hidden_size = w13.shape[2] * 2
    sf_block_size = 32
    epilogue_tile_m = 128
    w13 = w13.contiguous()
    w2 = w2.contiguous()
    s13 = s13.contiguous()
    s2 = s2.contiguous()
    w1_w = w13[:, :intermediate_size, :]
    w3_w = w13[:, intermediate_size:, :]
    w13 = torch.stack([w3_w, w1_w], dim=2).reshape(w13.shape)
    w1_s = s13[:, :intermediate_size, :]
    w3_s = s13[:, intermediate_size:, :]
    s13 = torch.stack([w3_s, w1_s], dim=2).reshape(s13.shape)
    w13_perm = get_w2_permute_indices_with_cache(
        cache, w13[0].view(torch.uint8), epilogue_tile_m
    ).to(w13.device)
    w13 = w13.view(torch.uint8)[:, w13_perm].contiguous()
    w13_sf_perm = get_w2_permute_indices_with_cache(
        cache, s13[0].view(torch.uint8), epilogue_tile_m, num_elts_per_sf=16
    ).to(s13.device)
    w13_s = s13.view(torch.uint8)[:, w13_sf_perm].contiguous()
    E_, N_s, K_s = w13_s.shape
    s13 = (
        nvfp4_block_scale_interleave(w13_s.reshape(E_ * N_s, K_s))
        .reshape(num_experts, 2 * intermediate_size, hidden_size // sf_block_size)
        .view(torch.float8_e4m3fn)
    )
    w2_perm = get_w2_permute_indices_with_cache(
        cache, w2[0].view(torch.uint8), epilogue_tile_m
    ).to(w2.device)
    w2 = w2.view(torch.uint8)[:, w2_perm].contiguous()
    w2_sf_perm = get_w2_permute_indices_with_cache(
        cache, s2[0].view(torch.uint8), epilogue_tile_m, num_elts_per_sf=16
    ).to(s2.device)
    w2_s = s2.view(torch.uint8)[:, w2_sf_perm].contiguous()
    E2, N2_s, K2_s = w2_s.shape
    s2 = (
        nvfp4_block_scale_interleave(w2_s.reshape(E2 * N2_s, K2_s))
        .reshape(num_experts, hidden_size, intermediate_size // sf_block_size)
        .view(torch.float8_e4m3fn)
    )
    return w13, w2, s13, s2


def make_logical(e: int, device, rng: torch.Generator, zero: bool = False):
    w13 = torch.zeros((e, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device=device)
    w2 = torch.zeros((e, HIDDEN, INTER // 2), dtype=torch.uint8, device=device)
    s13 = torch.full((e, 2 * INTER, HIDDEN // 32), 127, dtype=torch.uint8, device=device)
    s2 = torch.full((e, HIDDEN, INTER // 32), 127, dtype=torch.uint8, device=device)
    if not zero:
        w13.random_(0, 256, generator=rng)
        w2.random_(0, 256, generator=rng)
        s13.random_(119, 136, generator=rng)
        s2.random_(119, 136, generator=rng)
    return w13, w2, s13, s2


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    denom = b32.abs().clamp_min(1e-8)
    bit = (a.view(torch.uint16) == b.view(torch.uint16)) if a.dtype == torch.bfloat16 else (a == b)
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    n = int(a.numel())
    n_bit = int(bit.sum().item()) if n else 0
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "n": n,
        "n_bit_identical": n_bit,
        "frac_bit_identical": (n_bit / n) if n else 1.0,
        "mean_abs": float(diff.mean().item()) if n else 0.0,
    }
R = {
    "started": _now(),
    "gpu": _dev_names,
    "device": GB300,
    "mem_fraction": MEM_FRACTION,
    "free_giB_before": round(free_b / 1024**3, 3),
    "tests": {},
}


def save():
    R["gpu_now"] = {
        "free_giB": round(torch.cuda.mem_get_info(GB300)[0] / 1024**3, 3),
        "proc_alloc_giB": round(torch.cuda.memory_allocated(GB300) / 1024**3, 3),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(R, open(OUT_PATH, "w"), indent=2, default=str)


def log(m):
    sys.stderr.write(f"E4_DRY {m}\n")
    sys.stderr.flush()


def cmp(a, b):
    return compare(a, b)


# ---- build one real-geometry layer ----
rng = torch.Generator(device=device)
rng.manual_seed(SEED)
torch.manual_seed(SEED)

log("building E=384 shuffled layer")
w13, w2, s13, s2 = make_logical(E, device, rng, zero=False)
w1, w2, s1, s2 = trtllm_shuffle(w13, w2, s13, s2)
log(f"shuffled w1={tuple(w1.shape)} {w1.dtype} w2={tuple(w2.shape)}")

rowmap = pin.load_rowmap(ROWMAP)
hot_ids = rowmap[0]["hot"]
cold_ids = rowmap[0]["cold"]
assert len(hot_ids) == N_HOT and len(cold_ids) == N_COLD

# Simulate a RoutedExperts-like holder so rehome_one_layer can run.
class FakeRouted(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_name = "dry-layer0"
        self.w13_weight = torch.nn.Parameter(w1, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(s1, requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)


class FakeExperts:
    quant_config = type("QC", (), {})()
    gemm1_alpha = None
    gemm1_beta = None
    gemm1_clamp_limit = None


routed = FakeRouted()
experts = FakeExperts()

# Keep a full-E HBM clone as the reference source (rehome replaces params).
w1_ref = w1.clone()
w2_ref = w2.clone()
s1_ref = s1.clone()
s2_ref = s2.clone()

log("rehome_one_layer (HBM source → 295 HBM + 89 pinned UVA)")
st = pin.rehome_one_layer(routed, experts, 0, hot_ids, cold_ids)
R["rehome"] = {
    "uva_flag": st.is_uva,
    "hot_w1": list(st.hot["w1"].shape),
    "cold_w1": list(st.cold["w1"].shape),
    "hot_device": str(st.hot["w1"].device),
    "cold_is_cuda": bool(st.cold["w1"].is_cuda),
    "row_map_unique": int(st.row_map.unique().numel()),
    "hbm_free_giB": round(pin.hbm_free_bytes() / 1024**3, 3),
    "host_avail_giB": round(pin.mem_available_bytes() / 1024**3, 3),
}
save()
log(f"rehome ok {R['rehome']}")

# Also rehome a UVA-sourced copy of a tiny probe? Full E=384 UVA would double host.
# Instead: mark-path unit — pin a 2-row tensor, view, drop original, check keep.
log("UVA path smoke (2-row)")
try:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    probe = torch.arange(16, device=device, dtype=torch.uint8).view(2, 8)
    cpu = probe.detach().to("cpu").pin_memory()
    view = get_accelerator_view_from_cpu_tensor(cpu)
    view._vllm_is_uva_offloaded = True
    idx = torch.tensor([1], device=view.device)
    got = view.index_select(0, idx)
    R["uva_smoke"] = {
        "view_cuda": bool(view.is_cuda),
        "copy_ok": bool(torch.equal(got.cpu(), probe[1:].cpu())),
    }
except Exception as e:
    R["uva_smoke"] = {"error": f"{type(e).__name__}: {e}"}
save()


def make_case(T, seed=SEED):
    g = torch.Generator(device=device)
    g.manual_seed(seed + T)
    hidden = torch.randn((T, HIDDEN), dtype=torch.bfloat16, device=device, generator=g)
    ids = torch.randint(0, E, (T, TOPK), device=device, generator=g).to(torch.int32)
    wts = torch.rand((T, TOPK), dtype=torch.float32, device=device, generator=g)
    wts = wts / wts.sum(dim=-1, keepdim=True)
    return hidden, ids, wts


def call_full(hidden, ids, wts, do_finalize=True):
    return pin.call_routed(
        x_quant=hidden,
        x_scale=None,
        ids=ids,
        wts=wts,
        w1=w1_ref,
        s1=s1_ref,
        b1=None,
        a1=None,
        beta1=None,
        clamp1=None,
        w2=w2_ref,
        s2=s2_ref,
        b2=None,
        n_exp=E,
        topk=TOPK,
        activation_type=ACT,
        tune_max=max(128, hidden.shape[0]),
        enable_pdl=None,
        do_finalize=do_finalize,
        output=None,
        intermediate_size=INTER,
    )


def call_split(hidden, ids, wts):
    out = torch.empty((hidden.shape[0], HIDDEN), dtype=torch.bfloat16, device=device)
    return pin.invoke_f4_from_state(
        st,
        out,
        hidden,
        None,
        ids,
        wts,
        ACT,
        TOPK,
        max(128, hidden.shape[0]),
        intermediate_size=INTER,
    )


# ---- control (i): full-E unfinalized + our finalize vs kernel finalize ----
log("control (i) full-E unfinalized + triton fma vs kernel finalize")
ctrl = {}
for T in TS:
    hidden, ids, wts = make_case(T)
    ref = _as_tensor(call_full(hidden, ids, wts, do_finalize=True)).clone()
    unf = call_full(hidden, ids, wts, do_finalize=False)
    g2, e2p = pin._unfinalized_parts(unf, T, TOPK, HIDDEN)
    # Both partitions = full-E call: feed same g2 as hot, empty cold (all e2p_cold=-1)
    e2p_c = torch.full_like(e2p, -1)
    out = torch.empty_like(ref)
    pin.finalize_f4(g2, e2p, g2, e2p_c, wts, out)
    d = cmp(out, ref)
    ctrl[str(T)] = d
    log(f"control i T={T} bit={d['frac_bit_identical']:.6f} max_abs={d['max_abs']:.3e}")
R["tests"]["finalize_vs_kernel"] = ctrl
save()

# ---- F4 split vs full-E after re-home ----
log("F4 split vs full-E")
split = {}
for T in TS:
    hidden, ids, wts = make_case(T)
    ref = _as_tensor(call_full(hidden, ids, wts, do_finalize=True)).clone()
    out = call_split(hidden, ids, wts)
    d = cmp(out, ref)
    split[str(T)] = d
    log(f"split T={T} bit={d['frac_bit_identical']:.6f} max_abs={d['max_abs']:.3e} n={d['n']}")
R["tests"]["split_vs_fullE"] = split
save()

# ---- graph capture of split at T=6 ----
log("cuda graph capture of F4 split T=6")
try:
    hidden, ids, wts = make_case(6)
    out = torch.empty((6, HIDDEN), dtype=torch.bfloat16, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            pin.invoke_f4_from_state(
                st, out, hidden, None, ids, wts, ACT, TOPK, 128, intermediate_size=INTER
            )
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            pin.invoke_f4_from_state(
                st, out, hidden, None, ids, wts, ACT, TOPK, 128, intermediate_size=INTER
            )
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    ref = _as_tensor(call_full(hidden, ids, wts, do_finalize=True)).clone()
    d = cmp(out, ref)
    R["tests"]["graph_T6"] = d
    log(f"graph T=6 bit={d['frac_bit_identical']:.6f}")
except Exception as e:
    R["tests"]["graph_T6"] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}
    log(f"graph error {e}")
save()

# ---- adaptive: 5 swaps through the exact swap path, bit-id vs full-E ----
log("alloc staging + 5 ordered swaps")
staging = pin.alloc_staging(st)
R["staging_miB"] = round(pin._staging_bytes / 1024**2, 4)
swap_rows = []
# 5 distinct (hot, cold) pairs from layer-0 rowmap
pairs = list(zip(hot_ids[:5], cold_ids[:5]))
Tswap = 6
for i, (h_eid, c_eid) in enumerate(pairs):
    hidden, ids, wts = make_case(Tswap, seed=SEED + 1000 + i)
    # force both experts into the token so the map is actually used
    ids = ids.clone()
    ids[:, 0] = h_eid
    ids[:, 1] = c_eid
    ref = _as_tensor(call_full(hidden, ids, wts, do_finalize=True)).clone()
    before = call_split(hidden, ids, wts).clone()
    d_before = cmp(before, ref)
    rec = pin.swap_pair(st, h_eid, c_eid, staging=staging, misordered=False)
    torch.cuda.synchronize()
    after = call_split(hidden, ids, wts)
    d_after = cmp(after, ref)
    swap_rows.append(
        {
            "i": i,
            "hot_eid": h_eid,
            "cold_eid": c_eid,
            "hot_row": rec["hot_row"],
            "cold_row": rec["cold_row"],
            "before": d_before,
            "after": d_after,
        }
    )
    log(
        f"swap {i} {h_eid}<->{c_eid} before_bit={d_before['frac_bit_identical']:.6f} "
        f"after_bit={d_after['frac_bit_identical']:.6f}"
    )
R["tests"]["swaps_ordered"] = swap_rows
save()

# ---- mis-ordered flip MUST diverge ----
log("misordered flip (must DIVERGE)")
h_eid, c_eid = hot_ids[10], cold_ids[10]
hidden, ids, wts = make_case(Tswap, seed=SEED + 4242)
ids = ids.clone()
ids[:, 0] = h_eid
ids[:, 1] = c_eid
ref = _as_tensor(call_full(hidden, ids, wts, do_finalize=True)).clone()
# snapshot map to restore isn't needed; this is the last numeric check
pin.swap_pair(st, h_eid, c_eid, staging=staging, misordered=True)
torch.cuda.synchronize()
bad = call_split(hidden, ids, wts)
d_bad = cmp(bad, ref)
R["tests"]["misordered_flip"] = d_bad
log(
    f"misordered bit={d_bad['frac_bit_identical']:.6f} max_abs={d_bad['max_abs']:.3e} "
    f"(expect <1.0)"
)
save()

all_bits = []
for section in ("finalize_vs_kernel", "split_vs_fullE"):
    for T, d in R["tests"][section].items():
        all_bits.append(d.get("frac_bit_identical", 0))
graph = R["tests"].get("graph_T6", {})
if "frac_bit_identical" in graph:
    all_bits.append(graph["frac_bit_identical"])
for rec in swap_rows:
    all_bits.append(rec["before"].get("frac_bit_identical", 0))
    all_bits.append(rec["after"].get("frac_bit_identical", 0))

ordered_ok = bool(all_bits) and all(abs(x - 1.0) < 1e-12 for x in all_bits)
diverged = d_bad.get("frac_bit_identical", 1.0) < 1.0 - 1e-12
ok = ordered_ok and diverged
R["verdict"] = "PASS" if ok else "FAIL"
R["min_bit"] = min(all_bits) if all_bits else 0
R["misordered_diverged"] = diverged
R["ordered_ok"] = ordered_ok
R["finished"] = _now()
R["ordering_note"] = (
    "bit-exact-safe: hot-row -> HBM staging, cold-row -> hot slot, staging -> cold slot, "
    "then 2-element write to device int32[384] row_map. Map ADDRESS is fixed (CUDA graph); "
    "contents change between steps. Swaps enqueue on the main stream between steps "
    "(GPUModelRunner.execute_model / layer-0 eager _invoke). Never flip a layer mid-copy. "
    "misordered=True flips the map BEFORE the copies — dry-run asserts that diverges."
)
save()
log(
    f"VERDICT {R['verdict']} min_bit={R['min_bit']} "
    f"ordered_ok={ordered_ok} misordered_diverged={diverged}"
)
print(
    json.dumps(
        {
            "verdict": R["verdict"],
            "min_bit": R["min_bit"],
            "ordered_ok": ordered_ok,
            "misordered_diverged": diverged,
            "misordered_bit": d_bad.get("frac_bit_identical"),
            "out": OUT_PATH,
        }
    )
)
sys.exit(0 if ok else 1)

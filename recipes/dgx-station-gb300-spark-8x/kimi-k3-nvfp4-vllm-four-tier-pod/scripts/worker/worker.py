#!/usr/bin/env python3
"""K3 Spark expert worker v1 (A3).

Holds one staged bank (all 92 MoE layers x 62 experts, 98.7 GiB) resident on the GB10, listens for one
RC connection from the Station, and answers dispatch requests: 16 global expert ids in, the weighted sum
of the ones this Spark holds out, plus k_used / ids_used so the Station can assert the partition.

Zero-copy: the RDMA ring is cudaHostRegister'ed (GB10 is unified memory), so the kernel reads x straight
out of the receive slot and y is written straight into the reply slot — no pinned staging, no memmove.

Env: BANK, PLAN, TARGET, BIND_IP, PORT, DEV (rocep1s0f0), GID (3), LAYERS (default all), LOG_EVERY.
Runs in nvcr.io/nvidia/pytorch:26.07-py3 with /w = ~/k3pod, /bank = ~/k3/banks, /a3 = ~/k3pod/a3.
"""
from __future__ import annotations

import bisect, ctypes, hashlib, json, mmap, os, struct, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k3proto import (HDR, HIDDEN, MSG_BYTES, REP, REQ, SHUTDOWN_ID, ST_BAD, ST_EMPTY, ST_NONFINITE,  # noqa
                     ST_OK, TOPK, Conn, FLAG_PROBE, FLAG_BATCH, BATCH_M, BATCH_M_OFF, BATCH_MAX_M, X_BYTES,
                     batch_xoff, batch_req_bytes, batch_yoff, batch_rep_bytes, lib as _k3lib)

INTER, REC, HEADER = 3072, 18_579_480, 4096
E = 62          # experts per layer in this bank; overwritten from the bank header at boot
OFF = {"w13": 0, "w2": 11_010_048, "sf13": 16_515_072, "sf2": 17_891_328, "sc": 18_579_456}
NSLOTS, SLOT = 4, 8 << 20          # must match K3_NSLOTS / K3_SLOT in k3hop.c (checked at boot)
# K3-4 batched path: pair counts are padded up to one of these so the B12x static kernels (compiled per exact row
# count, ~1.5-3 s each) are all built at boot; >= 1024 pairs use the dynamic kernel, which takes any count.
BATCH_BUCKETS = (32, 64, 96, 128, 192, 256, 320, 384, 512, 640, 768, 896, 1024)   # 1024 = static cutover; above it every count runs the dynamic kernel
BATCH_CAP = BATCH_MAX_M * TOPK


class CAI:
    """Minimal __cuda_array_interface__ so torch can wrap host-registered (device-visible) memory."""
    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {"data": (ptr, False), "shape": shape, "typestr": typestr,
                                         "version": 3, "strides": None}


def dev_tensor(ptr, shape, typestr, dtype=None):
    t = torch.as_tensor(CAI(ptr, shape, typestr), device="cuda")
    return t.view(dtype) if dtype is not None else t


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_bank(path, n_layers):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import convert_sf_to_mma_layout
    fd = os.open(path, os.O_RDONLY)
    head = os.pread(fd, HEADER, 0)
    assert head[:8] == b"K3BANK\x01\x00", head[:8]
    ver, rec, e_hdr, nl = struct.unpack("<IIII", head[8:24])
    assert (rec, e_hdr) == (REC, E) and nl >= n_layers, (rec, e_hdr, nl)
    plan_sha = head[44:76].hex()
    stage = torch.empty(E * REC, dtype=torch.uint8, device="cuda")
    host = torch.empty(E * REC, dtype=torch.uint8, pin_memory=True)
    hv = memoryview(host.numpy())
    layers = []
    t0 = time.perf_counter()
    for L in range(n_layers):
        off = HEADER + L * E * REC
        got = 0
        while got < E * REC:
            got += os.preadv(fd, [hv[got:]], off + got)
        os.posix_fadvise(fd, off, E * REC, os.POSIX_FADV_DONTNEED)
        stage.copy_(host)

        def view(o, shape):
            st = (REC,) + tuple(int(np.prod(shape[i + 1:])) for i in range(len(shape)))
            return torch.as_strided(stage, (E,) + shape, st, o)
        w13 = view(OFF["w13"], (2 * INTER, HIDDEN // 2)).contiguous()
        w2 = view(OFF["w2"], (HIDDEN, INTER // 2)).contiguous()
        s13 = view(OFF["sf13"], (2 * INTER, HIDDEN // 16)).contiguous()
        s2 = view(OFF["sf2"], (HIDDEN, INTER // 16)).contiguous()
        sc = view(OFF["sc"], (24,)).contiguous().view(torch.float32).reshape(E, 6)
        layers.append({
            "w13": w13, "w2": w2,
            "w13_sf": convert_sf_to_mma_layout(s13.reshape(E * 2 * INTER, HIDDEN // 16), m=2 * INTER, k=HIDDEN,
                                               num_groups=E, sf_vec_size=16),
            "w2_sf": convert_sf_to_mma_layout(s2.reshape(E * HIDDEN, INTER // 16), m=HIDDEN, k=INTER,
                                              num_groups=E, sf_vec_size=16),
            "w1_alpha": sc[:, 1].contiguous(), "w2_alpha": sc[:, 3].contiguous(),
            "_keep": (s13, s2),
        })
        if L % 23 == 22:
            log(f"LOAD {L + 1}/{n_layers} layers  {(L + 1) * E * REC / (time.perf_counter() - t0) / 1e9:.2f} GB/s")
    torch.cuda.synchronize()
    os.close(fd)
    del stage, host
    torch.cuda.empty_cache()
    return layers, plan_sha, time.perf_counter() - t0


def main():
    from flashinfer.fused_moe import B12xMoEWrapper
    BANK = os.environ["BANK"]; PLAN = os.environ["PLAN"]; TARGET = os.environ["TARGET"]
    BIND = os.environ["BIND_IP"]; PORT = int(os.environ.get("PORT", "19600"))
    DEV = os.environ.get("DEV", "rocep1s0f0"); GID = int(os.environ.get("GID", "3"))
    LOG_EVERY = int(os.environ.get("LOG_EVERY", "20000"))

    global E
    with open(BANK, "rb") as f:
        E = struct.unpack("<IIII", f.read(24)[8:24])[2]
    plan = json.load(open(PLAN))
    moe_layers = plan["moe_layers"]
    n_layers = int(os.environ.get("LAYERS", len(moe_layers)))
    held = plan["targets"][TARGET]["layers"]
    # global id -> local slot, per model layer; -1 = not held here
    slot_of = {}
    for li, layer in enumerate(moe_layers[:n_layers]):
        ids = sorted(held[str(layer)])
        assert len(ids) == E
        m = np.full(896, -1, dtype=np.int32)
        m[np.array(ids)] = np.arange(E, dtype=np.int32)
        slot_of[layer] = (li, m)

    log(f"BOOT target={TARGET} bank={BANK} E={E} layers={n_layers} dev={DEV} bind={BIND}:{PORT} gpu={torch.cuda.get_device_name()}")
    layers, plan_sha, load_s = load_bank(BANK, n_layers)
    bank_tag = int(plan_sha[:8], 16)
    alloc = torch.cuda.memory_allocated() / 2**30
    log(f"BANK_RESIDENT {n_layers} layers in {load_s:.1f} s, allocated {alloc:.2f} GiB, plan_sha {plan_sha[:16]}")

    one = torch.ones((), dtype=torch.float32, device="cuda")
    wrappers = {}
    t0 = time.perf_counter()
    for k in range(1, TOPK + 1):
        wrappers[k] = B12xMoEWrapper(num_experts=E, top_k=k, hidden_size=HIDDEN, intermediate_size=INTER,
                                     use_cuda_graph=os.environ.get("K3_WGRAPH", "1") == "1", max_num_tokens=1,
                                     num_local_experts=E, output_dtype=torch.bfloat16,
                                     activation="situ", quant_mode="nvfp4")

    # registered ring (RDMA + GPU both see it) and a registered scratch page for ids/weights
    ring = mmap.mmap(-1, NSLOTS * SLOT + 4096)
    ring_ptr = ctypes.addressof(ctypes.c_char.from_buffer(ring))
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostRegister(ring_ptr, NSLOTS * SLOT + 4096, 0)
    assert int(rc) == 0, f"cudaHostRegister rc={rc}"
    scratch_ptr = ring_ptr + NSLOTS * SLOT
    ids_np = np.frombuffer((ctypes.c_int32 * 16).from_address(scratch_ptr), dtype=np.int32)
    w_np = np.frombuffer((ctypes.c_float * 16).from_address(scratch_ptr + 64), dtype=np.float32)
    ids_t = {k: dev_tensor(scratch_ptr, (1, k), "<i4") for k in range(1, TOPK + 1)}
    w_t = {k: dev_tensor(scratch_ptr + 64, (1, k), "<f4") for k in range(1, TOPK + 1)}
    x_t = [dev_tensor(ring_ptr + s * SLOT + HDR, (1, HIDDEN), "<i2", torch.bfloat16) for s in range(NSLOTS)]

    # Unified memory (GB10): the kernel reads x straight out of the registered RDMA ring. Discrete GPU (RTX PRO
    # 6000 on PCIe): every CTA's read of a host-registered x crosses PCIe — measured flat ~303 us kernel for
    # k=1..4 — so stage x into device memory first (7 KiB H2D) and read it from there.
    integrated = bool(torch.cuda.get_device_properties(0).is_integrated)
    x_dev = torch.empty((1, HIDDEN), dtype=torch.bfloat16, device="cuda")

    def run(k, L):
        Lr = layers[L]
        if integrated:
            xin = x_t[cur_slot]
        else:
            x_dev.copy_(x_t[cur_slot], non_blocking=True)
            xin = x_dev
        return wrappers[k].run(xin, Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], ids_t[k], w_t[k],
                               w1_alpha=Lr["w1_alpha"], w2_alpha=Lr["w2_alpha"],
                               fc2_input_scale=one, input_global_scale=one)

    # JIT every k now (cached cubins load in ~0.1 s; first-ever compile ~1.4 s each)
    cur_slot = 1
    x_t[1].normal_()
    for k in range(1, TOPK + 1):
        ids_np[:k] = np.arange(k); w_np[:k] = 1.0 / k
        torch.cuda.synchronize()
        y = run(k, 0); torch.cuda.synchronize()
        assert bool(torch.isfinite(y).all()), f"warmup k={k} non-finite"
    log(f"JIT k=1..16 in {time.perf_counter() - t0:.1f} s")

    # Whole-call CUDA graphs, one per (k, layer). The fused MoE kernel is ~15 + 75k us; the eager call around it
    # (wrapper Python + ~15 small launches + isfinite + two syncs) cost ~100 us more per request. A graph does:
    # MoE on a static x -> y into reply slot 0 -> finiteness flag into the registered scratch page.
    WG = os.environ.get("K3_WGRAPH", "1") == "1"
    graphs = {}
    x_static = torch.zeros((1, HIDDEN), dtype=torch.bfloat16, device="cuda")
    fin_t = dev_tensor(scratch_ptr + 128, (1,), "<i4")          # host reads it as fin_np[0] after sync
    fin_np = np.frombuffer((ctypes.c_int32 * 1).from_address(scratch_ptr + 128), dtype=np.int32)
    if WG:
        t1 = time.perf_counter()
        pool = torch.cuda.graph_pool_handle()
        gs = torch.cuda.Stream()

        def body(k, L):
            Lr = layers[L]
            y = wrappers[k].run(x_static, Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], ids_t[k], w_t[k],
                                w1_alpha=Lr["w1_alpha"], w2_alpha=Lr["w2_alpha"],
                                fc2_input_scale=one, input_global_scale=one)
            x_t[0].copy_(y)
            fin_t.copy_(torch.isfinite(y).all().to(torch.int32).view(1))
            return y

        x_static.normal_()
        with torch.cuda.stream(gs):
            for k in range(1, TOPK + 1):
                ids_np[:k] = np.arange(k); w_np[:k] = 1.0 / k
                body(k, 0)                                       # warm on the capture stream
            torch.cuda.synchronize()
            for L in range(n_layers):
                for k in range(1, TOPK + 1):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool, stream=gs):
                        body(k, L)
                    graphs[(k, L)] = g
        torch.cuda.synchronize()
        # bit-exact check against the eager call on sampled (k, layer)
        rs = np.random.default_rng(1)
        chk = []
        for _ in range(48):
            k = int(rs.integers(1, TOPK + 1)); L = int(rs.integers(0, n_layers))
            ids_np[:k] = rs.choice(E, k, replace=False); w_np[:k] = rs.random(k).astype(np.float32)
            x_static.normal_(); torch.cuda.synchronize()
            graphs[(k, L)].replay(); torch.cuda.synchronize()
            yg = x_t[0].clone()
            Lr = layers[L]
            ye = wrappers[k].run(x_static, Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], ids_t[k], w_t[k],
                                 w1_alpha=Lr["w1_alpha"], w2_alpha=Lr["w2_alpha"],
                                 fc2_input_scale=one, input_global_scale=one)
            ye = ye.clone()
            ye2 = wrappers[k].run(x_static, Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], ids_t[k], w_t[k],
                                  w1_alpha=Lr["w1_alpha"], w2_alpha=Lr["w2_alpha"],
                                  fc2_input_scale=one, input_global_scale=one).clone()
            torch.cuda.synchronize()
            def rel(a, b):
                return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
            r_ge, r_ee = rel(yg, ye), rel(ye2, ye)
            chk.append((k, L, r_ge, r_ee))
            assert fin_np[0] == 1
        # The B12x kernel is not run-to-run deterministic (GB10: 0.8% at k=1 .. 2.0% at k=16; RTX sm_120 heavy-tailed,
        # up to 6.6%), so the bar is distributional: graph-vs-eager must look like eager-vs-eager, not bit-equal.
        ge = np.array([c[2] for c in chk]); ee = np.array([c[3] for c in chk])
        # max over 48 samples is unstable on the RTX's heavy tail (eager alone reached 6.6% in 100 samples), so the max
        # bound also allows that known tail; the median is the real test
        assert np.median(ge) < 1.5 * np.median(ee) + 1e-3 and ge.max() < max(1.5 * ee.max() + 1e-2, 0.10), \
            f"graph differs from eager beyond kernel noise: median {np.median(ge):.2e} vs {np.median(ee):.2e}, " \
            f"max {ge.max():.2e} vs {ee.max():.2e}"
        log(f"WGRAPH check: graph-vs-eager rel median {np.median(ge):.2e} max {ge.max():.2e}, exact {sum(c[2] == 0 for c in chk)}/48; "
            f"eager-vs-eager rel median {np.median(ee):.2e} max {ee.max():.2e}, exact {sum(c[3] == 0 for c in chk)}/48")
        log(f"WGRAPH {len(graphs)} graphs captured+checked in {time.perf_counter() - t1:.1f} s, "
            f"alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        x_static.zero_()
    x_t[1].zero_()
    # The serving loop allocates no reference cycles. Python's cyclic GC otherwise runs a full gen-2 pass
    # over the torch/flashinfer/cute-dsl heap every N allocations — measured as 145-203 ms "kernel" stalls,
    # synchronised across all eight Sparks because every worker serves the same request count. (A5, 10/30)
    import gc
    gc.collect(); gc.freeze()
    if os.environ.get("K3_GC", "off") == "off":
        gc.disable()
    log(f"GC {'disabled' if not gc.isenabled() else 'enabled'}, frozen {gc.get_freeze_count()} objects; "
        f"x path: {'zero-copy (integrated)' if integrated else 'device-staged (discrete)'}")

    assert (_k3lib().k3_nslots(), _k3lib().k3_slot_bytes()) == (NSLOTS, SLOT), "libk3hop slot layout mismatch"
    BATCH = os.environ.get("K3_BATCH", "1") == "1"
    if BATCH:
        t1 = time.perf_counter()
        bwrap = B12xMoEWrapper(num_experts=E, top_k=1, hidden_size=HIDDEN, intermediate_size=INTER,
                               use_cuda_graph=False, max_num_tokens=BATCH_CAP, num_local_experts=E,
                               output_dtype=torch.bfloat16, activation="situ", quant_mode="nvfp4")
        bx = torch.empty((BATCH_MAX_M, HIDDEN), dtype=torch.bfloat16, device="cuda")
        b_ids = torch.zeros((BATCH_CAP, 1), dtype=torch.int32, device="cuda")
        b_w = torch.zeros((BATCH_CAP, 1), dtype=torch.float32, device="cuda")
        b_tix = torch.zeros((BATCH_CAP,), dtype=torch.int64, device="cuda")
        b_acc = torch.zeros((BATCH_MAX_M + 1, HIDDEN), dtype=torch.float32, device="cuda")
        h_ids = torch.zeros((BATCH_CAP,), dtype=torch.int32, pin_memory=True)
        h_w = torch.zeros((BATCH_CAP,), dtype=torch.float32, pin_memory=True)
        h_tix = torch.zeros((BATCH_CAP,), dtype=torch.int64, pin_memory=True)
        slot_np = {lay: np.array(mm, dtype=np.int32) for lay, (_, mm) in slot_of.items()}

        def bucket(p):
            i = bisect.bisect_left(BATCH_BUCKETS, p)
            return BATCH_BUCKETS[i] if i < len(BATCH_BUCKETS) else p

        def run_batch(L, xrows, n):
            Lr = layers[L]
            return bwrap.run(xrows, Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], b_ids[:n], b_w[:n],
                             w1_alpha=Lr["w1_alpha"], w2_alpha=Lr["w2_alpha"], fc2_input_scale=one,
                             input_global_scale=one)

        # build every bucket now (and the dynamic kernel), on layer 0, and check one bucket against the M=1 path
        bx.normal_(); b_w.fill_(0.5)
        # dynamic FIRST: once the dynamic workspace exists the wrapper re-sizes its static workspace (max_rows = the
        # 1,024-pair cutover), and the static kernels are keyed on it - warming statics first builds the wrong keys
        # and the first real chunk compiles for ~3 s (measured: Station batch timeout, 2026-10-31 00:26)
        for n in (BATCH_CAP, 2048) + BATCH_BUCKETS:
            b_ids[:n, 0] = torch.arange(n, device="cuda") % E
            y = run_batch(0, bx[torch.arange(n, device="cuda") % BATCH_MAX_M], n); torch.cuda.synchronize()
            assert bool(torch.isfinite(y).all()), f"batch warmup n={n} non-finite"
        b_ids[:32, 0] = 0; b_w[:32, 0] = 0.0
        b_ids[:3, 0] = torch.tensor([5, 9, 5], dtype=torch.int32); b_w[:3, 0] = torch.tensor([0.25, 0.75, 1.0])
        rows = torch.zeros(32, dtype=torch.int64, device="cuda"); rows[2] = 1
        yb = run_batch(0, bx.index_select(0, rows), 32)[:3].float().clone()
        cur_slot = 1
        x_t[1].copy_(bx[0:1]); ids_np[:2] = [5, 9]; w_np[:2] = [0.25, 0.75]; torch.cuda.synchronize()
        y1 = run(2, 0).float().clone(); torch.cuda.synchronize()
        r = float(((yb[0] + yb[1]) - y1[0]).norm() / y1[0].norm().clamp_min(1e-12))
        assert r < 0.05, f"batch path disagrees with the M=1 path: rel {r:.3e}"
        x_t[1].zero_()
        log(f"BATCH path ready: buckets {BATCH_BUCKETS} + dynamic, built in {time.perf_counter() - t1:.1f} s; "
            f"pairs-vs-M=1 rel {r:.2e}")
    b_served = b_tokens = b_pairs = 0
    b_us = 0.0

    conn = Conn.listen(BIND, PORT, DEV, GID, ring=ring_ptr, timeout_s=3600)
    log("READY (RTS)")

    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    served = empty = nonfinite = 0
    khist = [0] * (TOPK + 1)
    alloc0 = torch.cuda.memory_allocated()
    while True:
        s = conn.poll(5_000_000)
        if s < 0:
            continue
        cur_slot = s
        t_recv = time.monotonic_ns()
        buf = conn.view(s, MSG_BYTES)
        f = REQ.unpack_from(buf, 0)
        req_id, layer, n_tok, flags, tag = f[:5]
        gids = f[5:5 + TOPK]; ws = f[5 + TOPK:5 + 2 * TOPK]
        rep_slot = 0                       # slot 0 is send-only
        out = conn.view(rep_slot, MSG_BYTES)
        if req_id == SHUTDOWN_ID or flags & FLAG_PROBE:
            REP.pack_into(out, 0, req_id, ST_OK, served, 0, bank_tag, t_recv, 0, 0, 0, time.monotonic_ns(), 0)
            conn.send(rep_slot, 64)
            conn.post_recv(s)
            if req_id == SHUTDOWN_ID:
                break
            continue
        if BATCH and flags & FLAG_BATCH and layer in slot_of:
            Mb = BATCH_M.unpack_from(buf, BATCH_M_OFF)[0]
            if not 1 <= Mb <= BATCH_MAX_M:
                REP.pack_into(out, 0, req_id, ST_BAD, 0, 0, bank_tag, t_recv, 0, 0, 0, time.monotonic_ns(), 0)
                conn.send(rep_slot, 64); conn.post_recv(s)
                log(f"BAD batch M={Mb}")
                continue
            L, _m = slot_of[layer]
            v = conn.view(s, batch_req_bytes(Mb))
            gid = np.frombuffer(v, np.int32, Mb * TOPK, HDR).reshape(Mb, TOPK)
            gw = np.frombuffer(v, np.float32, Mb * TOPK, HDR + 64 * Mb).reshape(Mb, TOPK)
            valid = (gid >= 0) & (gid < 896)
            loc = np.where(valid, slot_np[layer][np.clip(gid, 0, 895)], -1)
            sel = loc >= 0
            kused = sel.sum(1).astype(np.uint16)
            tix, col = np.nonzero(sel)
            P = len(tix)
            t_x = time.monotonic_ns()
            xin = dev_tensor(ring_ptr + s * SLOT + batch_xoff(Mb), (Mb, HIDDEN), "<i2", torch.bfloat16)
            bx[:Mb].copy_(xin, non_blocking=True)
            ok = True
            yoff = batch_yoff(Mb)
            ydst = dev_tensor(ring_ptr + yoff, (Mb, HIDDEN), "<i2", torch.bfloat16)   # reply slot 0
            t_k = time.monotonic_ns()
            if P:
                n = bucket(P)
                hi, hw, ht = h_ids.numpy(), h_w.numpy(), h_tix.numpy()
                hi[:n] = 0; hw[:n] = 0.0; ht[:n] = Mb          # padding rows: weight 0, summed into a dummy row
                hi[:P] = loc[tix, col]; hw[:P] = gw[tix, col]; ht[:P] = tix
                b_ids[:n, 0].copy_(h_ids[:n], non_blocking=True)
                b_w[:n, 0].copy_(h_w[:n], non_blocking=True)
                b_tix[:n].copy_(h_tix[:n], non_blocking=True)
                y = run_batch(L, bx.index_select(0, b_tix[:n].clamp(max=Mb - 1)), n)
                acc = b_acc[:Mb + 1]; acc.zero_()
                acc.index_add_(0, b_tix[:n], y.float())
                yb16 = acc[:Mb].to(torch.bfloat16)
                ydst.copy_(yb16)
                fin = torch.isfinite(yb16).all()
                torch.cuda.current_stream().synchronize()
                ok = bool(fin)
            else:
                ydst.zero_(); torch.cuda.current_stream().synchronize()
            t_kd = time.monotonic_ns()
            ob = conn.view(rep_slot, batch_rep_bytes(Mb))
            np.frombuffer(ob, np.uint16, Mb, HDR)[:] = kused
            status = ST_OK if ok else ST_NONFINITE
            nonfinite += (not ok)
            REP.pack_into(ob, 0, req_id, status, P, 0, bank_tag, t_recv, t_x, t_k, t_kd, time.monotonic_ns(),
                          t_kd - t_k)
            conn.send(rep_slot, batch_rep_bytes(Mb))
            conn.post_recv(s)
            b_served += 1; b_tokens += Mb; b_pairs += P; b_us += (t_kd - t_k) / 1e3
            if b_served % (LOG_EVERY // 10 or 1) == 0:
                log(f"BATCH {b_served} calls, {b_tokens} tokens, {b_pairs} pairs, mean kernel+sum "
                    f"{b_us / b_served:.0f} us")
            continue
        if layer not in slot_of or n_tok != 1:
            REP.pack_into(out, 0, req_id, ST_BAD, 0, 0, bank_tag, t_recv, 0, 0, 0, time.monotonic_ns(), 0)
            conn.send(rep_slot, 64); conn.post_recv(s)
            log(f"BAD layer={layer} n_tok={n_tok}")
            continue
        L, m = slot_of[layer]
        k = 0; mask = 0
        for j in range(TOPK):
            g = gids[j]
            if 0 <= g < 896 and m[g] >= 0:
                ids_np[k] = m[g]; w_np[k] = ws[j]; k += 1; mask |= 1 << j
        khist[k] += 1
        if k == 0:
            REP.pack_into(out, 0, req_id, ST_EMPTY, 0, 0, bank_tag, t_recv, t_recv, t_recv, t_recv,
                          time.monotonic_ns(), 0)
            ctypes.memset(conn.addr(rep_slot) + HDR, 0, HIDDEN * 2)
            conn.send(rep_slot, MSG_BYTES); conn.post_recv(s)
            empty += 1
            continue
        t_x = time.monotonic_ns()
        if WG:
            x_static.copy_(x_t[cur_slot], non_blocking=True)
            t_k = time.monotonic_ns()
            graphs[(k, L)].replay()
            torch.cuda.current_stream().synchronize()
            t_kd = time.monotonic_ns()
            ok = bool(fin_np[0] == 1)
            kernel_ns = t_kd - t_k
        else:
            ev0.record(); t_k = time.monotonic_ns()
            y = run(k, L)
            x_t[rep_slot].copy_(y)             # y straight into the reply slot (device write to registered memory)
            ev1.record(); ev1.synchronize()
            t_kd = time.monotonic_ns()
            ok = bool(torch.isfinite(y).all())
            kernel_ns = int(ev0.elapsed_time(ev1) * 1e6)
        status = ST_OK if ok else ST_NONFINITE
        nonfinite += (not ok)
        REP.pack_into(out, 0, req_id, status, k, mask, bank_tag, t_recv, t_x, t_k, t_kd, time.monotonic_ns(),
                      kernel_ns)
        conn.send(rep_slot, MSG_BYTES)
        conn.post_recv(s)
        served += 1
        t_end = time.monotonic_ns()
        if t_end - t_recv > 5_000_000:
            log(f"SLOW req={req_id} layer={layer} k={k} total_us={(t_end - t_recv) / 1e3:.0f} "
                f"pre_us={(t_k - t_recv) / 1e3:.0f} kernel_us={kernel_ns / 1e3:.0f} gpu_wall_us={(t_kd - t_k) / 1e3:.0f}")
        if served % LOG_EVERY == 0:
            log(f"SERVED {served} empty={empty} nonfinite={nonfinite} khist={khist[:8]} "
                f"alloc_delta={torch.cuda.memory_allocated() - alloc0}")
    log(f"EXIT served={served} empty={empty} nonfinite={nonfinite} khist={khist} "
        f"alloc_delta={torch.cuda.memory_allocated() - alloc0}")
    conn.close()


if __name__ == "__main__":
    main()

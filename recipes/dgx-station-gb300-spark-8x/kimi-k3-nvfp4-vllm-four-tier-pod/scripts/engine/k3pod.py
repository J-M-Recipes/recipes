"""k3pod — the Kimi-K3 four-tier engine hook for vLLM 0.31.0 (eager, v1).

Activated by env K3POD_PLAN=<plan json>. Imported from sitecustomize in every process.

What it changes, and nothing else:
  * Weight loading: routed experts held by the Station (plan["station"]) are renamed to local slots
    0..n_local-1; every other routed expert is dropped (never read into the process).
  * KimiMoE.__init__: the routed experts are built with n_local experts; the gate stays 896 wide.
  * Router: select_experts runs over all 896 logits (vLLM's own router, bias re-pointed at the 896 bias),
    then global ids are remapped to local slots (-1 = held elsewhere). G2: -1 contributes exactly zero.
  * forward_modular: after the stock kernel computes the Station's share, the remote share is fetched
    from the nine expert workers (8 Sparks + RTX PRO 6000) and added. Asserted every layer and token:
    k_station + sum(k_used) == top_k.
"""
from __future__ import annotations

import json, os, re, sys, time

PLAN_PATH = os.environ.get("K3POD_PLAN")
LOG_EVERY = int(os.environ.get("K3POD_LOG_EVERY", "50"))
MAX_REMOTE_M = int(os.environ.get("K3POD_MAX_REMOTE_M", "512"))
# Shadow mode: the Station loads ALL experts, computes the exact layer output, also computes the pod output
# (Station share via the stock kernel + the nine workers), logs their per-layer cosine, and continues on the
# exact output (teacher forcing) so every layer is compared on reference inputs.
SHADOW = os.environ.get("K3POD_SHADOW") == "1"
FAST = os.environ.get("K3POD_FAST", "1") == "1"
# K3-4: multi-token calls (prefill chunks, batch > 1) send ONE message per worker carrying every token that worker
# serves, instead of one round trip per token. Workers must run K3_BATCH=1 (worker.py) on the 2 MiB-slot libk3hop.
BATCH = os.environ.get("K3POD_BATCH", "1") == "1"
# K3-5: decode steps of 2..PXM tokens also go through the graph-resident proxy (C batch path) instead of breaking the
# graph. 1 = on. Needs the K3-5 libk3hop (k3proxy_set_batch) and K3-5 workers (small-batch graphs).
PX_BATCH = os.environ.get("K3POD_PROXY_BATCH", "1") == "1"
PXM = 8
# diagnostics: time the Station's own routed experts (CUDA events, synchronizing - slows prefill a little)
TIME = os.environ.get("K3POD_TIME") == "1"
_TM = {"n": 0, "ms": 0.0, "tok": 0}
# Graph-resident expert step for one-token decode (idea: Javier, @javierbits): no eager graph break; the GPU writes
# x/ids/w + a ready flag into pinned host memory, a C proxy thread (libk3hop k3proxy_*) sends/collects, and the
# graph waits on a done flag with cuStreamWaitValue32.
PROXY = os.environ.get("K3POD_PROXY") == "1"
# Record mode (eager only): for every one-token MoE call, keep x, global ids, weights, the Station's share and the
# workers' share, and write them per chunk to K3POD_DUMP_DIR for an offline reference check of all 92 layers.
DUMP_DIR = os.environ.get("K3POD_DUMP_DIR") or None
EXPERT_RE = re.compile(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.")


def _log(*a):
    print(f"[k3pod {os.getpid()} {time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)


class _Plan:
    def __init__(self, path):
        p = json.load(open(path))
        self.layers = [int(x) for x in p["moe_layers"]]
        st = p["station"]["layers"]
        self.local = {}                      # layer -> {global id: local slot}
        for L in self.layers:
            ids = sorted(st[str(L)])
            self.local[L] = {g: i for i, g in enumerate(ids)}
        n = {len(v) for v in self.local.values()}
        assert len(n) == 1, f"station set size must be uniform per layer, got {n}"
        self.n_local = n.pop()
        self.workers = []                    # (name, ip, port, dev)
        self.owner = {}                      # layer -> list[896] worker index or -1
        names = sorted(t for t in p["targets"] if t.startswith("spark")) + \
            sorted(t for t in p["targets"] if t.startswith("rtx"))
        for wi, t in enumerate(names):
            if t.startswith("spark"):
                k = int(t[5:])
                self.workers.append((t, f"10.0.0.{k + 1}", 19600, "mlx5_0"))
            else:
                self.workers.append((t, "10.0.1.9", 19601, "mlx5_1"))
        for L in self.layers:
            o = [-1] * 896
            for wi, t in enumerate(names):
                for g in p["targets"][t]["layers"][str(L)]:
                    assert o[g] == -1, (L, g)
                    o[g] = wi
            for g in self.local[L]:
                assert o[g] == -1, (L, g)
            assert sum(1 for x in o if x == -1) == self.n_local, L
            self.owner[L] = o


PLAN = _Plan(PLAN_PATH) if PLAN_PATH else None


# ---------------------------------------------------------------------------------------------
# weight-name rewrite (called from the sitecustomize weights-iterator wrapper)
def rewrite_name(name: str):
    """Return the name to load under, or None to drop the tensor."""
    if PLAN is None or SHADOW:
        return name
    m = EXPERT_RE.search(name)
    if not m:
        return name
    L, g = int(m.group(1)), int(m.group(2))
    slot = PLAN.local.get(L, {}).get(g)
    if slot is None:
        return None
    return name[:m.start(2)] + str(slot) + name[m.end(2):]


# ---------------------------------------------------------------------------------------------
class _CAI:
    """Minimal __cuda_array_interface__ so torch can wrap host-registered (device-visible) memory."""
    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {"data": (ptr, False), "shape": shape, "typestr": typestr,
                                         "version": 3, "strides": None}


def _dev_tensor(ptr, shape, typestr, dtype=None):
    import torch
    t = torch.as_tensor(_CAI(ptr, shape, typestr), device="cuda")
    return t.view(dtype) if dtype is not None else t


class _Pod:
    """Station-side client for the nine expert workers. One request in flight per worker."""

    def __init__(self):
        self.conns = None
        self.calls = 0
        self.t_remote = 0.0      # waiting for / collecting worker replies (after the local kernel launch)
        self.t_send = 0.0
        self.t_d2h = 0.0         # GPU->CPU copy of x / ids / w (includes waiting for the GPU to get there)
        self.skipped = 0
        self.wstats = {}

    def connect(self):
        sys.path.insert(0, os.environ.get("K3POD_PROTO_DIR", "/k3a3"))
        import gc
        from k3proto import Conn, HDR, MSG_BYTES, REQ, REP  # noqa
        import k3proto as KP
        self.HDR, self.MSG, self.REQ, self.REP = HDR, MSG_BYTES, REQ, REP
        self.KP = KP
        self.SLOT = KP.lib().k3_slot_bytes()
        self.conns, self.req = [], []
        self.rings = []
        GPUB = BATCH and os.environ.get("K3POD_BATCH_GPU", "1") == "1"
        self.gpub = GPUB
        if GPUB:
            # Station rings allocated here (page-aligned for the 64 KiB-page kernel) and cudaHostRegister'ed, so the
            # GPU gathers batch rows straight into the send slot and sums replies straight out of the recv slots.
            import ctypes as _ct, mmap as _mm, torch as _torch
            _cudart = _torch.cuda.cudart()
            _rb = KP.lib().k3_ring_bytes()
        for name, ip, port, dev in PLAN.workers:
            ring = None
            if GPUB:
                mmo = _mm.mmap(-1, _rb)
                ring = _ct.addressof(_ct.c_char.from_buffer(mmo))
                rc = _cudart.cudaHostRegister(ring, _rb, 0)
                if int(rc) != 0:
                    raise RuntimeError(f"k3pod: cudaHostRegister of the {name} ring failed rc={rc}")
                self.rings.append((mmo, ring))
            c = Conn.connect(ip, port, dev, 3, ring=ring)
            self.conns.append(c)
            self.req.append(c.view(0, MSG_BYTES))
            if BATCH:
                assert self.SLOT >= KP.batch_req_bytes(KP.BATCH_MAX_M), f"libk3hop slot {self.SLOT} too small for batch"
            _log(f"connected {name} {ip}:{port} via {dev}")
        self.rid = 0
        # C fast path (one token): owner table [93][896] int8, conn handle array, reply rows [n][H] bf16
        import ctypes, numpy as np
        L = __import__("k3proto").lib()
        L.k3pod_send.restype = ctypes.c_longlong
        L.k3pod_send.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p,
                                 ctypes.c_void_p]
        L.k3pod_collect.restype = ctypes.c_int
        L.k3pod_collect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
        self.clib = L
        n = len(self.conns)
        self.handles = (ctypes.c_void_p * n)(*[c.h for c in self.conns])
        self.owner_tab = np.full((93, 896), -1, dtype=np.int8)
        for lay, o in PLAN.owner.items():
            self.owner_tab[lay] = np.array(o, dtype=np.int8)
        self.kexp = np.zeros(n, dtype=np.int8)
        self.nst = ctypes.c_int(0)
        self.fast_rid = 1 << 30
        gc.freeze()

    def _send_token(self, L, xrow_u8, row, wl):
        own = PLAN.owner[L]
        # padding rows (CUDA-graph capture/replay with fewer real tokens) carry ids outside [0, 896):
        # they belong to nobody (-2) and are neither computed nor counted against the Station
        o = [own[g] if 0 <= g < 896 else -2 for g in row]
        targets = sorted(set(v for v in o if v >= 0))
        self._t_sent = time.perf_counter()
        for wi in targets:
            self.rid += 1
            buf = self.req[wi]
            self.REQ.pack_into(buf, 0, self.rid, L, 1, 0, 0, *row, *wl)
            self._np.frombuffer(buf, dtype=self._np.uint8, count=xrow_u8.size, offset=self.HDR)[:] = xrow_u8
            self.conns[wi].send(0, self.MSG)
        return o, targets

    def _collect_token(self, L, o, targets, out_row, K):
        np = self._np
        k_station = sum(1 for v in o if v < 0)       # -1 Station, -2 padding
        got = 0
        last = None
        for wi in targets:
            c = self.conns[wi]
            s = c.poll(2_000_000)
            if s < 0:
                raise RuntimeError(f"k3pod: worker {PLAN.workers[wi][0]} timed out, layer {L}")
            rv = c.view(s, self.MSG)
            r = self.REP.unpack_from(rv, 0)
            if r[1] != 0:
                raise RuntimeError(f"k3pod: worker {PLAN.workers[wi][0]} status {r[1]} layer {L}")
            want = sum(1 for v in o if v == wi)
            if r[2] != want:
                raise RuntimeError(f"k3pod: {PLAN.workers[wi][0]} k_used {r[2]} != {want}, layer {L}")
            got += r[2]
            ta = time.perf_counter() - self._t_sent
            st = self.wstats.setdefault(wi, [0, 0.0, 0.0, 0])   # n, sum arrival s, sum worker-side s, n last
            st[0] += 1; st[1] += ta; st[2] += (r[9] - r[5]) / 1e9
            last = (wi, ta) if (not last or ta > last[1]) else last
            y = np.frombuffer(rv, dtype=np.uint16, count=out_row.shape[0], offset=self.HDR)
            out_row += (y.astype(np.uint32) << 16).view(np.float32)
            c.post_recv(s)
        if got + k_station != K:
            raise RuntimeError(f"k3pod: partition broken layer {L}: {got} + {k_station} != {K}")
        if last:
            self.wstats[last[0]][3] += 1

    def begin(self, L, x_u16, ids, w):
        """Send token 0 now (overlaps the Station's own experts). Returns a handle for end()."""
        import numpy as np
        self._np = np
        if self.conns is None:
            self.connect()
        if BATCH and ids.shape[0] > 1:
            return ("B", self._batch_send(L, x_u16, ids, w))
        t0 = time.perf_counter()
        row = [int(v) for v in ids[0]]
        o, targets = self._send_token(L, x_u16[0].view(np.uint8), row, [float(v) for v in w[0]])
        self.t_send += time.perf_counter() - t0
        return (L, x_u16, ids, w, o, targets)

    def fast_send(self, L, x_ptr, ids_ptr, w_ptr, hidden):
        """One token: C packs and posts every target's dispatch. Returns (mask, rid_base)."""
        import ctypes
        if self.conns is None:
            self.connect()
        t0 = time.perf_counter()
        self.fast_rid += len(self.conns)
        rid = self.fast_rid & 0x7FFFFFFF
        m = self.clib.k3pod_send(self.handles, len(self.conns), self.owner_tab[L].ctypes.data, ids_ptr, w_ptr, x_ptr,
                                 hidden, L, rid, self.kexp.ctypes.data, ctypes.byref(self.nst))
        if m < 0:
            raise RuntimeError(f"k3pod_send layer {L}: {self.clib.k3_last_error().decode()}")
        self.t_send += time.perf_counter() - t0
        return (m, rid, self.kexp.copy(), self.nst.value)

    def fast_collect(self, L, h, rows_ptr, hidden):
        m, rid, kexp, nst = h
        t0 = time.perf_counter()
        got = self.clib.k3pod_collect(self.handles, len(self.conns), m, kexp.ctypes.data, rows_ptr, hidden,
                                      2_000_000, rid)
        if got < 0:
            raise RuntimeError(f"k3pod_collect layer {L}: {self.clib.k3_last_error().decode()}")
        if got + nst != 16:
            raise RuntimeError(f"k3pod: partition broken layer {L}: {got} + {nst} != 16")
        self.t_remote += time.perf_counter() - t0

    def _batch_send(self, L, x_u16, ids, w):
        """One message per worker with every token it serves (M <= BATCH_MAX_M per message; larger M is split)."""
        np, KP = self._np, self.KP
        t0 = time.perf_counter()
        M = ids.shape[0]
        own = self.owner_tab[L].astype(np.int16)
        valid = (ids >= 0) & (ids < 896)
        o = np.where(valid, own[np.clip(ids, 0, 895)], -2)           # -1 Station, -2 padding/none, >=0 worker
        k_station = (o < 0).sum(1)
        parts = []                                                    # (wi, token rows, want per token)
        for wi in range(len(self.conns)):
            hit = (o == wi)
            want = hit.sum(1)
            rows = np.nonzero(want)[0]
            if rows.size:
                parts.append((wi, rows, want[rows]))
        x8 = x_u16.view(np.uint8).reshape(M, -1)
        groups = []
        for wi, rows, want in parts:
            for a in range(0, rows.size, KP.BATCH_MAX_M):
                r = rows[a:a + KP.BATCH_MAX_M]
                groups.append((wi, r, want[a:a + KP.BATCH_MAX_M]))
        # first message per worker now; any overflow messages are sent as earlier ones are collected
        pending = {}
        for g in groups:
            pending.setdefault(g[0], []).append(g)
        self._t_sent = time.perf_counter()
        for wi, gl in pending.items():
            self._batch_post(L, wi, gl[0], ids, w, x8)
        self.t_send += time.perf_counter() - t0
        return (L, M, pending, k_station, ids, w, x8)

    def _batch_post(self, L, wi, g, ids, w, x8):
        np, KP = self._np, self.KP
        _, r, _ = g
        m = r.size
        self.rid += 1
        v = self.conns[wi].view(0, KP.batch_req_bytes(m))
        self.REQ.pack_into(v, 0, self.rid, L, 0, KP.FLAG_BATCH, 0, *([-1] * 16), *([0.0] * 16))
        KP.BATCH_M.pack_into(v, KP.BATCH_M_OFF, m)
        np.frombuffer(v, np.int32, m * 16, self.HDR).reshape(m, 16)[:] = ids[r]
        np.frombuffer(v, np.float32, m * 16, self.HDR + 64 * m).reshape(m, 16)[:] = w[r]
        np.frombuffer(v, np.uint8, m * x8.shape[1], KP.batch_xoff(m)).reshape(m, -1)[:] = x8[r]
        self.conns[wi].send(0, KP.batch_req_bytes(m))

    def _batch_end(self, h):
        np, KP = self._np, self.KP
        L, M, pending, k_station, ids, w, x8 = h
        H = x8.shape[1] // 2
        out = np.zeros((M, H), dtype=np.float32)
        got = np.zeros(M, dtype=np.int64)
        t0 = time.perf_counter()
        for wi, gl in pending.items():
            c = self.conns[wi]
            for gi, (_, r, want) in enumerate(gl):
                if gi:
                    self._batch_post(L, wi, gl[gi], ids, w, x8)
                s = c.poll(5_000_000)          # 5 s: a stray kernel compile on a worker is ~3 s; decode keeps 2 s
                if s < 0:
                    raise RuntimeError(f"k3pod: worker {PLAN.workers[wi][0]} timed out (batch), layer {L}")
                m = r.size
                rv = c.view(s, KP.batch_rep_bytes(m))
                rr = self.REP.unpack_from(rv, 0)
                if rr[1] != 0:
                    raise RuntimeError(f"k3pod: worker {PLAN.workers[wi][0]} status {rr[1]} (batch) layer {L}")
                ku = np.frombuffer(rv, np.uint16, m, self.HDR).astype(np.int64)
                if not np.array_equal(ku, want):
                    bad = int(np.nonzero(ku != want)[0][0])
                    raise RuntimeError(f"k3pod: {PLAN.workers[wi][0]} batch k_used {ku[bad]} != {want[bad]} "
                                       f"(token {int(r[bad])}), layer {L}")
                got[r] += ku
                y = np.frombuffer(rv, np.uint16, m * H, KP.batch_yoff(m)).reshape(m, H)
                out[r] += (y.astype(np.uint32) << 16).view(np.float32)
                c.post_recv(s)
                st = self.wstats.setdefault(wi, [0, 0.0, 0.0, 0])
                st[0] += 1; st[1] += time.perf_counter() - self._t_sent; st[2] += (rr[9] - rr[5]) / 1e9
        bad = np.nonzero(got + k_station != ids.shape[1])[0]
        if bad.size:
            t = int(bad[0])
            raise RuntimeError(f"k3pod: partition broken layer {L} token {t}: {got[t]} + {k_station[t]} != {ids.shape[1]}")
        self.t_remote += time.perf_counter() - t0
        self.batch_calls = getattr(self, "batch_calls", 0) + 1
        self.batch_tokens = getattr(self, "batch_tokens", 0) + M
        return out

    def batch_send_gpu(self, L, x, ids, w):
        """Batched send with the x rows gathered on the GPU straight into each worker's (registered) send slot."""
        import numpy as np
        import torch
        self._np = np
        if self.conns is None:
            self.connect()
        KP = self.KP
        t0 = time.perf_counter()
        M, H = x.shape
        own = self.owner_tab[L].astype(np.int16)
        valid = (ids >= 0) & (ids < 896)
        o = np.where(valid, own[np.clip(ids, 0, 895)], -2)
        k_station = (o < 0).sum(1)
        parts = []
        for wi in range(len(self.conns)):
            want = (o == wi).sum(1)
            rows = np.nonzero(want)[0]
            if not rows.size:
                continue
            m = rows.size
            c = self.conns[wi]
            base = c.addr(0)
            self.rid += 1
            v = c.view(0, KP.batch_xoff(m))
            self.REQ.pack_into(v, 0, self.rid, L, 0, KP.FLAG_BATCH, 0, *([-1] * 16), *([0.0] * 16))
            KP.BATCH_M.pack_into(v, KP.BATCH_M_OFF, m)
            np.frombuffer(v, np.int32, m * 16, self.HDR).reshape(m, 16)[:] = ids[rows]
            np.frombuffer(v, np.float32, m * 16, self.HDR + 64 * m).reshape(m, 16)[:] = w[rows]
            rows_t = torch.from_numpy(rows).to(x.device)
            xv = _dev_tensor(base + KP.batch_xoff(m), (m, H), "<i2", torch.bfloat16)
            torch.index_select(x, 0, rows_t, out=xv)
            parts.append((wi, rows, want[rows], rows_t, m))
        torch.cuda.current_stream().synchronize()          # rows are in the slots before the NIC reads them
        self._t_sent = time.perf_counter()
        for wi, _, _, _, m in parts:
            self.conns[wi].send(0, KP.batch_req_bytes(m))
        self.t_send += time.perf_counter() - t0
        return (L, M, H, parts, k_station, ids.shape[1])

    def batch_end_gpu(self, h, add):
        """Collect batched replies; each reply's y rows are added on the GPU straight out of its recv slot."""
        import torch
        np, KP = self._np, self.KP
        L, M, H, parts, k_station, K = h
        t0 = time.perf_counter()
        add.zero_()
        got = np.zeros(M, dtype=np.int64)
        used = []
        try:
            for wi, rows, want, rows_t, m in parts:
                c = self.conns[wi]
                s = c.poll(5_000_000)
                if s < 0:
                    raise RuntimeError(f"k3pod: worker {PLAN.workers[wi][0]} timed out (batch), layer {L}")
                used.append((c, s))
                rv = c.view(s, KP.batch_yoff(m))
                rr = self.REP.unpack_from(rv, 0)
                if rr[1] != 0:
                    raise RuntimeError(f"k3pod: worker {PLAN.workers[wi][0]} status {rr[1]} (batch) layer {L}")
                ku = np.frombuffer(rv, np.uint16, m, self.HDR).astype(np.int64)
                if not np.array_equal(ku, want):
                    bad = int(np.nonzero(ku != want)[0][0])
                    raise RuntimeError(f"k3pod: {PLAN.workers[wi][0]} batch k_used {ku[bad]} != {want[bad]} "
                                       f"(token {int(rows[bad])}), layer {L}")
                got[rows] += ku
                y = _dev_tensor(c.addr(s) + KP.batch_yoff(m), (m, H), "<i2", torch.bfloat16)
                add.index_add_(0, rows_t, y.float())
                st = self.wstats.setdefault(wi, [0, 0.0, 0.0, 0])
                st[0] += 1; st[1] += time.perf_counter() - self._t_sent; st[2] += (rr[9] - rr[5]) / 1e9
            bad = np.nonzero(got + k_station != K)[0]
            if bad.size:
                t = int(bad[0])
                raise RuntimeError(f"k3pod: partition broken layer {L} token {t}: {got[t]} + {k_station[t]} != {K}")
            torch.cuda.current_stream().synchronize()      # the GPU has read every slot before it is re-armed
        finally:
            for c, s in used:
                c.post_recv(s)
        self.t_remote += time.perf_counter() - t0
        self.batch_calls = getattr(self, "batch_calls", 0) + 1
        self.batch_tokens = getattr(self, "batch_tokens", 0) + M

    def end(self, h):
        np = self._np
        if h[0] == "B":
            return self._batch_end(h[1])
        L, x_u16, ids, w, o, targets = h
        M, K = ids.shape
        out = np.zeros((M, x_u16.shape[1]), dtype=np.float32)
        t0 = time.perf_counter()
        self._collect_token(L, o, targets, out[0], K)
        for t in range(1, M):
            row = [int(v) for v in ids[t]]
            o2, tg2 = self._send_token(L, x_u16[t].view(np.uint8), row, [float(v) for v in w[t]])
            self._collect_token(L, o2, tg2, out[t], K)
        self.t_remote += time.perf_counter() - t0
        return out


POD = _Pod()


_OPS = {}


def _ops():
    """Eager-break ops for vLLM's breakable CUDA graphs: during capture they end the current graph segment,
    run eagerly, and are replayed in order. Built lazily inside the engine process."""
    if _OPS:
        return _OPS
    import numpy as np
    import torch
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    pend = {}
    pin = {}             # reused pinned D2H staging (begin() copies token rows out before the next layer)
    host = {}            # layer -> pinned f32 (8, H): one per layer so async H2D copies never race

    @eager_break_during_capture
    def pod_send(L, x, ids, w):
        t0 = time.perf_counter()
        M = x.shape[0]
        pb = pin.get("b")
        if pb is None or pb[0].shape[0] < M:
            cap = max(M, 8)
            pb = pin["b"] = (torch.empty((cap, x.shape[1]), dtype=torch.bfloat16, pin_memory=True),
                             torch.empty((cap, ids.shape[1]), dtype=torch.int32, pin_memory=True),
                             torch.empty((cap, w.shape[1]), dtype=torch.float32, pin_memory=True))
            pin["ev"] = torch.cuda.Event()
        xb, ib, wb = pb[0][:M], pb[1][:M], pb[2][:M]
        if POD.conns is None:
            POD.connect()
        gpu_batch = BATCH and POD.gpub and 1 < M <= POD.KP.BATCH_MAX_M
        if not gpu_batch:
            xb.copy_(x, non_blocking=True)      # one sync for all three copies instead of three
        ib.copy_(ids, non_blocking=True)
        wb.copy_(w, non_blocking=True)
        pin["ev"].record()
        pin["ev"].synchronize()
        POD.t_d2h += time.perf_counter() - t0
        if gpu_batch:
            pend[L] = ("gpu", POD.batch_send_gpu(L, x, ib.numpy(), wb.numpy()))
        elif M == 1 and FAST:
            pend[L] = ("fast", POD.fast_send(L, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), x.shape[1]))
        else:
            pend[L] = ("py", POD.begin(L, xb.view(torch.int16).numpy().view(np.uint16), ib.numpy(), wb.numpy()))

    @eager_break_during_capture
    def pod_collect(L, add):
        kind, h = pend.pop(L)
        if kind == "gpu":
            POD.batch_end_gpu(h, add)
            POD.calls += 1
            if POD.calls % (LOG_EVERY * len(PLAN.layers)) == 0:
                n = LOG_EVERY * len(PLAN.layers)
                _log(f"{POD.calls} layer calls (gpu batch); per layer over last {n}: d2h {POD.t_d2h / n * 1e3:.3f} ms, "
                     f"send {POD.t_send / n * 1e3:.3f} ms, wait+collect {POD.t_remote / n * 1e3:.3f} ms")
                POD.t_d2h = POD.t_send = POD.t_remote = 0.0
                ws = " ".join(f"{PLAN.workers[w][0]}:arr{v[1] / v[0] * 1e6:.0f}us/wrk{v[2] / v[0] * 1e6:.0f}us"
                              for w, v in sorted(POD.wstats.items()))
                _log("workers " + ws)
                POD.wstats = {}
            return
        if kind == "fast":
            rows = pin.get("rows")
            if rows is None:
                n = len(PLAN.workers)
                rows = pin["rows"] = torch.empty((n, add.shape[1]), dtype=torch.bfloat16, pin_memory=True)
                pin["rows_dev"] = torch.empty((n, add.shape[1]), dtype=torch.bfloat16, device=add.device)
            if pin.get("ev2_pending"):
                pin["ev2"].synchronize()
                pin["ev2_pending"] = False
            POD.fast_collect(L, h, rows.data_ptr(), add.shape[1])
            pin["rows_dev"].copy_(rows, non_blocking=True)
            torch.sum(pin["rows_dev"].float(), dim=0, keepdim=True, out=add[:1])
            pin["ev2"] = pin.get("ev2") or torch.cuda.Event()
            pin["ev2"].record()        # the next layer's collect must not overwrite `rows` before this copy runs
            pin["ev2_pending"] = True
            POD.calls += 1
            if POD.calls % (LOG_EVERY * len(PLAN.layers)) == 0:
                n = LOG_EVERY * len(PLAN.layers)
                _log(f"{POD.calls} layer calls (fast); per layer over last {n}: d2h {POD.t_d2h / n * 1e3:.3f} ms, "
                     f"send {POD.t_send / n * 1e3:.3f} ms, wait+collect {POD.t_remote / n * 1e3:.3f} ms")
                POD.t_d2h = POD.t_send = POD.t_remote = 0.0
            return
        r = POD.end(h)
        M = r.shape[0]
        if M <= 8:
            hb = host.get(L)
            if hb is None:
                hb = host[L] = torch.empty((8, r.shape[1]), dtype=torch.float32, pin_memory=True)
            hb[:M].numpy()[:] = r
            add.copy_(hb[:M], non_blocking=True)
        else:
            add.copy_(torch.from_numpy(r))
        POD.calls += 1
        if POD.calls % (LOG_EVERY * len(PLAN.layers)) == 0:
            n = LOG_EVERY * len(PLAN.layers)
            _log(f"{POD.calls} layer calls; per layer over last {n}: d2h {POD.t_d2h / n * 1e3:.3f} ms, "
                 f"send {POD.t_send / n * 1e3:.3f} ms, wait+collect {POD.t_remote / n * 1e3:.3f} ms")
            POD.t_d2h = POD.t_send = POD.t_remote = 0.0
            ws = " ".join(f"{PLAN.workers[w][0]}:arr{v[1] / v[0] * 1e6:.0f}us/wrk{v[2] / v[0] * 1e6:.0f}us/last{v[3]}"
                          for w, v in sorted(POD.wstats.items()))
            _log("workers " + ws)
            POD.wstats = {}

    @eager_break_during_capture
    def pod_compare(L, pod, full, station=None, add=None):
        # graph capture runs this eager op on buffers the (recorded, not executed) graph never filled; touching
        # /pod/shadow-reset after startup drops everything collected so far, so the stats cover real traffic only
        if os.path.exists("/pod/shadow-reset"):
            try:
                os.remove("/pod/shadow-reset")
            except OSError:
                pass
            _SH["cos"].clear(); _SH["rel"].clear(); _SH["n"] = 0
            _log("shadow stats reset")
        f = full.float()
        nf = f.norm(dim=-1)
        keep = nf > 0
        if os.environ.get("K3POD_SHADOW_DEBUG") == "1":
            npd = pod.norm(dim=-1)
            c_all = torch.nn.functional.cosine_similarity(pod, f, dim=-1)
            bad = torch.nonzero((c_all < 0.5) & keep).flatten()
            nz = int((~keep).sum())
            if bad.numel() or nz:
                b = bad[:8].tolist()
                ns = station.float().norm(dim=-1) if station is not None else None
                na = add.norm(dim=-1) if add is not None else None
                _log(f"SHDBG L{L} M={pod.shape[0]} bad={bad.numel()} f_zero={nz} idx={b} "
                     f"|f|={[round(float(nf[i]), 4) for i in b]} |pod|={[round(float(npd[i]), 4) for i in b]} "
                     f"|station|={[round(float(ns[i]), 4) for i in b] if ns is not None else '-'} "
                     f"|remote|={[round(float(na[i]), 4) for i in b] if na is not None else '-'}")
        if not bool(keep.any()):
            return
        cos = torch.nn.functional.cosine_similarity(pod[keep], f[keep], dim=-1)
        rel = (pod[keep] - f[keep]).norm(dim=-1) / nf[keep]
        _SH["cos"].setdefault(L, []).extend(cos.tolist())
        _SH["rel"].setdefault(L, []).extend(rel.tolist())
        _SH["n"] += 1
        if _SH["n"] % 200 == 0:
            import json as _j
            out = {str(k): {"n": len(v), "cos_min": min(v), "cos_mean": sum(v) / len(v),
                            "rel_max": max(_SH["rel"][k]), "rel_mean": sum(_SH["rel"][k]) / len(_SH["rel"][k])}
                   for k, v in sorted(_SH["cos"].items())}
            _j.dump(out, open(os.environ.get("K3POD_SHADOW_OUT", "/tmp/k3shadow.json"), "w"), indent=1)

    _OPS.update(send=pod_send, collect=pod_collect, compare=pod_compare)
    return _OPS


_PX = {}


def _proxy_init(device):
    """Pinned control block + proxy thread. Must run before CUDA-graph capture (first, eager forward)."""
    if _PX:
        return _PX
    import ctypes
    import torch
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as uva
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("k3pod: proxy must be initialised before graph capture")
    if POD.conns is None:
        POD.connect()
    NL, H, n = 93, int(os.environ.get("K3POD_HIDDEN", "3584")), len(PLAN.workers)
    t = dict(flags=torch.zeros((NL, 32), dtype=torch.int32, pin_memory=True),
             x=torch.zeros((NL, H), dtype=torch.bfloat16, pin_memory=True),
             ids=torch.zeros((NL, 16), dtype=torch.int32, pin_memory=True),
             w=torch.zeros((NL, 16), dtype=torch.float32, pin_memory=True),
             rows=torch.zeros((NL, n, H), dtype=torch.bfloat16, pin_memory=True))
    if PX_BATCH:
        t.update(xb=torch.zeros((NL, PXM, H), dtype=torch.bfloat16, pin_memory=True),
                 idsb=torch.full((NL, PXM, 16), -1, dtype=torch.int32, pin_memory=True),
                 wb=torch.zeros((NL, PXM, 16), dtype=torch.float32, pin_memory=True),
                 outb=torch.zeros((NL, PXM, H), dtype=torch.float32, pin_memory=True))
    _PX.update(host=t, dev={k: uva(v) for k, v in t.items()})
    lib = POD.clib
    lib.k3proxy_start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.k3proxy_served.restype = ctypes.c_longlong
    lib.k3proxy_work_s.restype = ctypes.c_double
    rc = lib.k3proxy_start(POD.handles, n, NL, H, POD.owner_tab.ctypes.data, t["flags"].data_ptr(),
                           t["x"].data_ptr(), t["ids"].data_ptr(), t["w"].data_ptr(), t["rows"].data_ptr(),
                           2_000_000, int(os.environ.get("K3POD_PROXY_CPU", "71")))
    if rc != 0:
        raise RuntimeError(f"k3proxy_start: {lib.k3_last_error().decode()}")
    if PX_BATCH:
        lib.k3proxy_set_batch.argtypes = [ctypes.c_void_p] * 4
        mx = lib.k3proxy_set_batch(t["xb"].data_ptr(), t["idsb"].data_ptr(), t["wb"].data_ptr(), t["outb"].data_ptr())
        if mx != PXM:
            raise RuntimeError(f"k3pod: libk3hop proxy batch max {mx} != {PXM}")
    cu = ctypes.CDLL("libcuda.so.1")
    for f in ("cuStreamWriteValue32_v2", "cuStreamWaitValue32_v2"):
        getattr(cu, f).argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        getattr(cu, f).restype = ctypes.c_int
    _PX.update(cu=cu, fbase=_PX["dev"]["flags"].data_ptr())

    import threading

    def stats():
        last = 0
        while True:
            time.sleep(30)
            sv = lib.k3proxy_served()
            if sv != last:
                _log(f"proxy: {sv} layer calls served; mean proxy work {lib.k3proxy_work_s() / max(sv - last, 1) * 1e6:.0f} us")
                last = sv
    threading.Thread(target=stats, daemon=True).start()
    _log(f"proxy started: {n} workers, control block pinned, graph-resident expert step ON"
         f"{f' (1..{PXM} tokens)' if PX_BATCH else ' (1 token)'}")
    return _PX


def _proxy_ok(M):
    return M == 1 or (PX_BATCH and 1 < M <= PXM)


def _proxy_issue(L, x, ids, w):
    """Captured: stage the token(s) for the proxy and raise ready[L] = M (constant per captured graph size)."""
    import torch
    d = _PX["dev"]
    M = x.shape[0]
    if M == 1:
        d["x"][L:L + 1].copy_(x)
        d["ids"][L:L + 1].copy_(ids)
        d["w"][L:L + 1].copy_(w)
    else:
        d["xb"][L, :M].copy_(x)
        d["idsb"][L, :M].copy_(ids)
        d["wb"][L, :M].copy_(w)
    s = torch.cuda.current_stream().cuda_stream
    rc = _PX["cu"].cuStreamWriteValue32_v2(s, _PX["fbase"] + (L * 32) * 4, M, 0)
    if rc != 0:
        raise RuntimeError(f"cuStreamWriteValue32 rc={rc}")


def _proxy_finish(L, add):
    """Captured: wait for done[L], reset it, sum the workers' rows into add[:1]."""
    import torch
    s = torch.cuda.current_stream().cuda_stream
    cu, fb = _PX["cu"], _PX["fbase"]
    rc = cu.cuStreamWaitValue32_v2(s, fb + (L * 32 + 16) * 4, 1, 1)          # CU_STREAM_WAIT_VALUE_EQ
    rc |= cu.cuStreamWriteValue32_v2(s, fb + (L * 32 + 16) * 4, 0, 0)
    if rc != 0:
        raise RuntimeError(f"cuStreamWait/WriteValue32 rc={rc}")
    M = add.shape[0]
    if M == 1:
        torch.sum(_PX["dev"]["rows"][L].float(), dim=0, keepdim=True, out=add[:1])
    else:
        add.copy_(_PX["dev"]["outb"][L, :M])


_ADD = {}
_DUMP = {"rows": [], "chunk": 0}


def _dump(L, x, ids, w, station, add):
    import numpy as np
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("k3pod: K3POD_DUMP_DIR needs --enforce-eager")
    _DUMP["rows"].append((L, x[0].float().cpu().numpy().astype(np.float32), ids[0].cpu().numpy().astype(np.int32),
                          w[0].float().cpu().numpy(), station[0].float().cpu().numpy(), add[0].float().cpu().numpy()))
    if len(_DUMP["rows"]) >= 92 * 32:
        _dump_flush()


def _dump_flush():
    import numpy as np
    r = _DUMP["rows"]
    if not r:
        return
    os.makedirs(DUMP_DIR, exist_ok=True)
    np.savez(os.path.join(DUMP_DIR, f"chunk{_DUMP['chunk']:04d}.npz"),
             layer=np.array([v[0] for v in r], np.int16), x=np.stack([v[1] for v in r]).astype(np.float32),
             ids=np.stack([v[2] for v in r]), w=np.stack([v[3] for v in r]),
             station=np.stack([v[4] for v in r]), remote=np.stack([v[5] for v in r]))
    _log(f"dump: wrote chunk {_DUMP['chunk']} ({len(r)} layer-tokens)")
    _DUMP["chunk"] += 1
    _DUMP["rows"] = []


def _add_buf(M, H, device):
    import torch
    b = _ADD.get("buf")
    if b is None:
        b = _ADD["buf"] = torch.empty((MAX_REMOTE_M, H), dtype=torch.float32, device=device)
    return b[:M]


def _attach(moe, layer_idx: int, n_global: int):
    import torch
    runner = moe.experts
    router = runner.router
    routed = runner.routed_experts
    router.global_num_experts = n_global
    router.e_score_correction_bias = moe.gate.e_score_correction_bias
    cfg = runner.moe_config
    try:
        cfg.should_defer_moe_finalize = lambda m: False
    except Exception as ex:  # noqa: BLE001
        _log("could not disable deferred finalize:", ex)

    table = torch.full((n_global + 1,), -1, dtype=torch.int32)   # [n_global] = sink for out-of-range ids
    for g, s in PLAN.local[layer_idx].items():
        table[g] = s
    state = {"table": table, "routed": None}
    L = layer_idx

    orig_sel = router.select_experts

    def select_experts(hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        w, ids = orig_sel(hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids)
        if state["table"].device != ids.device:
            state["table"] = state["table"].to(ids.device)
        M = hidden_states.shape[0]
        if PROXY and not _PX:
            _proxy_init(hidden_states.device)
        if M <= MAX_REMOTE_M:
            state["routed"] = (hidden_states, ids, w)
        else:
            state["routed"] = None
            POD.skipped += 1
            if POD.skipped <= 3:
                _log(f"layer {L}: M={M} > {MAX_REMOTE_M}, remote share skipped (profile/dummy run)")
        idx = ids.long()
        idx = torch.where((idx < 0) | (idx >= n_global), torch.full_like(idx, n_global), idx)
        return w, state["table"][idx].to(ids.dtype)

    router.select_experts = select_experts

    orig_fm = routed.forward_modular

    def forward_modular(*args, **kwargs):
        routed_in = state["routed"]
        state["routed"] = None
        if routed_in is not None and PROXY and _proxy_ok(routed_in[0].shape[0]):
            _proxy_issue(L, *routed_in)                   # captured: workers start now, no graph break
            out = orig_fm(*args, **kwargs)                # captured: the Station's own experts
            add = _add_buf(routed_in[0].shape[0], out.shape[1], out.device)
            _proxy_finish(L, add)                         # captured: in-graph wait on done[L]
            return (out.float() + add).to(out.dtype)
        ops = _ops() if routed_in is not None else None
        if routed_in is not None:
            ops["send"](L, *routed_in)                     # eager: D2H + post sends (workers start now)
        tm = TIME and routed_in is not None and routed_in[0].shape[0] > 8
        if tm:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
        out = orig_fm(*args, **kwargs)                    # captured: the Station's own experts
        if tm:
            e1.record(); e1.synchronize()
            _TM["n"] += 1; _TM["ms"] += e0.elapsed_time(e1); _TM["tok"] += routed_in[0].shape[0]
            if _TM["n"] % 920 == 0:
                _log(f"TIME station experts (M>8): {_TM['ms'] / _TM['n']:.3f} ms per layer call, "
                     f"mean M {_TM['tok'] / _TM['n']:.0f}, over {_TM['n']} calls")
        if routed_in is None:
            return out
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(f"k3pod: expected a finalized tensor from forward_modular, got {type(out)}")
        add = _add_buf(out.shape[0], out.shape[1], out.device)
        ops["collect"](L, add)                            # eager: wait for workers, H2D into a static buffer
        if DUMP_DIR and out.shape[0] == 1:
            _dump(L, routed_in[0], routed_in[1], routed_in[2], out, add)
        return (out.float() + add).to(out.dtype)          # captured

    routed.forward_modular = forward_modular


_SH = {"n": 0, "cos": {}, "rel": {}}


def _attach_shadow(moe, layer_idx: int, n_global: int):
    import numpy as np
    import torch
    runner = moe.experts
    router, routed = runner.router, runner.routed_experts
    try:
        runner.moe_config.should_defer_moe_finalize = lambda m: False
    except Exception:  # noqa: BLE001
        pass
    tab = torch.full((n_global + 1,), -1, dtype=torch.int32)
    for g in PLAN.local[layer_idx]:
        tab[g] = g                      # all experts resident: slot == global id
    state = {"tab": tab, "r": None}
    L = layer_idx
    orig_sel, orig_fm = router.select_experts, routed.forward_modular

    def select_experts(hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        w, ids = orig_sel(hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids)
        if PROXY and not _PX:
            _proxy_init(hidden_states.device)
        state["r"] = (hidden_states, ids, w) if hidden_states.shape[0] <= MAX_REMOTE_M else None
        return w, ids

    def forward_modular(*args, **kwargs):
        full = orig_fm(*args, **kwargs)                                   # captured
        r = state["r"]; state["r"] = None
        if r is None:
            return full
        x, ids, w = r
        if state["tab"].device != ids.device:
            state["tab"] = state["tab"].to(ids.device)
        ops = _ops()
        px = PROXY and _proxy_ok(x.shape[0])
        if px:
            _proxy_issue(L, x, ids, w)                                    # captured (proxy path)
        else:
            ops["send"](L, x, ids, w)                                     # eager break
        idx = ids.long().clamp(0, n_global)
        kw = dict(kwargs); kw["topk_ids"] = state["tab"][idx].to(ids.dtype)
        station = orig_fm(*args, **kw)                                    # captured
        add = _add_buf(full.shape[0], full.shape[1], full.device)
        if px:
            _proxy_finish(L, add)                                         # captured (proxy path)
        else:
            ops["collect"](L, add)                                        # eager break
        pod = station.float() + add                                       # captured
        ops["compare"](L, pod, full, station, add)                        # eager break
        return full

    router.select_experts = select_experts
    routed.forward_modular = forward_modular


def install_model_patch(module):
    """Called when vllm.models.kimi_k3.nvidia.model is imported."""
    import copy
    import torch
    from torch import nn
    KimiMoE = module.KimiMoE
    orig_init = KimiMoE.__init__

    def __init__(self, config, vllm_config, quant_config=None, prefix="", layer_idx=0, **kw):
        n_global = config.num_experts
        c2 = copy.copy(config)
        c2.num_experts = n_global if SHADOW else PLAN.n_local
        orig_init(self, c2, vllm_config, quant_config, prefix, layer_idx, **kw)
        G = type(self.gate)
        self.gate = G(input_size=config.hidden_size, output_size=n_global, bias=False,
                      out_dtype=torch.float32, prefix=f"{prefix}.gate")
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(n_global, dtype=torch.float32))
        if self.use_mega_moe:
            raise RuntimeError("k3pod supports the modular (non-MegaMoE) path only")
        if SHADOW:
            _attach_shadow(self, layer_idx, n_global)
        else:
            _attach(self, layer_idx, n_global)
        if layer_idx == PLAN.layers[0]:
            _log(f"KimiMoE patched: {PLAN.n_local} local experts/layer, gate {n_global}, "
                 f"{len(PLAN.workers)} workers")

    KimiMoE.__init__ = __init__

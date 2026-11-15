# TEST INSTRUMENT - NEVER THE SERVED FILE. Published copy (2026-11-15) of the shadow-compare build of
# fastpath/w1attn/integrate.py (milo-takeover/results/H/integrate_shadow_H.DRAFT.py). On the box it was copied over
# serve-bisect/tree/fastpath/w1attn/integrate.py for the shadow boots only (run_shadow_h.sh, run_shadow_i.sh,
# run_shadow_lc2.sh) and rolled back after each: every W1 decode-attention call also ran the stock FlashMLA op, the
# model consumed the stock op's output, and the two output buffers were compared byte for byte on the device.
# The integrate.py that serves Pg62 is the W13 file (sha256 bc87880f...), produced by
# patches/garner-8043a681/w1-exact/w13-fp8-compressed-records.patch. The docstring below is the original file's;
# its 'bit-identical to the stock op' sentence is the claim this instrument tested - it did not hold for Garner's
# shipped w1_attn_ext.so (70 of 17,797 window-only calls differed by one data byte, shadow-h1).
"""vLLM integration: DeepseekV4MegaAttnAttention's decode attention on W1's bitwise kernels.

Replaced call: the decode branch of ``DeepseekV4MegaAttnAttention.forward_mqa`` in
``vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`` -> ``_forward_decode_mega(q, positions_int32,
flashmla_metadata, swa_metadata, out)`` -> ``torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode``. W1 takes
the same tensors and writes the same (fp8 data, packed ue8m0 scale) buffers, bit-identical to the stock op.

Installed per call when all hold (else the previously installed ``_forward_decode_mega`` runs untouched):
  * layer class in {sparse, swa, draft}: sparse = compressed layers (compress_ratio 1 / 2), swa = window-only target
    layers (0-1), draft = window-only drafter layers (layer_id >= 40);
  * 1 <= num_decode_tokens <= the class's width limit. Sparse FP4 layers: up to 16 on the bitwise-parallel kernel,
    then up to 76 on the 2-CTA cluster kernel. Opt-in compressed FP8 is intentionally bounded to 16 because CX's
    later-pair loader is FP4-sized and requires a separately validated port.
    Window-only layers: up to 24 (swa and draft) on the split-dims kernel;
  * TP1 shapes: 64 padded heads, bf16 q, MXFP8 (528 B) window cache, NVFP4 (288 B) compressed cache, or separately
    enabled MXFP8 (528 B) compressed cache, <= 10 blocks of 64 keys per token. Failures run stock inline.
An installed layer prepares W1 in its constructor (before any CUDA-graph capture). ``stats()`` counts calls per path;
``capture_summary()`` (logged after GPUModelRunner.capture_model) gives the W1 / stock split of captured calls.
"""

import sys

import torch

from . import ops
from .formats import (CacheFormat, MAX_SQ, MAX_SQ_CX, MAX_SQ_SWA,
                      cache_layout_is_valid, classify_compressed_layout, compressed_fp8_enabled,
                      select_mode)

LAYERS = {"sparse", "swa", "draft"}
NUM_LAYERS = 40
MAX_BLOCKS = 10
COMPRESSED_FP8 = compressed_fp8_enabled()
_scratch: dict = {}
_stats = {"w1": 0, "w1_incumbent": 0, "w1_fp8_candidate": 0, "stock": 0, "stock_shape": 0}
_capture: dict = {}  # (width, stock/incumbent/candidate path) -> calls recorded into CUDA graphs
_logged: set = set()


# --- Milo bisect mask (2026-11-14): restrict the FP8 candidate by target layer id and row width. -----------------
# File next to this module: {"layers": [ints] | null, "widths": [ints] | null, "label": str}. null = all; missing
# file = all (the W13 behaviour). A masked-out FP8 call takes the same stock fallback as an unsupported shape.
def _load_mask():
    import json, os, re
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp8_mask.json")
    if not os.path.exists(path):
        return None, None, "absent"
    d = json.loads(open(path).read())
    layers = None if d.get("layers") is None else frozenset(int(x) for x in d["layers"])
    widths = None if d.get("widths") is None else frozenset(int(x) for x in d["widths"])
    global _MASK_PDL, _MASK_PLAN
    _MASK_PDL = bool(d.get("pdl", True))
    _MASK_PLAN = d.get("fp8_plan") or None
    global _MASK_EXTRA
    _MASK_EXTRA = d.get("fp8_extra") or None
    # E1 shadow options (only read when fp8_extra == "shadow")
    global _SHADOW_WIDTHS, _SHADOW_RECORDS, _SHADOW_VALID_ONLY, _SHADOW_DIR
    _SHADOW_WIDTHS = frozenset(int(x) for x in d.get("shadow_widths", [6, 8]))
    _SHADOW_RECORDS = bool(d.get("shadow_records", True))
    _SHADOW_VALID_ONLY = bool(d.get("shadow_valid_only", True))
    _SHADOW_DIR = str(d.get("shadow_dir", "/prof/shadow"))
    # desk H: shadow every incumbent-extension call too (comp_format NONE/FP4 with select_mode != stock)
    global _SHADOW_INCUMBENT
    _SHADOW_INCUMBENT = d.get("shadow_incumbent", False) is True
    return layers, widths, str(d.get("label", "?"))


_MASK_EXTRA = None  # "stock_after": W1 then stock into the same out; "stock_before": stock then W1 (FP8 calls only)
#                     "shadow": W1 into alt buffers, stock into out, device-side byte compare (E1, see below)
_SHADOW_WIDTHS = frozenset((6, 8))  # row widths that get a first-mismatch snapshot (stats are kept for every width)
_SHADOW_RECORDS = True  # snapshot also the cache records the call indexed (exact call-time cache bytes)
_SHADOW_VALID_ONLY = True  # snapshot trigger: mismatch on a row with swa_lens > 0 (stats count all rows regardless)
_SHADOW_DIR = "/prof/shadow"
_SHADOW_INCUMBENT = False  # desk H mask key "shadow_incumbent": true -> incumbent W1 into alt, stock into out, compare
_MASK_PLAN = None  # FP8-candidate calls only; e.g. "bitwise_seq" replaces the bitwise-parallel plan
_MASK_PDL = True  # FP8-candidate calls only; False = launch without programmatic dependent launch
_MASK_LAYERS, _MASK_WIDTHS, _MASK_LABEL = _load_mask()
_mask_seen: dict = {}  # (layer, width) -> allowed, recorded at capture/eager call time


def _mask_layer_id(self) -> int:
    import re
    m = re.search(r"layers\.(\d+)\.", str(getattr(self, "prefix", "")))
    if m:
        return int(m.group(1))
    return int(getattr(self, "layer_id", -1))


def _mask_ok(self, n: int) -> bool:
    lid = _mask_layer_id(self)
    ok = ((_MASK_LAYERS is None or lid in _MASK_LAYERS) and (_MASK_WIDTHS is None or n in _MASK_WIDTHS))
    _mask_seen[(lid, n)] = ok
    return ok


# --- Milo E1 shadow compare (2026-11-14): fp8_extra == "shadow". -------------------------------------------------
# Per FP8-candidate call: W1 writes alt buffers (same shape/dtype/strides as out.data / out.scale), the stock op
# writes the real out (the model stays on stock values), then a device-side byte compare accumulates per call site
# (a site = (target layer id, row width), static index assigned when first seen, normally during graph capture).
# Graph-safe by construction: torch ops only, fixed shapes, no .item()/sync/printing, no host branch on device
# values, no host->device copies, no init kernels on buffers allocated under capture (they would be replayed).
# Readback only through sentinel files in _SHADOW_DIR polled by a daemon thread (RESET / DUMP); see _shadow_poll.
# desk H (2026-11-14): mask key "shadow_incumbent": true applies the same shadow to every call the non-shadow code sends
# to the incumbent extension (comp_format NONE/FP4 with select_mode != stock: window-only target layers and drafter on
# "ss"; FP4 "bitwise"/"cx" if an FP4 cache were present). Sites carry path/mode/prefix; RESET/DUMP run under
# torch.inference_mode() in the poller thread. Absent key = E behaviour.
import json as _json
import os as _os
import threading as _threading
import time as _time

# desk H: a site is (path, module prefix, layer id, width). Expected upper bound with shadow_incumbent on the served
# FP8 lane: FP8 = 38 compressed layers x widths 1..16 = 608; incumbent = window-only modules (target layers 0, 1 and
# the drafter, logged as model.layers.40.attn) x widths 1..24 = 72 for one drafter module -> 680 <= 1024. Overflow is
# still counted (total and per path) and reported in stats.json and the dump log line.
SH_MAX_SITES = 1024
SH_ROWS = MAX_SQ_SWA if _SHADOW_INCUMBENT else 16  # incumbent window-only calls go to 24 rows; FP8-only stays 16
SH_CALLS, SH_ANY, SH_VALID_ANY, SH_DATA, SH_SCALE, SH_FIRST_SEQ, SH_FIRST_CALL = range(7)  # int64 stats columns
SH_ROW_DATA = 8  # [16] mismatching data bytes per row
SH_ROW_SCALE = SH_ROW_DATA + SH_ROWS  # [16] mismatching scale bytes per row
SH_ROW_CALLS = SH_ROW_SCALE + SH_ROWS  # [16] calls in which this row had any mismatching byte
SH_NCOL = SH_ROW_CALLS + SH_ROWS


class _ShadowState:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        # allocated in prepare(), i.e. before any capture: plain zeros are safe here and never re-run on replay
        self.stats = torch.zeros((SH_MAX_SITES, SH_NCOL), dtype=torch.int64, device=device)
        self.flags = torch.zeros(SH_MAX_SITES, dtype=torch.bool, device=device)
        self.seq = torch.zeros(1, dtype=torch.int64, device=device)  # global shadow-call counter (cross-site order)
        self.sites: list = []
        self.site_of: dict = {}
        self.alt: dict = {}
        self.overflow = 0
        self.overflow_by_path = {"fp8": 0, "incumbent": 0}  # desk H
        self.logged_types = False
        self.logged_incumbent = False  # desk H: first incumbent shadow call logged separately


_shadow: dict = {}  # device -> _ShadowState (model-building process only)
_shadow_ready = False  # True once capture_model returned (or DS41_SHADOW_EAGER_OK=1 for an eager lane)
_shadow_thread = None


def _shadow_init(device: torch.device) -> None:
    global _shadow_thread
    if device not in _shadow:
        _shadow[device] = _ShadowState(device)
    if _shadow_thread is None:
        _shadow_thread = _threading.Thread(target=_shadow_loop, name="w1-fp8-shadow", daemon=True)
        _shadow_thread.start()
        _log(f"FP8_SHADOW poller started pid={_os.getpid()} dir={_SHADOW_DIR} (acts only after capture_model)")


def _shadow_alt(S: _ShadowState, out):
    """Alternate output pair with out's exact shape/dtype/strides. out.scale is an MN-major [:n] slice whose rows
    are padded to ceil4(num_tokens): not dense for n % 4 != 0, so torch.empty_like would silently go contiguous;
    empty_strided keeps the layout the W1 extension is given in the real call. No init kernel (capture-safe)."""
    key = (tuple(out.data.shape), tuple(out.data.stride()), tuple(out.scale.shape), tuple(out.scale.stride()))
    v = S.alt.get(key)
    if v is None:
        v = (torch.empty_strided(out.data.shape, out.data.stride(), dtype=out.data.dtype, device=out.data.device),
             torch.empty_strided(out.scale.shape, out.scale.stride(), dtype=out.scale.dtype, device=out.scale.device))
        S.alt[key] = v
    return v


def _gather_records(cache: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """[rows, width, record] bytes of the slots in idx (global slot = page * page_tokens + offset). Negative or
    out-of-range slots are clamped to a real slot (bytes are junk there; the repro uses only valid entries)."""
    pt = cache.shape[1]
    s = idx.to(torch.int64).clamp(0, cache.shape[0] * pt - 1)
    return cache[torch.div(s, pt, rounding_mode="floor"), s % pt]


def _snap_spec(n, q, positions, swa_idx, swa_lens, extra_idx, extra_len, out, swa_cache, comp):
    spec = {
        "q": (tuple(q.shape), q.dtype),
        "positions": (tuple(positions.shape), positions.dtype),
        "swa_idx": (tuple(swa_idx.shape), swa_idx.dtype),
        "swa_lens": (tuple(swa_lens.shape), swa_lens.dtype),
        "stock_data": ((n, out.data[0].numel()), torch.uint8),
        "stock_scale": (tuple(out.scale.shape), out.scale.dtype),  # stored in logical (contiguous) order
        "w1_data": ((n, out.data[0].numel()), torch.uint8),
        "w1_scale": (tuple(out.scale.shape), out.scale.dtype),
    }
    if comp is not None:  # desk H: window-only (incumbent ss) calls have no compressed cache / top-k
        spec["extra_idx"] = (tuple(extra_idx.shape), extra_idx.dtype)
        spec["extra_len"] = (tuple(extra_len.shape), extra_len.dtype)
    if _SHADOW_RECORDS:
        spec["swa_records"] = ((n, swa_idx.shape[-1], swa_cache.shape[-1]), torch.uint8)
        if comp is not None:
            spec["extra_records"] = ((n, extra_idx.shape[-1], comp.shape[-1]), torch.uint8)
    return spec


def _shadow_site(S: _ShadowState, self, n: int, spec: dict, swa_cache, comp, plan, capturing: bool,
                 path: str = "fp8", mode: str = "bitwise", comp_format=None):
    lid = _mask_layer_id(self)
    prefix = str(getattr(self, "prefix", "?"))
    # desk H: the path and the module prefix are part of the key, so target layer 0/1 and drafter modules never share
    # a site. For FP8 calls this is the same partition as E's (layer id, width) key: one prefix per compressed layer.
    key = (path, prefix, lid, n)
    i = S.site_of.get(key)
    if i is None:
        if len(S.sites) >= SH_MAX_SITES:
            return None
        i = len(S.sites)
        S.site_of[key] = i
        snap = None
        if n in _SHADOW_WIDTHS:
            # torch.empty only: an init kernel recorded under capture would re-zero the snapshot on every replay
            snap = {k: torch.empty(shape, dtype=dt, device=S.device) for k, (shape, dt) in spec.items()}
        S.sites.append({"index": i, "layer": lid, "width": n, "prefix": prefix,
                        "compress_ratio": int(getattr(self, "compress_ratio", 0) or 0), "layer_obj": self,
                        "swa": swa_cache, "comp": comp, "snap": snap, "plan": str(plan), "pdl": bool(_MASK_PDL),
                        "first_seen_under_capture": capturing, "layouts": set(),
                        "path": path, "mode": str(mode), "layer_class": _layer_class(self),
                        "comp_format": "?" if comp_format is None else CacheFormat(comp_format).name})
        if path == "incumbent":
            S.sites[i]["pdl"] = True  # the incumbent call always launches with pdl=True (the mask pdl is FP8-only)
    site = S.sites[i]
    site["swa"], site["comp"] = swa_cache, comp  # the live cache views this call used (stable after KV alloc)
    return i


def _keep(buf: torch.Tensor, cur: torch.Tensor, take: torch.Tensor) -> None:
    buf.copy_(torch.where(take, cur.reshape(buf.shape), buf))


def _shadow_compare(S: _ShadowState, i: int, n: int, out, alt_d, alt_s, swa_lens, cur: dict) -> None:
    od = out.data.view(torch.uint8).reshape(n, -1)
    ad = alt_d.view(torch.uint8).reshape(n, -1)
    osc = out.scale.contiguous()  # logical order (the MN-major view cannot be byte-viewed in place)
    asc = alt_s.contiguous()
    d_rows = (od != ad).sum(1)  # int64 [n]
    s_rows = (osc.view(torch.uint8).reshape(n, -1) != asc.view(torch.uint8).reshape(n, -1)).sum(1)
    row_mis = (d_rows + s_rows) > 0
    valid = swa_lens.reshape(-1)[:n] > 0
    any_all = row_mis.any()
    any_valid = (row_mis & valid).any()
    trig = any_valid if _SHADOW_VALID_ONLY else any_all
    take = trig & ~S.flags[i]  # materialized before the flag update below (stream order)
    st = S.stats[i]
    S.seq.add_(1)
    st[SH_CALLS:SH_CALLS + 1].add_(1)
    st[SH_ANY:SH_ANY + 1].add_(any_all.to(torch.int64))
    st[SH_VALID_ANY:SH_VALID_ANY + 1].add_(any_valid.to(torch.int64))
    st[SH_DATA:SH_DATA + 1].add_(d_rows.sum())
    st[SH_SCALE:SH_SCALE + 1].add_(s_rows.sum())
    m = min(n, SH_ROWS)
    st[SH_ROW_DATA:SH_ROW_DATA + m].add_(d_rows[:m])
    st[SH_ROW_SCALE:SH_ROW_SCALE + m].add_(s_rows[:m])
    st[SH_ROW_CALLS:SH_ROW_CALLS + m].add_(row_mis[:m].to(torch.int64))
    st[SH_FIRST_SEQ:SH_FIRST_SEQ + 1].copy_(torch.where(take, S.seq, st[SH_FIRST_SEQ:SH_FIRST_SEQ + 1]))
    st[SH_FIRST_CALL:SH_FIRST_CALL + 1].copy_(
        torch.where(take, st[SH_CALLS:SH_CALLS + 1], st[SH_FIRST_CALL:SH_FIRST_CALL + 1]))
    snap = S.sites[i]["snap"]
    if snap is not None:
        cur = dict(cur, stock_data=od, stock_scale=osc, w1_data=ad, w1_scale=asc)
        if all(k in cur and cur[k].numel() == snap[k].numel() and cur[k].dtype == snap[k].dtype for k in snap):
            for k, buf in snap.items():
                _keep(buf, cur[k], take)
    S.flags[i:i + 1].logical_or_(trig)


def _shadow_call(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out, comp, swa_idx,
                 scratch, mode, capturing, comp_format=CacheFormat.FP8) -> None:
    """FP8 (E1, unchanged) or, desk H, incumbent (comp_format NONE/FP4, candidate_fp8=False). In both cases W1 gets
    exactly the arguments the non-shadow path would pass it, only out.data/out.scale are swapped for the alt pair."""
    S = _shadow.get(q.device)
    if S is None:  # prepare() never ran shadow init on this device: behave exactly like the stock fallback
        _stock(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out)
        return
    fp8 = comp_format == CacheFormat.FP8
    path = "fp8" if fp8 else "incumbent"
    n = q.shape[0]
    swa_cache = _u8(self.swa_cache_layer.kv_cache)
    swa_lens = swa_metadata.decode_swa_lens
    alt_d, alt_s = _shadow_alt(S, out)
    plan = (_MASK_PLAN or mode) if fp8 else mode  # same plan rule as the non-shadow call (mask plan is FP8-only)
    pdl = _MASK_PDL if fp8 else True  # same pdl rule as the non-shadow call
    if (not S.logged_types) if fp8 else (not S.logged_incumbent):
        if fp8:
            S.logged_types = True
        else:
            S.logged_incumbent = True
        tag = "" if fp8 else f"path=incumbent mode={mode} comp_format={CacheFormat(comp_format).name} "
        cshape = "None" if comp is None else f"{tuple(comp.shape)}/{tuple(comp.stride())}"
        eshape = "None" if extra_idx is None else f"{tuple(extra_idx.shape)}"
        lshape = "None" if extra_len is None else f"{tuple(extra_len.shape)}"
        _log(f"FP8_SHADOW first call {tag}{getattr(self, 'prefix', '?')} n={n} out={type(out).__name__} "
             f"data={out.data.dtype}{tuple(out.data.shape)}/{tuple(out.data.stride())} "
             f"scale={out.scale.dtype}{tuple(out.scale.shape)}/{tuple(out.scale.stride())} "
             f"swa={tuple(swa_cache.shape)}/{tuple(swa_cache.stride())} comp={cshape} "
             f"swa_idx={tuple(swa_idx.shape)} extra_idx={eshape} swa_lens={tuple(swa_lens.shape)} "
             f"extra_len={lshape} plan={plan} pdl={pdl} capturing={capturing}")
    # 1. W1 into the alternate buffers (same call as the W13 path, only the outputs differ)
    ops.attn_decode(q, positions_int32, self.rotary_emb.cos_sin_cache, self.attn_sink, self.scale, swa_cache, swa_idx,
                    swa_lens, comp, extra_idx, extra_len, alt_d, alt_s, scratch, plan=plan,
                    comp_format=comp_format, candidate_fp8=fp8, pdl=pdl)
    # 2. stock into the real out (same order as the oracle-clean stock_after arm: W1 first, stock second)
    _stock(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out)
    # 3. device-side compare + sticky first-mismatch snapshot
    spec = _snap_spec(n, q, positions_int32, swa_idx, swa_lens, extra_idx, extra_len, out, swa_cache, comp)
    i = _shadow_site(S, self, n, spec, swa_cache, comp, plan, capturing, path=path, mode=mode,
                     comp_format=comp_format)
    if i is None:
        S.overflow += 1
        S.overflow_by_path[path] += 1
        return
    S.sites[i]["layouts"].add((tuple(out.data.shape), tuple(out.data.stride()), tuple(out.scale.shape),
                               tuple(out.scale.stride())))  # alt_* share these by construction (_shadow_alt)
    cur = {"q": q, "positions": positions_int32, "swa_idx": swa_idx, "swa_lens": swa_lens}
    if comp is not None:
        cur["extra_idx"], cur["extra_len"] = extra_idx, extra_len
    if _SHADOW_RECORDS and S.sites[i]["snap"] is not None:
        cur["swa_records"] = _gather_records(swa_cache, swa_idx)
        if comp is not None:
            cur["extra_records"] = _gather_records(comp, extra_idx)
    _shadow_compare(S, i, n, out, alt_d, alt_s, swa_lens, cur)


def _by_path(S: _ShadowState) -> dict:
    out = {}
    for s in S.sites:
        p = out.setdefault(s["path"], {"sites": 0, "snapshot_sites": 0})
        p["sites"] += 1
        p["snapshot_sites"] += s["snap"] is not None
    return out


def _shadow_summary() -> str:
    parts = []
    for dev, S in _shadow.items():
        snap_b = sum(t.numel() * t.element_size() for s in S.sites if s["snap"] for t in s["snap"].values())
        alt_b = sum(t.untyped_storage().nbytes() for v in S.alt.values() for t in v)
        parts.append(f"{dev}: sites={len(S.sites)} snapshot_sites={sum(1 for s in S.sites if s['snap'])} "
                     f"snapshot_bytes={snap_b} alt_bytes={alt_b} overflow={S.overflow} "
                     f"widths={sorted({s['width'] for s in S.sites})} layers={len({s['layer'] for s in S.sites})}"
                     + (f" by_path={_by_path(S)} overflow_by_path={S.overflow_by_path}" if _SHADOW_INCUMBENT else ""))
    return "; ".join(parts) or "no shadow state"


# ---- readback (daemon thread; never touches the hot path) -------------------------------------------------------
def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def _shadow_loop() -> None:
    while True:
        _time.sleep(2.0)
        try:
            _shadow_poll()
        except Exception as exc:  # never kill the thread; the sentinel is renamed so the error is not retried
            _log(f"FP8_SHADOW poll error {type(exc).__name__}: {exc}"[:600])


def _shadow_poll() -> None:
    d = _SHADOW_DIR
    reset, dump = _os.path.join(d, "RESET"), _os.path.join(d, "DUMP")
    has_reset, has_dump = _os.path.exists(reset), _os.path.exists(dump)
    if not (has_reset or has_dump):
        return
    ready = _shadow_ready or _os.environ.get("DS41_SHADOW_EAGER_OK") == "1"
    if not ready or not _shadow:
        if not _os.path.exists(_os.path.join(d, "NOT_READY")):
            _write(_os.path.join(d, "NOT_READY"), f"capture_model not finished or no shadow state; pid {_os.getpid()}\n")
        return
    for name, path, fn in (("RESET", reset, _shadow_reset), ("DUMP", dump, _shadow_dump)):
        if not _os.path.exists(path):
            continue
        for old in (name + ".done", name + ".error"):
            if _os.path.exists(_os.path.join(d, old)):
                _os.remove(_os.path.join(d, old))
        try:
            msg = fn()
        except Exception as exc:
            _os.replace(path, _os.path.join(d, name + ".error"))
            _write(_os.path.join(d, name + ".error"), f"{type(exc).__name__}: {exc}\n")
            raise
        _os.remove(path)
        _write(_os.path.join(d, name + ".done"), msg + "\n")


def _shadow_reset() -> str:
    # desk H: run under inference mode. The stats/flags/seq tensors and the snapshot buffers are inference tensors
    # (allocated / first written under vLLM's inference_mode), and an in-place zero_() on an inference tensor outside
    # InferenceMode raises "Inplace update to inference tensor outside InferenceMode is not allowed" (the E1 second
    # RESET failure). inference_mode is thread-local: it only affects this poller thread, never the serving thread.
    # zero_() inside it is legal for inference tensors and for normal tensors alike, and these tensors have no autograd
    # history, so nothing about the values or the captured graphs (which hold the same storages) changes.
    with torch.inference_mode():
        for dev, S in _shadow.items():
            with torch.cuda.device(dev) if dev.type == "cuda" else _nullctx():
                _sync(dev)
                S.stats.zero_()
                S.flags.zero_()
                S.seq.zero_()
                for s in S.sites:
                    if s["snap"]:
                        for t in s["snap"].values():
                            t.zero_()
                _sync(dev)
    # overflow (host count of calls that never got a site) is cumulative since boot, as in E: sites persist across RESET
    msg = f"FP8_SHADOW reset at {_time.strftime('%F %T')}: {_shadow_summary()}"
    _log(msg)
    return msg


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _pages_of(cache: torch.Tensor, idx: torch.Tensor) -> dict:
    """Every page holding a nonnegative in-range slot of idx (a superset of the valid entries), raw, at dump time."""
    pt, npg = int(cache.shape[1]), int(cache.shape[0])
    flat = idx.reshape(-1).to(torch.int64)
    sel = flat[(flat >= 0) & (flat < npg * pt)]
    page_ids = torch.unique(torch.div(sel, pt, rounding_mode="floor"))  # sorted, cpu
    raw = cache.index_select(0, page_ids.to(cache.device)).to(device="cpu")
    return {"page_ids": page_ids, "pages": raw, "page_tokens": pt, "record_bytes": int(cache.shape[-1]),
            "num_pages": npg, "cache_shape": list(cache.shape), "cache_stride": list(cache.stride()),
            "cache_dtype": str(cache.dtype), "out_of_range_slots": int(((flat >= npg * pt)).sum())}


def _dump_site(S: _ShadowState, s: dict, row: list, path: str) -> dict:
    snap = {k: v.detach().to(device="cpu").clone() for k, v in s["snap"].items()}
    lay = s["layer_obj"]
    pages = {"swa": _pages_of(s["swa"], snap["swa_idx"])}
    if s["comp"] is not None and "extra_idx" in snap:  # desk H: window-only incumbent sites have no compressed cache
        pages["extra"] = _pages_of(s["comp"], snap["extra_idx"])
    cs = lay.rotary_emb.cos_sin_cache
    pos = snap["positions"].to(torch.int64).clamp(0, cs.shape[0] - 1)
    base, last = int(pos.min()), int(pos.max())
    if last - base + 1 > 65536:  # padded rows with junk positions: keep the table bounded, record the fact
        last = base + 65535
    payload = {
        "meta": {"site": s["index"], "layer": s["layer"], "width": s["width"], "prefix": s["prefix"],
                 "compress_ratio": s["compress_ratio"], "plan": s["plan"], "pdl": s["pdl"],
                 "n_wv_group": int(getattr(lay, "n_wv_group", 8)), "scale": float(lay.scale),
                 "layouts": [list(map(list, x)) for x in sorted(s["layouts"])],
                 "first_seq": row[SH_FIRST_SEQ], "first_call": row[SH_FIRST_CALL], "calls": row[SH_CALLS],
                 "records_in_snapshot": "swa_records" in snap, "valid_only_trigger": _SHADOW_VALID_ONLY,
                 "cos_sin_base": base, "cos_sin_shape": list(cs.shape), "cos_sin_dtype": str(cs.dtype),
                 "rows": s["width"],
                 "path": s["path"], "mode": s["mode"], "layer_class": s["layer_class"],
                 "comp_format": s["comp_format"], "candidate_fp8": s["path"] == "fp8"},
        "snap": snap,
        "pages": pages,
        "attn_sink": lay.attn_sink.detach().to(device="cpu").clone(),
        "cos_sin_rows": cs[base:last + 1].detach().to(device="cpu").clone(),
    }
    torch.save(payload, path)
    return {"swa_pages": int(pages["swa"]["page_ids"].numel()),
            "extra_pages": int(pages["extra"]["page_ids"].numel()) if "extra" in pages else 0}


def _shadow_dump() -> str:
    # desk H: inference mode for the same reason as RESET (thread-local to the poller; DUMP only reads the shadow
    # tensors, so this is for uniformity and so a future in-place op here cannot hit the RESET failure). torch.save of
    # the resulting CPU tensors works under inference mode (checked in test_shadow_cpu_H.py, torch 2.8 CPU).
    with torch.inference_mode():
        return _shadow_dump_body()


def _shadow_dump_body() -> str:
    d = _SHADOW_DIR
    k = 0
    while _os.path.exists(_os.path.join(d, f"dump-{k}")):
        k += 1
    final = _os.path.join(d, f"dump-{k}")
    tmp = final + ".partial"
    _os.makedirs(tmp, exist_ok=True)
    report = {"dump": k, "time": _time.strftime("%F %T"), "pid": _os.getpid(), "valid_only": _SHADOW_VALID_ONLY,
              "records": _SHADOW_RECORDS, "widths_snapshotted": sorted(_SHADOW_WIDTHS), "devices": {},
              "shadow_fp8": _MASK_EXTRA == "shadow" and COMPRESSED_FP8, "shadow_incumbent": _SHADOW_INCUMBENT}
    tot = {"sites": 0, "calls": 0, "calls_any": 0, "calls_valid": 0, "flagged": 0, "saved": 0}
    by_path = {}  # desk H: same counters split by path ("fp8" | "incumbent")
    by_group = {}  # desk H: (path, mode, layer_class, prefix) -> counters, for the runbook's per-module reading
    for dev, S in _shadow.items():
        with torch.cuda.device(dev) if dev.type == "cuda" else _nullctx():
            _sync(dev)
            stats = S.stats[:len(S.sites)].to(device="cpu").tolist()
            flags = S.flags[:len(S.sites)].to(device="cpu").tolist()
            sites = []
            for s, row, flag in zip(S.sites, stats, flags):
                w = min(s["width"], SH_ROWS)
                rec = {"site": s["index"], "layer": s["layer"], "width": s["width"], "prefix": s["prefix"],
                       "path": s["path"], "mode": s["mode"], "layer_class": s["layer_class"],
                       "comp_format": s["comp_format"],
                       "compress_ratio": s["compress_ratio"], "plan": s["plan"], "pdl": s["pdl"],
                       "calls": row[SH_CALLS], "calls_with_any_mismatch": row[SH_ANY],
                       "calls_with_valid_row_mismatch": row[SH_VALID_ANY], "mismatching_data_bytes": row[SH_DATA],
                       "mismatching_scale_bytes": row[SH_SCALE], "first_mismatch_seq": row[SH_FIRST_SEQ],
                       "first_mismatch_call": row[SH_FIRST_CALL], "flag": bool(flag),
                       "row_data_bytes": row[SH_ROW_DATA:SH_ROW_DATA + w],
                       "row_scale_bytes": row[SH_ROW_SCALE:SH_ROW_SCALE + w],
                       "row_mismatch_calls": row[SH_ROW_CALLS:SH_ROW_CALLS + w],
                       "snapshot_buffers": s["snap"] is not None, "snapshot_file": None,
                       "layouts": [list(map(list, x)) for x in sorted(s["layouts"])]}
                if flag and s["snap"] is not None:
                    if s["path"] == "fp8":  # E's file name, unchanged
                        fname = f"site-{s['index']:04d}-L{s['layer']:02d}-w{s['width']}.pt"
                    else:
                        fname = f"site-{s['index']:04d}-inc-{s['layer_class']}-L{s['layer']:02d}-w{s['width']}.pt"
                    rec.update(_dump_site(S, s, row, _os.path.join(tmp, fname)))
                    rec["snapshot_file"] = fname
                    tot["saved"] += 1
                grp = [tot, by_path.setdefault(s["path"], dict.fromkeys(tot, 0)),
                       by_group.setdefault((s["path"], s["mode"], s["layer_class"], s["prefix"]),
                                           dict.fromkeys(tot, 0))]
                for g in grp[1:]:
                    g["saved"] += rec["snapshot_file"] is not None
                for g in grp:
                    g["sites"] += 1
                    g["calls"] += row[SH_CALLS]
                    g["calls_any"] += row[SH_ANY] > 0
                    g["calls_valid"] += row[SH_VALID_ANY] > 0
                    g["flagged"] += bool(flag)
                sites.append(rec)
            sites.sort(key=lambda r: (r["first_mismatch_seq"] if r["flag"] else 1 << 62, r["layer"], r["width"]))
            report["devices"][str(dev)] = {"seq": int(S.seq.to(device="cpu")[0]), "overflow": S.overflow,
                                           "overflow_by_path": dict(S.overflow_by_path), "sites": sites}
    report["totals"] = tot
    report["totals_by_path"] = by_path
    report["totals_by_group"] = [dict(path=p, mode=m, layer_class=c, prefix=x, **v)
                                 for (p, m, c, x), v in sorted(by_group.items())]
    _write(_os.path.join(tmp, "stats.json"), _json.dumps(report, indent=1) + "\n")
    _os.replace(tmp, final)
    msg = (f"FP8_SHADOW dump-{k}: sites={tot['sites']} shadow_calls={tot['calls']} "
           f"sites_with_any_mismatch={tot['calls_any']} sites_with_valid_row_mismatch={tot['calls_valid']} "
           f"flagged={tot['flagged']} snapshots_saved={tot['saved']} dir={final}")
    if _SHADOW_INCUMBENT:
        msg += " by_path=" + " ".join(
            f"{p}:sites={v['sites']},calls={v['calls']},any={v['calls_any']},valid={v['calls_valid']},"
            f"flagged={v['flagged']}" for p, v in sorted(by_path.items())) + \
            " overflow=" + ",".join(f"{p}:{S.overflow_by_path[p]}" for S in _shadow.values() for p in ("fp8", "incumbent"))
    _log(msg)
    return msg


def stats() -> dict:
    return dict(_stats)


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH attn: {msg}\n")


def prepare(device: torch.device) -> ops.Scratch:
    """Everything W1 needs before CUDA-graph capture, once per device: the extension, the kernel's smem opt-in, and
    the zeroed scratch (one per device, shared by every layer: launches are stream-ordered and tags unique per
    launch). Called from the layer constructor, so the first decode forward under capture already runs W1."""
    s = _scratch.get(device)
    if s is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("W1 attn: scratch must be prepared before CUDA-graph capture")
        # The legacy extension remains the only path for NONE/FP4.  Prepare it
        # unconditionally, then prepare the isolated FP8 candidate as an add-on.
        ops._ext(candidate_fp8=False).w1_attn_prepare()
        if COMPRESSED_FP8:
            ops._ext(candidate_fp8=True).w1_attn_prepare()
        words = MAX_SQ * MAX_BLOCKS * (128 + 128 + 2048 + 1)  # tagged 64-bit words (stage_attn_par.cuh)
        with torch.cuda.device(device):
            s = ops.Scratch(
                part_o=torch.zeros(2 * words, dtype=torch.float32, device=device),
                part_ml=torch.zeros(1, dtype=torch.float32, device=device),
                flags=torch.zeros(1, dtype=torch.int32, device=device),
                epoch=torch.zeros(2, dtype=torch.int32, device=device),
            )
        _scratch[device] = s
        _log(f"prepared on {device}: scratch {2 * words * 4 / 2**20:.1f} MiB; layers={sorted(LAYERS)} "
             f"max_sq sparse={MAX_SQ} cluster={MAX_SQ_CX} swa={MAX_SQ_SWA}; "
             f"compressed_fp8={COMPRESSED_FP8}")
    if ((_MASK_EXTRA == "shadow" and COMPRESSED_FP8) or _SHADOW_INCUMBENT) and device not in _shadow:
        _shadow_init(device)  # stats/flags allocated here, before capture; poller thread in this (model) process
    return s


def _capture_note(width: int, path: str, prefix: str, cls_: str) -> None:
    """Count calls recorded into CUDA graphs per (width, path); log the first W1 layer of each captured width."""
    key = (width, path)
    first = key not in _capture
    _capture[key] = _capture.get(key, 0) + 1
    if first and path.startswith("w1_"):
        _log(f"capture width {width}: {prefix} ({cls_}) on {path}")


def capture_summary() -> str:
    widths = sorted({w for w, _ in _capture})
    parts = []
    for width in widths:
        incumbent = _capture.get((width, "w1_incumbent"), 0)
        candidate = _capture.get((width, "w1_fp8_candidate"), 0)
        stock = sum(v for (w, path), v in _capture.items() if w == width and not path.startswith("w1_"))
        parts.append(f"{width}:{incumbent + candidate}/{stock}"
                     f"[incumbent={incumbent},candidate_fp8={candidate}]")
    return (f"captured decode attention calls, W1/stock per width: {' '.join(parts) or 'none'} "
            f"(total W1 {sum(v for (w, p), v in _capture.items() if p.startswith('w1_'))}, "
            f"stock {sum(v for (w, p), v in _capture.items() if not p.startswith('w1_'))})")


def _u8(t: torch.Tensor | None) -> torch.Tensor | None:
    """Byte view of a 1-byte cache (vLLM may type the V4.1 records as float8_e4m3fn)."""
    if t is None or t.dtype == torch.uint8 or t.element_size() != 1:
        return t
    return t.view(torch.uint8)


def _layer_class(self) -> str:
    if (getattr(self, "compress_ratio", 0) or 0) != 0:
        return "sparse"
    return "draft" if getattr(self, "layer_id", 0) >= NUM_LAYERS else "swa"


def _comp_format(comp: torch.Tensor | None) -> CacheFormat | None:
    if comp is None:
        return CacheFormat.NONE
    # Rank and dtype must be checked before any indexed shape/stride access.
    if comp.dtype != torch.uint8 or comp.dim() != 3:
        return None
    comp_format = classify_compressed_layout(
        present=True,
        byte_dtype=True,
        ndim=3,
        record_bytes=comp.shape[-1],
        token_stride_bytes=comp.stride(1),
        allow_fp8=COMPRESSED_FP8,
    )
    if comp_format is None or not _cache_layout_ok(comp, comp.shape[-1]):
        return None
    return comp_format


def _cache_layout_ok(cache: torch.Tensor, record_bytes: int) -> bool:
    """Metadata-only page-layout check; safe to call during graph capture."""
    if cache.dtype != torch.uint8 or cache.dim() != 3 or cache.shape[-1] != record_bytes:
        return False
    return cache_layout_is_valid(
        num_pages=cache.shape[0],
        page_tokens=cache.shape[1],
        record_bytes=record_bytes,
        block_stride_bytes=cache.stride(0),
        token_stride_bytes=cache.stride(1),
        byte_stride=cache.stride(-1),
        base_address=cache.data_ptr(),
    )


def _shapes_ok(self, q: torch.Tensor, comp: torch.Tensor | None, comp_format: CacheFormat,
               swa_idx: torch.Tensor, extra_idx, extra_len) -> bool:
    if (q.dim() != 3 or q.shape[1] != 64 or q.shape[2] != 512
            or q.dtype != torch.bfloat16 or q.stride(-1) != 1):
        return False
    swa = _u8(self.swa_cache_layer.kv_cache)
    if not _cache_layout_ok(swa, 528):
        return False
    if (comp is None) != (comp_format == CacheFormat.NONE):
        return False
    if swa_idx is None or swa_idx.dim() != 2:
        return False
    if comp is not None and (extra_idx is None or extra_idx.dim() != 2 or extra_len is None):
        return False
    total = swa_idx.shape[-1] + (0 if comp is None else extra_idx.shape[-1])
    return (total + 63) // 64 <= MAX_BLOCKS


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _cx_ok(self, comp: torch.Tensor, swa_idx: torch.Tensor) -> bool:
    """The cluster kernel's extra conditions: the window fits the first key pair, power-of-two pages."""
    return swa_idx.shape[-1] <= 128 and _pow2(self.swa_cache_layer.kv_cache.shape[1]) and _pow2(comp.shape[1])


def _width_limit(cls_: str) -> int:
    return max(MAX_SQ, MAX_SQ_CX) if cls_ == "sparse" else MAX_SQ_SWA


def _stock(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out) -> None:
    """The stock decode call, verbatim from _forward_decode_mega (top-k already computed)."""
    torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
        q,
        self.swa_cache_layer.kv_cache.unsqueeze(-2),
        swa_metadata.decode_swa_indices.view(q.shape[0], -1),
        self.scale,
        mod._HEAD_DIM_V,
        self.attn_sink,
        swa_metadata.decode_swa_lens,
        extra_cache,
        extra_idx,
        extra_len,
        False,  # enable_q_norm
        0.0,  # rms_norm_eps
        positions_int32,
        *mod._ROPE_ARGS,
        self.rotary_emb.cos_sin_cache,
        self.n_wv_group,
        *mod._SF_ARGS,
        out.data,
        out.scale,
    )


def _eligible(self) -> bool:
    return _layer_class(self) in LAYERS


def make_forward_decode_mega(mod, orig):
    def _forward_decode_mega(self, q, positions_int32, flashmla_metadata, swa_metadata, out) -> None:
        cls_ = _layer_class(self)
        n = q.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        scratch = None
        if cls_ in LAYERS and 1 <= n <= _width_limit(cls_):
            scratch = _scratch.get(q.device)
            if scratch is None and not capturing:  # constructor hook missed this device: prepare eagerly
                scratch = prepare(q.device)
        if scratch is None:
            _stats["stock"] += 1
            if capturing:
                _capture_note(n, "stock", getattr(self, "prefix", "?"), cls_)
            return orig(self, q, positions_int32, flashmla_metadata, swa_metadata, out)
        extra_cache, extra_idx, extra_len = self._decode_compressed_kv_and_topk(flashmla_metadata, swa_metadata)
        assert swa_metadata.decode_swa_indices is not None
        swa_idx = swa_metadata.decode_swa_indices.view(n, -1)
        comp = None if extra_cache is None else _u8(extra_cache.squeeze(-2))
        comp_format = _comp_format(comp)
        if comp_format == CacheFormat.FP8 and not _mask_ok(self, n):
            comp_format = None  # masked out: stock fallback, identical to the disabled-flag path
        mode = "stock" if comp_format is None else select_mode(n, comp_format)
        if (comp_format is None or mode == "stock"
                or not _shapes_ok(self, q, comp, comp_format, swa_idx, extra_idx, extra_len)
                or (mode == "cx" and not _cx_ok(self, comp, swa_idx))):
            _stats["stock_shape"] += 1
            if capturing:
                _capture_note(n, "stock_shape", getattr(self, "prefix", "?"), cls_)
            _stock(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out)
            return
        _stats["w1"] += 1
        extension_path = "w1_fp8_candidate" if comp_format == CacheFormat.FP8 else "w1_incumbent"
        _stats[extension_path] += 1
        if capturing:
            _capture_note(n, extension_path, getattr(self, "prefix", "?"), cls_)
        elif cls_ not in _logged:
            _logged.add(cls_)
            _log(f"{getattr(self, 'prefix', '?')} ({cls_}) on the W1 kernel ({n} tokens, eager)")
        _fp8 = comp_format == CacheFormat.FP8
        if _fp8 and _MASK_EXTRA == "shadow":
            _shadow_call(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out, comp,
                         swa_idx, scratch, mode, capturing)
            return
        if not _fp8 and _SHADOW_INCUMBENT:  # desk H: this call would go to ops._ext(candidate_fp8=False)
            _shadow_call(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out, comp,
                         swa_idx, scratch, mode, capturing, comp_format=comp_format)
            return
        if _fp8 and _MASK_EXTRA == "stock_before":
            _stock(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out)
        ops.attn_decode(q, positions_int32, self.rotary_emb.cos_sin_cache, self.attn_sink, self.scale,
                        _u8(self.swa_cache_layer.kv_cache), swa_idx, swa_metadata.decode_swa_lens, comp, extra_idx,
                        extra_len, out.data, out.scale, scratch,
                        plan=(_MASK_PLAN if (_MASK_PLAN and comp_format == CacheFormat.FP8) else mode),
                        comp_format=comp_format,
                        candidate_fp8=(comp_format == CacheFormat.FP8),
                        pdl=(_MASK_PDL if comp_format == CacheFormat.FP8 else True))
        if _fp8 and _MASK_EXTRA == "stock_after":
            _stock(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out)

    return _forward_decode_mega


def make_init(orig_init):
    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        if _eligible(self):
            prepare(torch.device("cuda", torch.cuda.current_device()))

    return __init__


def install(mod) -> None:
    """Patch ``mod.DeepseekV4MegaAttnAttention`` (``mod`` = the imported flash_mla_mega_attn module). Idempotent.

    * ``__init__``: an installed layer prepares W1 (extension, smem opt-in, scratch) at model build, before any capture.
    * ``_forward_decode_mega``: wraps whatever is installed (stock or another component's), which stays the fallback.
    """
    cls = mod.DeepseekV4MegaAttnAttention
    if getattr(cls, "_w1_installed", False):
        return
    cls.__init__ = make_init(cls.__init__)
    cls._forward_decode_mega = make_forward_decode_mega(mod, cls._forward_decode_mega)
    cls._w1_installed = True
    _log(f"FP8_MASK loaded label={_MASK_LABEL} layers={'all' if _MASK_LAYERS is None else sorted(_MASK_LAYERS)} "
         f"widths={'all' if _MASK_WIDTHS is None else sorted(_MASK_WIDTHS)} fp8_pdl={_MASK_PDL} fp8_plan={_MASK_PLAN} fp8_extra={_MASK_EXTRA}"
         + (" shadow_incumbent=True" if _SHADOW_INCUMBENT else ""))
    if _MASK_EXTRA == "shadow":
        _log(f"FP8_SHADOW installed pid={_os.getpid()} compressed_fp8={COMPRESSED_FP8} "
             f"snapshot_widths={sorted(_SHADOW_WIDTHS)} records={_SHADOW_RECORDS} valid_only={_SHADOW_VALID_ONLY} "
             f"dir={_SHADOW_DIR}; W1->alt, stock->out, device compare; RESET/DUMP sentinels after capture_model")
    if _SHADOW_INCUMBENT:
        _log(f"FP8_SHADOW incumbent installed pid={_os.getpid()} compressed_fp8={COMPRESSED_FP8} "
             f"fp8_shadow={_MASK_EXTRA == 'shadow' and COMPRESSED_FP8} snapshot_widths={sorted(_SHADOW_WIDTHS)} "
             f"records={_SHADOW_RECORDS} valid_only={_SHADOW_VALID_ONLY} rows={SH_ROWS} max_sites={SH_MAX_SITES} "
             f"dir={_SHADOW_DIR}; incumbent W1 (candidate_fp8=False)->alt, stock->out, device compare")
    _log(f"patched {cls.__name__}.__init__ / _forward_decode_mega (W1_LAYERS={sorted(LAYERS)}, W1_MAX_SQ={MAX_SQ}, "
         f"W1_MAX_SQ_CX={MAX_SQ_CX}, W1_MAX_SQ_SWA={MAX_SQ_SWA})")


def install_capture_summary(runner_mod) -> None:
    """Wrap ``GPUModelRunner.capture_model`` of ``runner_mod`` to log the W1 / stock split of the captured graphs."""
    cls = getattr(runner_mod, "GPUModelRunner", None)
    if cls is None or getattr(cls, "_w1_capture_summary", False) or not hasattr(cls, "capture_model"):
        return
    orig = cls.capture_model

    def capture_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        _log(capture_summary())
        on = sorted(k for k, v in _mask_seen.items() if v)
        _log(f"FP8_MASK label={_MASK_LABEL} fp8_on_pairs={len(on)} fp8_off_pairs={len(_mask_seen) - len(on)} "
             f"on_layers={sorted({l for l, _ in on})} on_widths={sorted({w for _, w in on})}")
        if _MASK_EXTRA == "shadow" or _SHADOW_INCUMBENT:
            global _shadow_ready
            _shadow_ready = True
            _log(f"FP8_SHADOW capture done: {_shadow_summary()}")
        return out

    cls.capture_model = capture_model
    cls._w1_capture_summary = True

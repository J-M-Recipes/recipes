"""hotsplit: per-expert HBM residency for UVA-offloaded MXFP4/Marlin MoE layers (experiment, not upstream).

vLLM's UVA offloader works per parameter, so a layer's 384 experts are either all in HBM or all in pinned
host memory. hotsplit runs after process_weights_after_loading and re-partitions every routed-expert layer
into two expert sets:
  hot  -> compact HBM tensors  (index_select of the kernel-format weights)
  cold -> compact pinned-host tensors viewed through UVA
then replaces the layer's quant_method.apply with two Marlin launches (one per set) driven by expert_maps
(global id -> local id, -1 = not in this set) and sums the partial outputs. Marlin + moe_sum already
honour expert_map (the EP path), so no kernel changes.

Budget: by default the hot set gets exactly the HBM bytes the stock layer-granular placement used for
routed experts, so KV-cache size is unchanged. Cells (layer, expert) are ranked by routed count from a
histogram JSON and taken greedily until the budget is spent.

Env:
  HOTSPLIT_COUNTS   path to counts JSON ({"train": {"decode"|"prefill": {layer: [384]}}} or {layer: {"counts": [...]}})
  HOTSPLIT_WEIGHTS  "decode=1,prefill=0.25"  mix of phases (default)
  HOTSPLIT_HOT_GIB  override hot budget (GiB)
  HOTSPLIT_DRYRUN=1 compute + log the plan only
"""
from __future__ import annotations

import json
import os
import re
import time
import types

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.hotsplit")
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _load_counts(path: str) -> dict[int, torch.Tensor]:
    raw = json.load(open(path))
    weights = {"decode": 1.0, "prefill": 0.25}
    for kv in os.environ.get("HOTSPLIT_WEIGHTS", "").split(","):
        if "=" in kv:
            k, v = kv.split("="); weights[k.strip()] = float(v)
    out: dict[int, torch.Tensor] = {}
    if "train" in raw:
        for phase, w in weights.items():
            for li, c in raw["train"].get(phase, {}).items():
                t = torch.tensor(c, dtype=torch.float64)
                t = t / max(t.sum().item(), 1.0) * w  # normalise per layer+phase so phases mix by weight
                out[int(li)] = out.get(int(li), 0) + t
    else:
        for li, v in raw.items():
            t = torch.tensor(v["counts"], dtype=torch.float64); out[int(li)] = t / max(t.sum().item(), 1.0)
    return out


def _is_uva(p: torch.Tensor) -> bool:
    return bool(getattr(p, "_vllm_is_uva_offloaded", False))


def _pinned_view_from_gpu(src: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Exact-size pinned host buffer (cudaHostAlloc via the unpinned-input branch) holding src, as a UVA view."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    host = torch.empty(src.shape, dtype=src.dtype, device="cpu")  # untouched pages; helper allocs pinned + copies
    view = get_accelerator_view_from_cpu_tensor(host)
    del host
    for s in range(0, src.shape[0], chunk):
        view[s:s + chunk].copy_(src[s:s + chunk])
    torch.cuda.synchronize()
    view._vllm_is_uva_offloaded = True
    return view


def _gather_rows(w: torch.Tensor, ids: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """HBM copy of w[ids] (w may be a UVA view; reads stream over C2C)."""
    out = torch.empty((ids.numel(),) + tuple(w.shape[1:]), dtype=w.dtype, device="cuda")
    for s in range(0, ids.numel(), chunk):
        out[s:s + chunk].copy_(w.index_select(0, ids[s:s + chunk]))
    return out


def _gather_rows_to_host(w: torch.Tensor, ids: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    host = torch.empty((ids.numel(),) + tuple(w.shape[1:]), dtype=w.dtype, device="cpu")
    view = get_accelerator_view_from_cpu_tensor(host)
    del host
    for s in range(0, ids.numel(), chunk):
        view[s:s + chunk].copy_(w.index_select(0, ids[s:s + chunk]))
    torch.cuda.synchronize()
    view._vllm_is_uva_offloaded = True
    return view


def _host_avail_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return -1.0


def apply_hotsplit(model: torch.nn.Module) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        make_mxfp4_moe_kernel,
        make_mxfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    t0 = time.time()
    counts = _load_counts(os.environ["HOTSPLIT_COUNTS"])
    layers: dict[int, torch.nn.Module] = {}
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, Mxfp4MoEMethod) and hasattr(mod, "w13_weight") and qm.moe_kernel is not None:
            m = _LAYER_RE.search(name)
            if m and int(m.group(1)) in counts:
                layers[int(m.group(1))] = mod
    if not layers:
        logger.warning("hotsplit: no eligible MXFP4 MoE layers found; nothing to do")
        return
    any_l = next(iter(layers.values()))
    E = any_l.w13_weight.shape[0]
    cell_bytes = {li: (l.w13_weight[0].numel() * l.w13_weight.element_size()
                       + l.w2_weight[0].numel() * l.w2_weight.element_size()) for li, l in layers.items()}
    hbm_now = sum(cell_bytes[li] * E for li, l in layers.items() if not _is_uva(l.w13_weight))
    budget = float(os.environ["HOTSPLIT_HOT_GIB"]) * 2**30 if os.environ.get("HOTSPLIT_HOT_GIB") else hbm_now
    # rank all (layer, expert) cells by normalised routing share per byte
    cells = []
    for li in layers:
        c = counts[li]
        for e in range(E):
            cells.append((c[e].item() / cell_bytes[li], li, e))
    cells.sort(reverse=True)
    hot: dict[int, list[int]] = {li: [] for li in layers}
    used = 0
    for _, li, e in cells:
        if used + cell_bytes[li] > budget:
            continue
        hot[li].append(e); used += cell_bytes[li]
    cov = sum(counts[li][hot[li]].sum().item() for li in layers) / sum(counts[li].sum().item() for li in layers)
    logger.info("hotsplit plan: %d layers, budget %.1f GiB (stock HBM experts %.1f GiB), hot cells %d/%d, "
                "train-weighted coverage %.1f%%", len(layers), budget / 2**30, hbm_now / 2**30,
                sum(len(v) for v in hot.values()), E * len(layers), 100 * cov)
    logger.info("hotsplit per-layer hot counts: %s", {li: len(hot[li]) for li in sorted(hot)})
    if os.environ.get("HOTSPLIT_DRYRUN") == "1":
        return

    # order: UVA layers (consume HBM, free host) while HBM allows, else an HBM layer (frees HBM, consumes host)
    uva_q = [li for li in sorted(layers) if _is_uva(layers[li].w13_weight)]
    hbm_q = [li for li in sorted(layers) if not _is_uva(layers[li].w13_weight)]
    margin = 6 * 2**30
    done = 0
    while uva_q or hbm_q:
        free, _ = torch.cuda.mem_get_info()
        if uva_q and (free - len(hot[uva_q[0]]) * cell_bytes[uva_q[0]] > margin or not hbm_q):
            li = uva_q.pop(0)
        else:
            li = hbm_q.pop(0)
        host_need = (E - len(hot[li])) * cell_bytes[li] / 2**30
        if _host_avail_gib() - host_need < float(os.environ.get("HOTSPLIT_HOST_FLOOR_GIB", "24")):
            logger.warning("hotsplit: host floor hit (avail %.1f GiB, need %.1f); leaving layer %d and %d others stock",
                           _host_avail_gib(), host_need, li, len(uva_q) + len(hbm_q))
            break
        _split_layer(layers[li], hot[li], E, make_mxfp4_moe_kernel, make_mxfp4_moe_quant_config, li)
        done += 1
        if done % 10 == 0:
            torch.cuda.empty_cache()
            logger.info("hotsplit: %d/%d layers; HBM free %.1f GiB; host avail %.1f GiB",
                        done, len(layers), torch.cuda.mem_get_info()[0] / 2**30, _host_avail_gib())
    torch.cuda.empty_cache()
    if _LIVE is not None and _LIVE.layer_idx:
        mnbt = int(os.environ.get("HOTSPLIT_MAX_TOKENS", "8192"))
        _LIVE.finalize(E, mnbt, 8)
    logger.info("hotsplit done in %.0fs; HBM free %.1f GiB; host avail %.1f GiB",
                time.time() - t0, torch.cuda.mem_get_info()[0] / 2**30, _host_avail_gib())


def _split_layer(layer, hot_ids: list[int], E: int, make_kernel, make_qcfg, li: int = -1) -> None:
    qm = layer.quant_method
    dev = torch.device("cuda")
    hot_set = set(hot_ids)
    h = torch.tensor(sorted(hot_set), dtype=torch.long, device=dev)
    c = torch.tensor([e for e in range(E) if e not in hot_set], dtype=torch.long, device=dev)

    def emap(ids: torch.Tensor) -> torch.Tensor:
        m = torch.full((E,), -1, dtype=torch.int32, device=dev)
        if ids.numel():
            m[ids] = torch.arange(ids.numel(), dtype=torch.int32, device=dev)
        return m

    w13, w2 = layer.w13_weight, layer.w2_weight
    s13, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    parts = []
    for ids, where in ((h, "hbm"), (c, "host")):
        if ids.numel() == 0:
            continue
        if where == "hbm":
            pw13, pw2 = _gather_rows(w13, ids), _gather_rows(w2, ids)
        else:
            pw13, pw2 = _gather_rows_to_host(w13, ids), _gather_rows_to_host(w2, ids)
        ps13, ps2 = s13.index_select(0, ids).contiguous(), s2.index_select(0, ids).contiguous()
        qcfg = make_qcfg(mxfp4_backend=qm.mxfp4_backend, w1_scale=ps13, w2_scale=ps2,
                         swiglu_limit=getattr(layer, "swiglu_limit", None), layer=layer)
        kern = make_kernel(moe_quant_config=qcfg, moe_config=qm.moe, experts_cls=qm.experts_cls,
                           mxfp4_backend=qm.mxfp4_backend, routing_tables=layer._expert_routing_tables())
        parts.append((kern, pw13, pw2, emap(ids)))

    # drop the original full tensors (frees HBM or the pinned host buffer)
    placeholder = torch.nn.Parameter(torch.empty(0, dtype=w13.dtype, device=dev), requires_grad=False)
    layer.w13_weight = placeholder
    layer.w2_weight = torch.nn.Parameter(torch.empty(0, dtype=w2.dtype, device=dev), requires_grad=False)
    del w13, w2
    # free the full-size scales in place (old kernel's quant config still points at these Parameters; unused now)
    for sname in ("w13_weight_scale", "w2_weight_scale"):
        sp = getattr(layer, sname, None)
        if isinstance(sp, torch.Tensor):
            sp.data = torch.empty(0, dtype=sp.dtype, device=sp.device)
    torch.cuda.synchronize()
    qm._hotsplit_parts = parts
    qm._hotsplit_nhot = h.numel()

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        assert shared_experts is None, "hotsplit: shared experts path not supported"
        slot = getattr(self, "_hotsplit_live_slot", None)
        if slot is not None and _LIVE is not None and _LIVE.buf is not None \
                and topk_ids.shape[0] <= _LIVE.decode_max:  # decode-sized steps only (static shape under capture)
            _LIVE.count(slot, topk_ids)
        out = None
        for kern, pw13, pw2, m in self._hotsplit_parts:
            o = kern.apply(hidden_states=x, w1=pw13, w2=pw2, topk_weights=topk_weights, topk_ids=topk_ids,
                           activation=layer.activation, global_num_experts=layer.global_num_experts,
                           apply_router_weight_on_input=layer.apply_router_weight_on_input, expert_map=m,
                           shared_experts=None, shared_experts_input=None)
            out = o if out is None else out.add_(o)
        return out

    qm.apply = types.MethodType(apply, qm)
    if _LIVE is not None:
        _LIVE.register(qm, li)


class _LiveCounter:
    """Graph-safe per-layer routed-expert counter for the serving lane.
    index_add_ into a preallocated int64 device tensor (no host sync, no allocation) so it is captured
    into the decode CUDA graph and replays with it. A daemon thread snapshots to HOTSPLIT_LIVE_COUNTS
    every HOTSPLIT_LIVE_SECS (default 600) as {"live": {"decode": {layer: [E]}}, "meta": {...}}.
    Caveat: padded slots in a captured batch are counted too (C1 has no padding)."""

    def __init__(self, path: str, secs: int):
        self.path, self.secs = path, secs
        self.decode_max = int(os.environ.get("HOTSPLIT_LIVE_DECODE_MAX", "16"))
        self.buf = None
        self.layer_idx: list[int] = []
        self.ones = None

    def register(self, qm, li: int):
        slot = len(self.layer_idx)
        self.layer_idx.append(li)
        qm._hotsplit_live_slot = slot

    def finalize(self, E: int, max_tokens: int, topk: int):
        dev = torch.device("cuda")
        self.buf = torch.zeros((len(self.layer_idx), E), dtype=torch.int64, device=dev)
        self.ones = torch.ones(max_tokens * topk, dtype=torch.int64, device=dev)
        import threading
        threading.Thread(target=self._loop, daemon=True, name="hotsplit-live").start()
        logger.info("hotsplit live counter: %d layers -> %s every %ds", len(self.layer_idx), self.path, self.secs)

    def count(self, slot: int, topk_ids: torch.Tensor):
        ids = topk_ids.reshape(-1).to(torch.int64)
        self.buf[slot].index_add_(0, ids, self.ones[: ids.numel()])

    def _loop(self):
        t0 = time.time()
        while True:
            time.sleep(self.secs)
            try:
                snap = self.buf.to("cpu", non_blocking=False)
                out = {"live": {"decode": {str(li): snap[s].tolist() for s, li in enumerate(self.layer_idx)}},
                       "meta": {"since": t0, "at": time.time(), "note": "live lane counts incl. padded slots"}}
                tmp = self.path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(out, f)
                os.replace(tmp, self.path)
            except Exception as e:  # never take the lane down for telemetry
                logger.warning("hotsplit live snapshot failed: %s", e)


_LIVE = _LiveCounter(os.environ["HOTSPLIT_LIVE_COUNTS"], int(os.environ.get("HOTSPLIT_LIVE_SECS", "600"))) \
    if os.environ.get("HOTSPLIT_LIVE_COUNTS") else None

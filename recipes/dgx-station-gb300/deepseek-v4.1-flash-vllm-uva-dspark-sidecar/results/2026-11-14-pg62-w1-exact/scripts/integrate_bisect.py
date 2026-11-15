# TEST INSTRUMENT - NEVER THE SERVED FILE. Published copy (2026-11-15) of milo-takeover/bisect/integrate.py: the W13
# integrate.py plus a layer x width mask (fp8_mask.json) and plan/PDL switches used by the oracle bisect arms. The
# serve-bisect test tree ran this file for the oracle arms and for the perf cells (B-divfix, B-stockall, F2, P-B,
# P-F2) with the mask set to all layers and all widths; the cx arms G32/G76 ran results/G drafts (not published,
# not in the recipe). The integrate.py that serves Pg62 is the W13 file (sha256 bc87880f...).
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
    return layers, widths, str(d.get("label", "?"))


_MASK_EXTRA = None  # "stock_after": W1 then stock into the same out; "stock_before": stock then W1 (FP8 calls only)
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
         f"widths={'all' if _MASK_WIDTHS is None else sorted(_MASK_WIDTHS)} fp8_pdl={_MASK_PDL} fp8_plan={_MASK_PLAN} fp8_extra={_MASK_EXTRA}")
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
        return out

    cls.capture_model = capture_model
    cls._w1_capture_summary = True

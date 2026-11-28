"""PF1 (W15 ProjFuse) -- direct eager call plan for ``vllm::mm_mxfp8`` (backend "cute-dsl", M > 8).

Why (receipts/AP-manual, Card AP): one layer-1 attention forward at T=8162 spends ~0.65 ms of profiled self time
(corrected for the backend_impl double count, see results/W15/attr_pstats.py; ~0.22-0.44 ms once de-profiled,
results/W15/arith_pf.out.txt) in the host path of its three projection GEMMs that is NOT the GEMM launch: the torch
CustomOpDef layer (A), FlashInfer's ``backend_requirement`` / api-logging / inspect.bind wrapper (B), a per-call
rebuild of the CuTe-DSL runner class + imports (C) and the AutoTuner lookup (D).
Every one of those is a pure function of (M-bucket, K, N, out_dtype, autotuner state) and returns the same
``(runner, tactic)`` for every call -- except the first -- so the plan is resolved once per key and the
GEMM is launched by calling the SAME runner ``forward`` with the SAME tensors and the SAME tactic.

What it changes: exactly one module attribute, ``vllm.utils.flashinfer.mm_mxfp8`` (the CustomOpDef) is replaced
by a python shim that falls back to the CustomOpDef for everything that is not [eager, cuda-graph-free,
backend == "cute-dsl", 1-D scales, autotuner idle].  ``ll_gemm.apply_weights`` (closure cell
``mxfp8_e4m3_quantize`` that ``producer_binding.rebind_upstream_quant`` rebinds) is NOT touched; the
quantize calls are NOT touched; captured graphs are NOT touched (``is_current_stream_capturing`` and
``torch.compiler.is_compiling`` go first).

Coupled to FlashInfer 0.7.0.post1 (the box's, gemm_base.py line numbers match the Card-AP pstats exactly).
Any other version, any missing name, any shadow mismatch => install refused / plan disabled, the original
CustomOpDef path runs (fail open).  Desk evidence: results/W15/test_pf1.py.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import torch

SUPPORTED_FLASHINFER = ("0.7.0.post1",)
REVALIDATE_EVERY = 1024          # direct calls per plan between tuner re-resolutions
STATS_EVERY = 20000              # one log line per N direct calls
CUSTOM_OP = "mxfp8_gemm"         # gemm_base.py:6273 (verbatim)
WS_NAME = "mm_mxfp8_workspace"   # gemm_base.py:6225 (verbatim)


def _LOG(msg):
    sys.stderr.write(f"MEGA_PEER PF1 {msg}\n")


class _Plan:
    __slots__ = ("runner", "tactic", "calls")

    def __init__(self, runner, tactic):
        self.runner, self.tactic, self.calls = runner, tactic, 0


class _State:
    def __init__(self):
        self.orig = None          # the CustomOpDef
        self.gb = None            # flashinfer.gemm.gemm_base
        self.tuner = None
        self.get_cache_buf = None
        self.ws_size = None
        self.plans = {}
        self.fp = None
        self.bucket = None
        self.disabled = None      # reason string once disabled
        self.confirmed = 0
        self.active_logged = False
        self.stats = Counter()


_ST = _State()


# ------------------------------------------------------------------------------------------------------------
def _fingerprint(t):
    return (id(t._file_configs), len(t._file_configs), t._dirty_seq, t._active_tuning_contexts)


def _tuner_idle(t):
    """True when the AutoTuner would answer from its steady-state cache for a plain call."""
    return (not t.is_tuning_mode) and (not t._get_skip_ops_stack()) and (not t._get_override_stack())


def _resolve(A, B, A_scale, B_scale, out_dtype, out):
    """Resolve (runner, tactic) exactly as gemm_base.mm_mxfp8 lines 6228-6277 do for backend == "cute-dsl"."""
    st, gb = _ST, _ST.gb
    major, minor = gb.get_compute_capability(A.device)
    runner = gb._cute_dsl_gemm_mxfp8_runner(major, minor, True, out_dtype)
    ws = st.get_cache_buf(WS_NAME, st.ws_size, A.device)
    inputs = [A, B, A_scale, B_scale, out_dtype, out, ws]
    sel, tactic = st.tuner.choose_one(
        custom_op=CUSTOM_OP,
        runners=[runner],
        tuning_config=gb._MM_MXFP8_CUTE_DSL_TUNING_CONFIG,
        inputs=inputs,
    )
    return sel, tactic


def _direct(plan, A, B, A_scale, B_scale, out_dtype):
    st, gb = _ST, _ST.gb
    # same validation call the backend_requirement wrapper makes (raises the same ValueErrors)
    gb._check_mm_mxfp8_problem_size(A, B, A_scale, B_scale, None, out_dtype, False, "cute-dsl")
    out = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=out_dtype)       # gemm_base.py:6218
    ws = st.get_cache_buf(WS_NAME, st.ws_size, A.device)                                  # gemm_base.py:6224
    plan.runner(inputs=[A, B, A_scale, B_scale, out_dtype, out, ws], tactic=plan.tactic)  # gemm_base.py:6279
    return out


def _same_bits(x, y):
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.element_size() == 2:
        return bool(torch.equal(x.view(torch.int16), y.view(torch.int16)))
    return bool(torch.equal(x.view(torch.uint8), y.view(torch.uint8)))


def _miss(key, A, B, A_scale, B_scale, out_dtype, backend, use_8x4):
    """First sight of a key: original path (its result is returned), then build + shadow-check the plan."""
    st = _ST
    ref = st.orig(A, B, A_scale, B_scale, out_dtype=out_dtype, backend=backend, use_8x4_sf_layout=use_8x4)
    try:
        probe = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=out_dtype)
        sel, tactic = _resolve(A, B, A_scale, B_scale, out_dtype, probe)
        plan = _Plan(sel, tactic)
        got = _direct(plan, A, B, A_scale, B_scale, out_dtype)
        if not _same_bits(ref, got):
            st.disabled = f"shadow mismatch key={key} tactic={tactic!r}"
            _LOG(f"DISABLED {st.disabled}; original CustomOpDef path from here on")
            return ref
        st.plans[key] = plan
        st.confirmed += 1
        st.stats["plans"] += 1
        if not st.active_logged:
            st.active_logged = True
            _LOG("ACTIVE vllm.utils.flashinfer.mm_mxfp8 -> direct cute-dsl call plan "
                 f"(eager, M>8; flashinfer {SUPPORTED_FLASHINFER[0]}; shadow-confirmed per key)")
    except Exception as exc:  # fail open
        st.disabled = f"plan build error {exc!r}"
        _LOG(f"DISABLED {st.disabled}; original CustomOpDef path from here on")
    return ref


def mm_mxfp8(A, B, A_scale, B_scale, out_dtype, backend="cutlass", use_8x4_sf_layout=False):
    st = _ST
    orig = st.orig
    if torch.compiler.is_compiling():            # dynamo/inductor trace: keep the op in the graph exactly as before
        return orig(A, B, A_scale, B_scale, out_dtype=out_dtype, backend=backend, use_8x4_sf_layout=use_8x4_sf_layout)
    if (backend != "cute-dsl" or use_8x4_sf_layout or st.disabled is not None
            or torch.cuda.is_current_stream_capturing()):
        st.stats["orig_other"] += 1
        return orig(A, B, A_scale, B_scale, out_dtype=out_dtype, backend=backend, use_8x4_sf_layout=use_8x4_sf_layout)
    t = st.tuner
    if (A.ndim != 2 or B.ndim != 2 or A_scale.ndim != 1 or B_scale.ndim != 1 or A.shape[0] <= 8
            or not _tuner_idle(t)):
        st.stats["orig_shape_or_tuner"] += 1
        return orig(A, B, A_scale, B_scale, out_dtype=out_dtype, backend=backend, use_8x4_sf_layout=use_8x4_sf_layout)
    fp = _fingerprint(t)
    if fp != st.fp:
        st.plans.clear()
        st.fp = fp
        st.bucket = t.get_effective_map_to_tuning_buckets(st.gb._MM_MXFP8_CUTE_DSL_TUNING_CONFIG, spec_idx=0)
    m = A.shape[0]
    key = (st.bucket(m), A.shape[1], B.shape[1], out_dtype, A.device)
    plan = st.plans.get(key)
    if plan is None:
        st.stats["miss"] += 1
        return _miss(key, A, B, A_scale, B_scale, out_dtype, backend, use_8x4_sf_layout)
    plan.calls += 1
    if plan.calls % REVALIDATE_EVERY == 0:       # bounded staleness: re-run the tuner's lookup, compare the answer
        probe = torch.empty((m, B.shape[1]), device=A.device, dtype=out_dtype)
        _, tactic = _resolve(A, B, A_scale, B_scale, out_dtype, probe)
        if tactic != plan.tactic:
            st.stats["tactic_changed"] += 1
            st.plans.clear()
            return orig(A, B, A_scale, B_scale, out_dtype=out_dtype, backend=backend, use_8x4_sf_layout=use_8x4_sf_layout)
    st.stats["direct"] += 1
    if st.stats["direct"] % STATS_EVERY == 0:
        _LOG(f"STATS {dict(st.stats)} plans={len(st.plans)} confirmed={st.confirmed}")
    return _direct(plan, A, B, A_scale, B_scale, out_dtype)


# ------------------------------------------------------------------------------------------------------------
def install(kernel_module):
    """kernel_module = vllm.model_executor.kernels.linear.mxfp8.flashinfer (already imported, ll_gemm applied)."""
    if os.environ.get("MEGA_PF1", "1") != "1":
        _LOG("off (MEGA_PF1=0)")
        return False
    vf = kernel_module.vllm_flashinfer                      # vllm.utils.flashinfer
    cur = getattr(vf, "mm_mxfp8", None)
    if cur is None or getattr(cur, "_pf1", False):
        return False
    try:
        import flashinfer
        from flashinfer.autotuner.autotuner import AutoTuner
        from flashinfer.gemm import gemm_base as gb
        from flashinfer.utils import _get_cache_buf
        ver = getattr(flashinfer, "__version__", "?")
        if ver not in SUPPORTED_FLASHINFER:
            _LOG(f"NOT installed: flashinfer {ver} not in {SUPPORTED_FLASHINFER}")
            return False
        for name in ("_cute_dsl_gemm_mxfp8_runner", "_check_mm_mxfp8_problem_size", "_MM_MXFP8_CUTE_DSL_TUNING_CONFIG",
                     "DEFAULT_WORKSPACE_SIZE", "get_compute_capability"):
            if not hasattr(gb, name):
                _LOG(f"NOT installed: gemm_base.{name} missing")
                return False
        tuner = AutoTuner.get()
        for name in ("_file_configs", "_dirty_seq", "_active_tuning_contexts", "is_tuning_mode",
                     "_get_skip_ops_stack", "_get_override_stack", "get_effective_map_to_tuning_buckets", "choose_one"):
            if not hasattr(tuner, name):
                _LOG(f"NOT installed: AutoTuner.{name} missing")
                return False
    except Exception as exc:
        _LOG(f"NOT installed: {exc!r}")
        return False
    _ST.orig, _ST.gb, _ST.tuner, _ST.get_cache_buf = cur, gb, tuner, _get_cache_buf
    _ST.ws_size = gb.DEFAULT_WORKSPACE_SIZE
    mm_mxfp8._pf1 = True
    mm_mxfp8.__wrapped__ = cur
    vf.mm_mxfp8 = mm_mxfp8
    _LOG("installed (waiting for first shadow-confirmed eager call)")
    return True

# vllm-sitecustomize-hooks.py — bind-mount over /usr/lib/python3.12/sitecustomize.py in a vLLM container:
#   -v $C/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro
# STATUS: both hooks confirmed installing on the v0.28.0 image (stderr lines appear in `docker logs`
# within ~40 s of launch). Still confirm per launch: a "Using FlashInfer autotune cache file" hit on the
# pinned dir (seed it first — see SKILL.md), and "ROUTE_TRACE attn_gid unwrap patch installed" when tracing.
#
# MLA / DSA models (GLM-5.x, DeepSeek): the stock capturer's get_routed_experts_attn_gid does a bare
# isinstance(spec, FullAttentionSpec); at TP1 vLLM wraps the model's MLAAttentionSpec inside a
# UniformTypeKVCacheSpecs aggregate, so the check fails and the engine dies AFTER the full weight load
# with "Routed-experts capture requires a full-attention KV cache group." The trace hook below replaces
# that function with one that unwraps the aggregate. Without it the flag costs a whole load and returns nothing.
#
# Two opt-in hooks, both installed lazily via a meta-path finder so nothing is imported early:
#   VLLM_AUTOTUNE_CACHE_KEY=<name>  pins the FlashInfer autotune cache dir to <name> instead of the
#       config hash, so shape-identical configs (offload-GB, seqs, instrumentation flags) share one
#       cache. Use a distinct key per --moe-backend.
#   ROUTE_TRACE_DIR=/trace          appends every scheduler step's routed-expert ids to disk from
#       RoutedExpertsManager.store_batch (CPU side, once per step, outside the compiled graph).
#       Needs --enable-return-routed-experts. Files: trace-<pid>.i16 (int16 [tokens,layers,topk]),
#       trace-<pid>.slots.i64, trace-<pid>.steps ("epoch tokens layers topk" per step).
# The image's stock sitecustomize only installs the apport hook — keep that block.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import os, sys
import importlib.abc, importlib.util


def _lazy_patch(module_name, patch_fn):
    """Run patch_fn(module) right after module_name is first imported."""

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != module_name:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                patch_fn(module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


_AT_KEY = os.environ.get("VLLM_AUTOTUNE_CACHE_KEY")
if _AT_KEY:
    def _patch_autotune(module):
        module.flashinfer_autotune_cache_hash = lambda runner: _AT_KEY
        sys.stderr.write(f"AUTOTUNE_KEY hook installed: {_AT_KEY}\n")

    _lazy_patch("vllm.model_executor.warmup.flashinfer_autotune_cache", _patch_autotune)


_OUT = os.environ.get("ROUTE_TRACE_DIR")
if _OUT:
    def _patch_trace(module):
        import numpy as np, time
        # Accept UniformTypeKVCacheSpecs whose inner specs are all full-attention (MLA models at TP1).
        from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

        def _gid(kv_cache_config):
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if isinstance(spec, FullAttentionSpec):
                    return gid
                if isinstance(spec, UniformTypeKVCacheSpecs) and all(
                    isinstance(s, FullAttentionSpec) for s in spec.kv_cache_specs.values()
                ):
                    return gid
            raise ValueError("Routed-experts capture requires a full-attention KV cache group.")

        module.get_routed_experts_attn_gid = _gid
        sys.stderr.write("ROUTE_TRACE attn_gid unwrap patch installed\n")
        cls = module.RoutedExpertsManager
        orig = cls.store_batch
        os.makedirs(_OUT, exist_ok=True)
        pid = os.getpid()
        fb = open(os.path.join(_OUT, f"trace-{pid}.i16"), "ab")
        fs = open(os.path.join(_OUT, f"trace-{pid}.slots.i64"), "ab")
        fm = open(os.path.join(_OUT, f"trace-{pid}.steps"), "a")
        seen = {"n": 0}

        def store_batch(self, data, slot_mapping):
            try:
                d = np.ascontiguousarray(data, dtype=np.int16)
                s = np.ascontiguousarray(slot_mapping, dtype=np.int64)
                fb.write(d.tobytes()); fs.write(s.tobytes())
                fm.write(f"{time.time():.3f} {d.shape[0]} {d.shape[1]} {d.shape[2]}\n")
                seen["n"] += d.shape[0]
                if seen["n"] % 2000 < d.shape[0]:
                    fb.flush(); fs.flush(); fm.flush()
            except Exception as e:  # never break serving
                sys.stderr.write(f"ROUTE_TRACE error: {e!r}\n")
            return orig(self, data, slot_mapping)

        cls.store_batch = store_batch
        sys.stderr.write(f"ROUTE_TRACE hook installed pid={pid} dir={_OUT}\n")

    _lazy_patch("vllm.model_executor.layers.fused_moe.routed_experts_capturer", _patch_trace)

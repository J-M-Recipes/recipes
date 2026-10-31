# Mounted at /w with PYTHONPATH=/w. Runs in every python process, including the spawned EngineCore.
#  K3_MAX_LAYER=N  : drop checkpoint tensors for decoder layers >= N (truncated-model boots).
#  K3POD_PLAN=path : four-tier pod hook (k3pod.py): rename/drop routed experts, patch KimiMoE.
import os, re

_N = os.environ.get("K3_MAX_LAYER")
_N = int(_N) if _N is not None else None
_POD = bool(os.environ.get("K3POD_PLAN"))
_MODULAR = _POD or os.environ.get("K3_FORCE_MODULAR") == "1"
_SPLIT_MOE = os.environ.get("K3_SPLIT_MOE") == "1"

if _N is not None or _POD or _MODULAR or _SPLIT_MOE:
    import importlib.abc, importlib.machinery, sys
    _LAYER = re.compile(r"\.layers\.(\d+)\.")
    _rewrite = None
    if _POD:
        import k3pod
        _rewrite = k3pod.rewrite_name

    def _map(name):
        if _N is not None:
            m = _LAYER.search(name)
            if m and int(m.group(1)) >= _N:
                return None
        if _rewrite is not None:
            return _rewrite(name)
        return name

    def _patch_weights(mod):
        for attr in ("safetensors_weights_iterator", "fastsafetensors_weights_iterator",
                     "multi_thread_safetensors_weights_iterator"):
            f = getattr(mod, attr, None)
            if f is None or getattr(f, "_k3_wrapped", False):
                continue

            def wrap(fn):
                def it(*a, **k):
                    dropped = renamed = 0
                    for item in fn(*a, **k):
                        new = _map(item[0])
                        if new is None:
                            dropped += 1
                            continue
                        if new != item[0]:
                            renamed += 1
                            item = (new,) + tuple(item[1:])
                        yield item
                    print(f"[k3 hook] {fn.__name__}: dropped {dropped}, renamed {renamed}", flush=True)
                it._k3_wrapped = True
                it.__name__ = fn.__name__
                return it
            setattr(mod, attr, wrap(f))

    _HOOKS = {"vllm.model_executor.model_loader.weight_utils": _patch_weights}
    def _force_modular(mod):
        # The monolithic TRT-LLM NVFP4 kernel routes inside the kernel over all experts; the pod needs
        # vLLM's own router (select_experts) followed by the modular kernel.
        mod.TrtLlmNvFp4ExpertsMonolithic._supports_routing_method = staticmethod(lambda *a, **k: False)
        print("[k3 hook] TRT-LLM NVFP4 MoE: monolithic disabled, modular path forced", flush=True)

    def _split_moe(mod):
        # Make the MoE custom op a piecewise-CUDA-graph split point: it runs eagerly between captured
        # pieces, so the pod's network wait never sits inside a graph.
        ops = mod.CompilationConfig._attention_ops
        for op in ("vllm::moe_forward", "vllm::moe_forward_shared"):
            if op not in ops:
                ops.append(op)
        print("[k3 hook] MoE ops added to piecewise splitting ops", flush=True)

    if _SPLIT_MOE:
        _HOOKS["vllm.config.compilation"] = _split_moe
    if _MODULAR:
        _HOOKS["vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe"] = _force_modular
    if _POD:
        _HOOKS["vllm.models.kimi_k3.nvidia.model"] = lambda m: k3pod.install_model_patch(m)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            hook = _HOOKS.get(fullname)
            if hook is None:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig, _hook=hook):
                _orig(module)
                _hook(module)
            spec.loader.exec_module = exec_module
            return spec

    import importlib.util
    sys.meta_path.insert(0, _Finder())

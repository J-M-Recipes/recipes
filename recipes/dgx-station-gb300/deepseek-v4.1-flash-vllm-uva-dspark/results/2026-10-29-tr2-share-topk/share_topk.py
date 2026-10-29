"""L2 top-k table sharing for DeepSeek-V4.1-Flash on the 0909 image (Card TR2, J&M Recipes).

Idea (from the sidecar's h1, `attn_share.py`): in one decode forward, every compressed
layer that shares a (kv source, index source, compress_ratio) group calls
`compute_global_topk_indices_and_lens` on identical inputs. The op is a pure
gather/count, so 38 calls per step can be 3 without changing a byte of output.

This file patches the 0909 (`deepseekv41-flash-0909`, engine 179dd0fa9) code shape, which
differs from af7f9488: the call sits inline in
`DeepseekV4FlashMLAAttention._forward_decode` (vllm/models/deepseek_v4_1/nvidia/flashmla.py:201)
and is imported into that module's namespace, so the memo wraps the *module attribute*
and needs no re-implementation of the method.

Env:
  L2_SHARE=0        report-only: count would-be hits/misses, call the original every time
  L2_SHARE=1        strict: the stamp carries every input pointer
  L2_SHARE=2        relaxed: token_to_req_indices / is_valid_token pointers dropped from the
                    stamp (they are per-SWA-group buffers with equal content by construction)
  L2_VERIFY=1       on a hit, also run the original and torch.equal() the result; on mismatch
                    log and serve the original. Host sync, skipped while capturing.
  L2_STATS_EVERY=N  print per-forward deltas (first 8 forwards, then every N)

Reset: the cache is cleared at the start of every forward, detected on the backbone's
layer 0 in `DeepseekV4Attention.forward`. Under CUDA-graph capture that fires at capture
and the dedupe is baked into the graph; under piecewise/eager decode it fires per step.

Nothing here changes the launch when L2_SHARE is unset.
"""

import os
import sys

SHARE = os.environ.get("L2_SHARE", "0").strip()
VERIFY = os.environ.get("L2_VERIFY", "0") == "1"
VERIFY_FILE = os.environ.get("L2_VERIFY_FILE", "")
AUDIT_FILE = os.environ.get("L2_AUDIT_FILE", "")
_audit_calls = [0]
AUDIT_LOG_MAX = int(os.environ.get("L2_AUDIT_LOG_MAX", "40") or 40)
STATS_EVERY = int(os.environ.get("L2_STATS_EVERY", "64") or 0)
_VERIFY_CHECK_EVERY = 200
_verify_calls = [0]


def _audit_enabled():
    if not AUDIT_FILE:
        return False
    try:
        with open(AUDIT_FILE) as fh:
            return fh.read().strip() == "1"
    except OSError:
        return False


def _verify_enabled():
    """VERIFY can be toggled at runtime from a file, so one boot can run a verify window and then time cells."""
    if not VERIFY_FILE:
        return VERIFY
    try:
        with open(VERIFY_FILE) as fh:
            return fh.read().strip() == "1"
    except OSError:
        return VERIFY

ATTN_MODULE = "vllm.models.deepseek_v4_1.attention"
FLASHMLA_MODULE = "vllm.models.deepseek_v4_1.nvidia.flashmla"
# i.e. vllm/models/deepseek_v4_1/attention.py and .../nvidia/flashmla.py — V4.1, not V4.

_LOG = lambda m: sys.stderr.write("L2 " + m + "\n")


class _State:
    def __init__(self):
        self.gen = 0
        self.cur = None            # the attention layer currently inside _forward_decode
        self.cache = {}            # key -> (stamp, indices, lens)
        self.verify_on = False
        self.stats = {"forwards": 0, "src": 0, "hit": 0, "hit_noskip": 0, "miss": 0,
                      "skip": 0, "peer": 0, "verify_ok": 0, "verify_bad": 0,
                      "audit_ok": 0, "audit_bad": 0}

    def reset(self):
        n = self.stats["forwards"]
        last = getattr(self, "_last", {})
        if STATS_EVERY and n and (n <= 8 or n % STATS_EVERY == 0):
            delta = {k: v - last.get(k, 0) for k, v in self.stats.items() if k != "forwards"}
            _LOG("stats forward=%d this_forward=%s total=%s" % (n, delta, self.stats))
        self._last = dict(self.stats)
        self.gen += 1
        self.cache.clear()
        self.cur = None
        self.stats["forwards"] += 1


S = _State()


def _capturing():
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


# ---------------------------------------------------------------- memo over the module op
def _memo(orig):
    strict = SHARE == "1"
    report_only = SHARE == "0"

    def compute_global_topk_indices_and_lens(local_indices, token_to_req_indices,
                                             block_table, block_size, is_valid_token):
        layer = S.cur
        if layer is None:
            return orig(local_indices, token_to_req_indices, block_table, block_size,
                        is_valid_token)
        key = (layer.kv_source_layer_id, layer.index_source_layer_id, layer.compress_ratio)
        stamp = (S.gen, local_indices.data_ptr(), tuple(local_indices.shape),
                 block_table.data_ptr(), int(block_size))
        if strict:
            stamp += (is_valid_token.data_ptr(), is_valid_token.numel(),
                      token_to_req_indices.data_ptr())

        is_source = layer.layer_id == layer.index_source_layer_id
        hit = S.cache.get(key)
        matched = hit is not None and hit[0] == stamp
        if hit is not None and _audit_enabled():
            # AUDIT: the logical key matched — is the recomputed table identical anyway?
            aref = orig(local_indices, token_to_req_indices, block_table, block_size,
                        is_valid_token)
            same = (_same(aref[0], hit[1]) and _same(aref[1], hit[2]))
            S.stats["audit_ok" if same else "audit_bad"] += 1
            _audit_calls[0] += 1
            if _audit_calls[0] <= AUDIT_LOG_MAX:
                diff = [i for i, (a, b) in enumerate(zip(hit[0], stamp)) if a != b]
                _LOG("audit layer=%s key=%s equal=%s cached_stamp_vs_now_diff_idx=%s"
                     % (layer.layer_id, key, same, diff))
        if is_source:
            S.stats["src"] += 1
        elif matched and not report_only:
            S.stats["hit"] += 1
            _verify_calls[0] += 1
            want_verify = (_verify_enabled() if (_verify_calls[0] % _VERIFY_CHECK_EVERY == 1)
                           else S.verify_on)
            S.verify_on = want_verify
            if want_verify and not _capturing():
                ref = orig(local_indices, token_to_req_indices, block_table, block_size,
                           is_valid_token)
                if (ref[0].shape == hit[1].shape and ref[1].shape == hit[2].shape
                        and _equal(ref[0], hit[1]) and _equal(ref[1], hit[2])):
                    S.stats["verify_ok"] += 1
                else:
                    S.stats["verify_bad"] += 1
                    _LOG("VERIFY MISMATCH layer %s key %s" % (layer.layer_id, key))
                    return ref
            return hit[1], hit[2]
        elif matched:
            # report-only: the call runs anyway; count what a real run would have saved
            S.stats["hit_noskip"] += 1
        else:
            S.stats["miss"] += 1

        out = orig(local_indices, token_to_req_indices, block_table, block_size, is_valid_token)
        S.cache[key] = (stamp, out[0], out[1])
        return out

    compute_global_topk_indices_and_lens._l2_memo = True
    return compute_global_topk_indices_and_lens


def _equal(a, b):
    import torch
    return bool(torch.equal(a, b))


def _same(a, b):
    try:
        import torch
        return bool(torch.equal(a, b))
    except Exception:
        return False


# ---------------------------------------------------------------- reset + layer context
def _wrap_attention_forward(orig):
    def forward(self, *args, **kwargs):
        if getattr(self, "layer_id", -1) == 0:
            S.reset()
        return orig(self, *args, **kwargs)

    forward._l2 = True
    return forward


def _wrap_decode(orig):
    def _forward_decode(self, *args, **kwargs):
        prev = S.cur
        S.cur = self
        try:
            return orig(self, *args, **kwargs)
        finally:
            S.cur = prev

    _forward_decode._l2 = True
    return _forward_decode


# ---------------------------------------------------------------- installation
def patch_flashmla(module):
    if getattr(module, "_l2_installed", False):
        return
    orig = module.compute_global_topk_indices_and_lens
    if getattr(orig, "_l2_memo", False):
        module._l2_installed = True
        return
    module.compute_global_topk_indices_and_lens = _memo(orig)
    cls = getattr(module, "DeepseekV4FlashMLAAttention", None)
    if cls is not None and not getattr(cls._forward_decode, "_l2", False):
        cls._forward_decode = _wrap_decode(cls._forward_decode)
    module._l2_installed = True


def patch_attention(module):
    if getattr(module, "_l2_installed", False):
        return
    cls = getattr(module, "DeepseekV4Attention", None)
    if cls is not None and not getattr(cls.forward, "_l2", False):
        cls.forward = _wrap_attention_forward(cls.forward)
    module._l2_installed = True


def install():
    import importlib.abc
    import importlib.util

    if getattr(sys, "_l2_installed", False):
        return
    sys._l2_installed = True
    plan = ((FLASHMLA_MODULE, patch_flashmla), (ATTN_MODULE, patch_attention))

    class _Finder(importlib.abc.MetaPathFinder):
        def __init__(self, name, patch):
            self._name = name
            self._patch = patch

        def find_spec(self, name, path, target_mod=None):
            if name != self._name:
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec, _p=self._patch):
                _orig(module)
                try:
                    _p(module)
                except Exception as exc:  # never break the model import
                    _LOG("patch FAILED on %s: %r" % (module.__name__, exc))

            loader.exec_module = exec_module
            return spec

    for name, patch in plan:
        sys.meta_path.insert(0, _Finder(name, patch))
        mod = sys.modules.get(name)
        if mod is not None:
            try:
                patch(mod)
            except Exception as exc:
                _LOG("patch FAILED on %s: %r" % (name, exc))
    _LOG("installed: SHARE=%s VERIFY=%s STATS_EVERY=%s" % (SHARE, int(VERIFY), STATS_EVERY))
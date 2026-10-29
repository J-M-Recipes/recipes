"""CPU unit test for share_topk.py (Card TR2). No torch, no GPU: stub tensors carry data_ptr().

Run: python3 test_share_topk.py
"""
import importlib
import os
import sys
import types


_NEXT = [1000]


class T:
    """Minimal tensor stand-in: a stable unique address + shape + numel, enough for the stamp.

    The address is a counter, not id(): id() is reused by CPython as soon as a temporary is
    freed, which would make two different buffers look identical and hide a real bug.
    """

    def __init__(self, n, tag):
        self._n = n
        self._tag = tag
        self._addr = _NEXT[0]
        _NEXT[0] += 1
        self.shape = (n,)

    def data_ptr(self):
        return self._addr

    def numel(self):
        return self._n

    def __repr__(self):
        return "T(%s)" % self._tag


class Layer:
    def __init__(self, layer_id, kv_src, idx_src, ratio):
        self.layer_id = layer_id
        self.kv_source_layer_id = kv_src
        self.index_source_layer_id = idx_src
        self.compress_ratio = ratio


def load(mode, verify=False, stats=0):
    os.environ["L2_SHARE"] = mode
    os.environ["L2_VERIFY"] = "1" if verify else "0"
    os.environ["L2_STATS_EVERY"] = str(stats)
    if "share_topk" in sys.modules:
        del sys.modules["share_topk"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    return importlib.import_module("share_topk")


def make_module(st):
    calls = []

    def orig(local, t2r, bt, bs, iv):
        calls.append((local, bt))
        return (T(48, "idx"), T(24, "lens"))

    mod = types.SimpleNamespace(compute_global_topk_indices_and_lens=orig,
                                DeepseekV4FlashMLAAttention=type("A", (), {
                                    "_forward_decode": lambda self, *a: None}))
    st.patch_flashmla(mod)
    return mod, calls


_IV = None


def drive(mod, st, layer, local, bt, t2r, bs=512, iv=None):
    """One layer's call. is_valid is `swa_metadata.is_valid_token[:n]` — the same slice for
    every layer of a step — so it is a single object here, not a fresh one per call."""
    global _IV
    if _IV is None:
        _IV = T(24, "valid")
    st.S.cur = layer
    try:
        return mod.compute_global_topk_indices_and_lens(local, t2r, bt, bs, iv or _IV)
    finally:
        st.S.cur = None


def test_relaxed_shares_across_layers():
    st = load("2")
    mod, calls = make_module(st)
    bt = T(24, "bt")
    l_src = Layer(2, 2, 2, 1)
    l_a = Layer(6, 2, 2, 1)
    l_b = Layer(39, 2, 2, 1)
    # the model allocates ONE topk_indices_buffer and hands it to every layer
    # (nvidia/model.py:424 and :462), so `self.topk_indices_buffer[:n]` is the same storage
    local = T(24, "local")
    st.S.reset()
    drive(mod, st, l_src, local, bt, T(24, "t2r"))
    drive(mod, st, l_a, local, bt, T(24, "t2r"))
    drive(mod, st, l_b, local, bt, T(24, "t2r"))
    assert len(calls) == 1, calls          # one kernel for three layers
    assert st.S.stats["src"] == 1 and st.S.stats["hit"] == 2, st.S.stats
    print("ok  relaxed: 3 layers -> 1 call (src 1, hit 2)")


def test_distinct_local_buffer_not_shared():
    st = load("2")
    mod, calls = make_module(st)
    bt = T(24, "bt")
    t2r = T(24, "t2r")
    st.S.reset()
    drive(mod, st, Layer(2, 2, 2, 1), T(24, "local-a"), bt, t2r)
    drive(mod, st, Layer(6, 2, 2, 1), T(24, "local-b"), bt, t2r)
    assert len(calls) == 2, calls
    print("ok  a different local indices buffer -> its own table (no false sharing)")


def test_strict_misses_on_local_buffer():
    st = load("1")
    mod, calls = make_module(st)
    bt = T(24, "bt")
    st.S.reset()
    drive(mod, st, Layer(2, 2, 2, 1), T(24, "l2"), bt, T(24, "t2r"))
    drive(mod, st, Layer(6, 2, 2, 1), T(24, "l6"), bt, T(24, "t2r"))
    assert len(calls) == 2, calls
    print("ok  strict: per-layer local buffer -> miss (no false sharing)")


def test_strict_misses_on_token_to_req_pointer():
    st = load("1")
    mod, calls = make_module(st)
    bt = T(24, "bt")
    local = T(24, "local")
    st.S.reset()
    drive(mod, st, Layer(2, 2, 2, 1), local, bt, T(24, "t2r-a"))
    drive(mod, st, Layer(6, 2, 2, 1), local, bt, T(24, "t2r-b"))
    assert len(calls) == 2, calls
    print("ok  strict: token_to_req identity in the stamp")


def test_distinct_ratio_not_shared():
    st = load("2")
    mod, calls = make_module(st)
    bt = T(24, "bt")
    local = T(24, "local")
    st.S.reset()
    drive(mod, st, Layer(2, 2, 2, 1), local, bt, T(24, "t2r"))
    drive(mod, st, Layer(3, 2, 2, 4), local, bt, T(24, "t2r"))
    assert len(calls) == 2, calls
    print("ok  different compress_ratio -> separate table")


def test_reset_per_forward():
    st = load("2")
    mod, calls = make_module(st)
    bt = T(24, "bt")
    local = T(24, "local")
    st.S.reset()
    drive(mod, st, Layer(2, 2, 2, 1), local, bt, T(24, "t2r"))
    drive(mod, st, Layer(6, 2, 2, 1), local, bt, T(24, "t2r"))
    st.S.reset()                                   # next forward
    drive(mod, st, Layer(2, 2, 2, 1), local, bt, T(24, "t2r"))
    assert len(calls) == 2, calls
    print("ok  new generation -> table recomputed")


def test_block_table_change_invalidates():
    st = load("2")
    mod, calls = make_module(st)
    local = T(24, "local")
    t2r = T(24, "t2r")
    st.S.reset()
    drive(mod, st, Layer(2, 2, 2, 1), local, T(24, "bt-a"), t2r)
    drive(mod, st, Layer(6, 2, 2, 1), local, T(24, "bt-b"), t2r)
    assert len(calls) == 2, calls
    print("ok  block_table identity in the stamp")


def test_report_only_never_skips():
    st = load("0")
    mod, calls = make_module(st)
    bt = T(24, "bt")
    local = T(24, "local")
    t2r = T(24, "t2r")
    st.S.reset()
    for lid in (2, 6, 39):
        drive(mod, st, Layer(lid, 2, 2, 1), local, bt, t2r)
    assert len(calls) == 3, calls                    # behaviour unchanged
    assert st.S.stats["hit_noskip"] == 2, st.S.stats  # but the lever is counted
    print("ok  report-only: every call runs, would-be hits counted (2)")


def test_verify_catches_bad_reuse():
    st = load("2", verify=True)
    mod, calls = make_module(st)
    bt = T(24, "bt")
    local = T(24, "local")
    t2r = T(24, "t2r")
    st.S.reset()
    drive(mod, st, Layer(2, 2, 2, 1), local, bt, t2r)
    # a second layer with a *different* block_size is a different stamp, so force a hit by
    # poisoning the cached table with a wrong value
    key = (2, 2, 1)
    stamp, ind, lens = st.S.cache[key]
    st.S.cache[key] = (stamp, T(1, "bad"), lens)
    drive(mod, st, Layer(6, 2, 2, 1), local, bt, t2r)
    assert st.S.stats["verify_bad"] == 1, st.S.stats
    print("ok  VERIFY: a poisoned cache entry is caught and the original is served")


def test_no_layer_context_passthrough():
    st = load("2")
    mod, calls = make_module(st)
    mod.compute_global_topk_indices_and_lens(T(24, "l"), T(24, "t"), T(24, "b"), 512, T(24, "v"))
    assert len(calls) == 1
    print("ok  no current layer (prefill / warmup) -> passthrough")


if __name__ == "__main__":
    for fn in [test_relaxed_shares_across_layers, test_strict_misses_on_local_buffer,
               test_strict_misses_on_token_to_req_pointer, test_distinct_ratio_not_shared,
               test_reset_per_forward, test_block_table_change_invalidates,
               test_report_only_never_skips, test_verify_catches_bad_reuse,
               test_no_layer_context_passthrough]:
        fn()
    print("\nALL PASS")
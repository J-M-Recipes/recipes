"""Explicit opt-in: RUN_SLOT_CACHE_CUDA_TESTS=1 pytest tests/test_slotcache_dma_cuda.py -q.

Requires CUDA PyTorch, Triton, a C++ compiler, Ninja and CUDA headers/libraries.
The mapped-UVA case additionally uses the recipe's vLLM helper.
"""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

if os.environ.get("RUN_SLOT_CACHE_CUDA_TESTS") != "1":
    pytest.skip("set RUN_SLOT_CACHE_CUDA_TESTS=1 on a CUDA host", allow_module_level=True)

import torch

PATCHES = Path(__file__).resolve().parents[1] / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/patches"
sys.path.insert(0, str(PATCHES))
from slot_cache_dma import DmaRowCopier


@pytest.fixture
def hook(monkeypatch):
    monkeypatch.setenv("SLOT_CACHE_ROUTER", "vllm")
    monkeypatch.setenv("SLOT_CACHE_STATS_SEC", "0")
    spec = importlib.util.spec_from_file_location("slot_cache_cuda_test", PATCHES / "slot_cache_hook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copy_pointer_kinds_and_immediate_slot_reuse():
    compute = torch.cuda.Stream()
    with torch.cuda.stream(compute):
        host = torch.arange(8 * 4096, dtype=torch.int64).reshape(8, 4096).pin_memory()
        device = host.to("cuda")
        destinations = [torch.empty((3, 4096), dtype=torch.int64, device="cuda") for _ in range(2)]
        copier = DmaRowCopier([host, device], destinations)
        assert [item["copy_kind"] for item in copier.describe()] == ["H2D", "D2D"]
        snapshots = []
        for expert in [0, 6, 3, 7, 2]:
            # The previous reader and these writes must complete before overwriting slot 1.
            for dst in destinations:
                dst.fill_(-1)
            copier.copy([expert], [1])
            snapshots.append([dst.clone() for dst in destinations])
        for expert, snapshot in zip([0, 6, 3, 7, 2], snapshots):
            for dst in snapshot:
                actual = dst.cpu()
                assert torch.equal(actual[1], host[expert])
                assert (actual[[0, 2]] == -1).all()
    compute.synchronize()


def test_mapped_uva_view_uses_cuda_host_alias():
    utils = pytest.importorskip("vllm.utils.torch_utils")
    host = torch.arange(8 * 4096, dtype=torch.int64).reshape(8, 4096).pin_memory()
    mapped = utils.get_accelerator_view_from_cpu_tensor(host)
    dst = torch.empty((3, 4096), dtype=torch.int64, device="cuda")
    copier = DmaRowCopier([mapped], [dst])
    assert mapped.is_cuda
    assert copier.describe()[0]["copy_kind"] == "H2D"
    copier.copy([7], [2])
    assert torch.equal(dst[2].cpu(), host[7])


def test_invalid_banks_and_indices_fail_before_writing():
    src = torch.ones((8, 16), dtype=torch.int64).pin_memory()
    dst = torch.zeros((3, 16), dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="contiguous"):
        DmaRowCopier([src[:, ::2]], [dst[:, ::2]])
    with pytest.raises(RuntimeError, match="unsupported source memory type"):
        DmaRowCopier([torch.ones(src.shape, dtype=src.dtype)], [dst])
    copier = DmaRowCopier([src], [dst])
    for sources, destinations, message in [([1, 8], [0, 1], "expert index"), ([1], [3], "slot index"),
                                           ([1, 2], [0, 0], "duplicate"), ([1], [], "count mismatch")]:
        with pytest.raises(RuntimeError, match=message):
            copier.copy(sources, destinations)
    torch.cuda.synchronize()
    assert torch.count_nonzero(dst).item() == 0


def make_cache(hook, backend):
    # Exercise the real fused bookkeeping and forward without loading a model.
    lc = hook.LayerCache.__new__(hook.LayerCache)
    lc.E, lc.S, lc.SB, lc.dev = 8, 3, 4, torch.device("cuda")
    lc.e2s = torch.full((9,), -1, dtype=torch.int32, device="cuda")
    lc.s2e = torch.full((4,), 8, dtype=torch.int32, device="cuda")
    lc.last = torch.full((3,), -1, dtype=torch.int64, device="cuda")
    lc.step = torch.zeros((), dtype=torch.int64, device="cuda")
    lc.misses = torch.zeros_like(lc.step)
    lc.bufs = {}
    sources = [torch.arange(8 * n, dtype=torch.int64).reshape(8, n).cuda() for n in (1024, 512, 128, 64)]
    destinations = [torch.zeros((3, src.shape[1]), dtype=torch.int64, device="cuda") for src in sources]
    lc.rows = dict(zip(("w13", "w2"), zip(sources[:2], destinations[:2])))
    lc.rows_res = dict(zip(("w13_scale", "w2_scale"), zip(sources[2:], destinations[2:])))
    lc.scal_pad = [torch.arange(9, dtype=torch.float32, device="cuda") * scale for scale in (1, 2, 3)]
    lc.scal_slot_pad = [torch.zeros(4, device="cuda") for _ in range(3)]
    lc.dma = DmaRowCopier(sources, destinations) if backend == "dma" else None
    return lc, sources, destinations


def test_dma_matches_triton_cold_hits_duplicates_evictions_and_scales(hook):
    caches = [make_cache(hook, backend) for backend in ("triton", "dma")]
    # N <= S, with duplicates, all hits, mixed hits/misses, and full churn.
    routes = [[0, 1, 1], [0, 1, 2], [2, 1, 0], [3, 2, 3], [4, 5, 6], [5, 5, 4]]
    for route in routes:
        ids = torch.tensor(route, dtype=torch.int32, device="cuda")
        slots = []
        for lc, sources, destinations in caches:
            slot = hook._cache_forward(lc, ids, len(route)).long()
            slots.append(slot.clone())
            for src, dst in zip(sources, destinations):
                assert torch.equal(dst[slot].cpu(), src[ids.long()].cpu())
            for index, scale in enumerate((1, 2, 3)):
                assert torch.equal(lc.scal_slot_pad[index][slot].cpu(), ids.float().cpu() * scale)
        assert torch.equal(slots[0], slots[1])
        assert torch.equal(caches[0][0].e2s, caches[1][0].e2s)
        assert torch.equal(caches[0][0].misses, caches[1][0].misses)


def test_forward_rejects_capture_before_bookkeeping(hook, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="--enforce-eager"):
        hook._cache_forward(SimpleNamespace(dma=object()), None, 1)

"""CPU tests for orchestration; actual CUDA copies are tested separately."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PATCHES = Path(__file__).resolve().parents[1] / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/patches"


@pytest.fixture
def dma(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    spec = importlib.util.spec_from_file_location("dma_unit", PATCHES / "slot_cache_dma.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_miss_compaction_ignores_hit_sentinels(dma):
    assert dma.miss_pairs([[5, 1, 1], [256, 112, 0], [7, 3, 1]]) == ([5, 7], [1, 3])
    assert dma.miss_pairs([[256, 112, 0]]) == ([], [])


def test_copy_orders_producers_transfer_and_consumers(dma):
    calls = []
    compute = SimpleNamespace(wait_event=lambda event: calls.append(("consumer_wait", event)))
    dma.torch.cuda.current_stream = lambda device: compute
    copier = dma.DmaRowCopier.__new__(dma.DmaRowCopier)
    copier.device = "cuda:0"
    copier.ready = SimpleNamespace(record=lambda stream: calls.append(("ready", stream)))
    copier.done = SimpleNamespace(record=lambda stream: calls.append(("done", stream)))
    copier.stream = SimpleNamespace(cuda_stream=123, wait_event=lambda event: calls.append(("copy_wait", event)))
    copier.plan = SimpleNamespace(enqueue=lambda *args: calls.append(("copy", args)))
    copier.copy([5, 7], [1, 3])
    assert calls == [
        ("ready", compute), ("copy_wait", copier.ready),
        ("copy", ([5, 7], [1, 3], 123)), ("done", copier.stream),
        ("consumer_wait", copier.done),
    ]
    calls.clear()
    copier.copy([], [])
    assert calls == []


def test_capture_rejected_even_on_all_hit_path(dma):
    dma.torch.cuda.is_current_stream_capturing = lambda: True
    copier = dma.DmaRowCopier.__new__(dma.DmaRowCopier)
    with pytest.raises(RuntimeError, match="--enforce-eager"):
        copier.copy([], [])
    with pytest.raises(RuntimeError, match="--enforce-eager"):
        copier.copy_misses({})


def test_metadata_read_and_compaction(dma):
    calls = []
    dma.torch.stack = lambda values, dim: SimpleNamespace(
        cpu=lambda: SimpleNamespace(tolist=lambda: [[2, 1, 1], [8, 3, 0]]))
    copier = dma.DmaRowCopier.__new__(dma.DmaRowCopier)
    copier.copy = lambda src, dst: calls.append((src, dst))
    copier.copy_misses({"src": "src", "dst": "dst", "mask": "mask"})
    assert calls == [([2], [1])]


def test_forward_uses_dma_without_launching_triton_row_copy(dma, monkeypatch):
    monkeypatch.setitem(sys.modules, "slot_cache_dma", dma)
    monkeypatch.setenv("SLOT_CACHE_ROUTER", "vllm")
    monkeypatch.setenv("SLOT_CACHE_COPY_BACKEND", "dma")
    spec = importlib.util.spec_from_file_location("hook_unit", PATCHES / "slot_cache_hook.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append("bookkeeping")

    class ForbiddenCopy:
        def __getitem__(self, grid):
            pytest.fail("DMA must not launch Triton row copies")

    hook._ensure_triton = lambda: (None, Kernel(), ForbiddenCopy())
    buffers = dict(src=None, dst=None, mask=None, slot="remapped")
    lc = SimpleNamespace(
        dma=SimpleNamespace(copy_misses=lambda b: calls.append("dma")),
        bufs_for=lambda n: buffers, e2s=None, s2e=None, last=None, step=None,
        misses=None, S=3, E=8, SB=4, scal_pad=[], scal_slot_pad=[],
    )
    assert hook._cache_forward(lc, None, 3) == "remapped"
    assert calls == ["bookkeeping", "dma"]
    calls.clear()
    dma.torch.cuda.is_current_stream_capturing = lambda: True
    with pytest.raises(RuntimeError, match="--enforce-eager"):
        hook._cache_forward(lc, None, 3)
    assert calls == []

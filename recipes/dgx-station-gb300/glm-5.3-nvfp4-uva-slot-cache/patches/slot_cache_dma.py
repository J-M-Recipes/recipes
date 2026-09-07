"""Opt-in eager DMA copies. The default slot-cache path never imports this module."""
from pathlib import Path

import torch

_extension = None
_streams = {}


def require_eager():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SLOT_CACHE_COPY_BACKEND=dma requires --enforce-eager; "
                           "miss descriptors must be read by the host")


def extension():
    global _extension
    if _extension is None:
        require_eager()
        from torch.utils.cpp_extension import load
        _extension = load(
            name="slot_cache_dma_ext",
            sources=[str(Path(__file__).with_suffix(".cpp"))],
            with_cuda=True,
            extra_cflags=["-O2"],
        )
    return _extension


def miss_pairs(descriptors):
    """Compact fused-bookkeeping descriptors; sentinel rows on hits are never copied."""
    sources, destinations = [], []
    for src, dst, mask in descriptors:
        if mask:
            sources.append(src)
            destinations.append(dst)
    return sources, destinations


class DmaRowCopier:
    def __init__(self, sources, destinations):
        require_eager()
        self.device = destinations[0].device
        # Keep the original views, not _rows64(contiguous()) materializations.
        self.sources = list(sources)
        self.destinations = list(destinations)
        self.plan = extension().RowCopier(self.sources, self.destinations)
        with torch.cuda.device(self.device):
            if self.device not in _streams:
                _streams[self.device] = torch.cuda.Stream(device=self.device, priority=-1)
            self.stream = _streams[self.device]
            self.ready = torch.cuda.Event()
            self.done = torch.cuda.Event()

    def describe(self):
        return self.plan.describe()

    def copy(self, sources, destinations):
        """Called in the consuming compute stream; calls for a layer must be serialized.

        Order prior consumers/producers -> copy -> next consumer without a device-wide
        synchronization. Keep this object and its backing allocations alive until the
        consuming stream completes (LayerCache owns them for the model's lifetime).
        """
        require_eager()
        if not sources:
            return
        compute = torch.cuda.current_stream(self.device)
        self.ready.record(compute)
        self.stream.wait_event(self.ready)
        self.plan.enqueue(sources, destinations, self.stream.cuda_stream)
        self.done.record(self.stream)
        compute.wait_event(self.done)

    def copy_misses(self, buffers):
        require_eager()
        # This D2H read synchronizes the current stream, including bookkeeping.
        # It is deliberately visible here: CUDA graph replay cannot schedule these
        # data-dependent host calls. Even an all-hit step pays this metadata cost.
        descriptors = torch.stack((buffers["src"], buffers["dst"], buffers["mask"]), dim=1)
        sources, destinations = miss_pairs(descriptors.cpu().tolist())
        self.copy(sources, destinations)

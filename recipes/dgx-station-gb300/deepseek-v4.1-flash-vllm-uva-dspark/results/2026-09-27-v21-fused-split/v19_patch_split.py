"""v19 patch: fuse the six-op torch split_ids() into one Triton kernel.
Applied to a COPY of the v18 hook (never the shared ~/pin-hot-experts/hook).
Semantics identical: hot_ids = rm[id] if id>=0 and rm[id]>=0 else -1; cold_ids = -rm[id]-1 if id>=0 and rm[id]<0 else -1.
Env PIN_SPLIT_FUSED=0 falls back to the torch path (same-boot A/B without a rebuild).
"""
import re, sys
src_path, dst_path = sys.argv[1], sys.argv[2]
s = open(src_path).read()

old = s[s.index("def split_ids(ids: torch.Tensor, row_map: torch.Tensor):"):s.index("def _uva_view(cpu: torch.Tensor)")]
new = '''SPLIT_FUSED = os.environ.get("PIN_SPLIT_FUSED", "1").strip() not in ("0", "false", "no")
_split_kernel = None


def _get_split():
    global _split_kernel
    if _split_kernel is not None:
        return _split_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _sk(ids_ptr, rm_ptr, hot_ptr, cold_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        i = tl.load(ids_ptr + offs, mask=m, other=-1)
        valid = i >= 0
        ic = tl.where(valid, i, 0)
        rm = tl.load(rm_ptr + ic, mask=m, other=0)
        is_hot = valid & (rm >= 0)
        is_cold = valid & (rm < 0)
        neg1 = tl.full([BLOCK], -1, tl.int32)
        tl.store(hot_ptr + offs, tl.where(is_hot, rm, neg1), mask=m)
        tl.store(cold_ptr + offs, tl.where(is_cold, -rm - 1, neg1), mask=m)

    def launch(ids, row_map):
        ids_c = ids.contiguous()
        n = ids_c.numel()
        hot = torch.empty_like(ids_c)
        cold = torch.empty_like(ids_c)
        BLOCK = 1024
        _sk[(triton.cdiv(n, BLOCK),)](ids_c, row_map, hot, cold, n, BLOCK=BLOCK)
        return hot, cold

    _split_kernel = launch
    return launch


def split_ids_torch(ids: torch.Tensor, row_map: torch.Tensor):
    """Reference (v15-v18) six-op remap. ids [T,K] int32, row_map [384] int32."""
    valid = ids >= 0
    clamped = ids.clamp_min(0).long()
    rm = row_map[clamped]
    is_hot = valid & (rm >= 0)
    is_cold = valid & (rm < 0)
    neg1 = torch.full_like(ids, -1)
    hot_ids = torch.where(is_hot, rm, neg1)
    cold_ids = torch.where(is_cold, -rm - 1, neg1)
    return hot_ids, cold_ids


def split_ids(ids: torch.Tensor, row_map: torch.Tensor):
    """Graph-safe remap. Returns hot_ids, cold_ids (both [T,K] int32, -1 for skip).
    v19: one Triton kernel (PIN_SPLIT_FUSED=1, default) instead of six torch ops."""
    if SPLIT_FUSED and ids.dtype == torch.int32 and row_map.dtype == torch.int32:
        return _get_split()(ids, row_map)
    return split_ids_torch(ids, row_map)


'''
assert old.count("def split_ids") == 1
s = s.replace(old, new)
# log the mode at install time next to the existing summary line
s = s.replace('f"counter={COUNTER if MODE == \'adaptive\' else \'-\'}"',
              'f"counter={COUNTER if MODE == \'adaptive\' else \'-\'} split_fused={SPLIT_FUSED}"')
assert "split_fused=" in s
open(dst_path, "w").write(s)
print("patched", dst_path, len(s))

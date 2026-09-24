#!/usr/bin/env python3
"""W3a offline unit tests — run INSIDE lmsysorg/sglang:v0.5.20-cu130 with the two patched files bind-mounted.
Small tensors only (a few MB of GPU), safe beside a serving container.

T1  patched modules import; tags present; env gate parses.
T2  greedy accept at width K is the exact truncation of width 7:  accept_K = min(accept_7, K-1),
    bonus_K = target_top1[accept_K], out_tokens_K[:, :commit_K] == out_tokens_7[:, :commit_K]
    — i.e. a narrower verify commits a prefix of what the full verify would commit (lossless in exact arithmetic).
    Checked on the Triton kernel the server uses and on the eager reference, 20k random rows.
T3  DSA KPool tail ring: a request advanced by N=5 verify writes + random accepts produces byte-identical
    compressed index pages whether the tail ring is sized POOL+5 (stock-legal) or POOL+7 (what the W3a boot has).
T4  relaxed kpool assert: pool.tail_extra_slots=7 accepts num_draft_tokens=5 and still rejects 8.
"""
import os, random, sys, types
import torch

dev = "cuda"
print("device:", torch.cuda.get_device_name(0), flush=True)
ok = True
def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    ok = ok and bool(cond)

# ---------------- T1
import sglang.srt.speculative.dflash_worker_v2 as W
import sglang.srt.layers.attention.dsa.kpool_plan as KP
src = open(W.__file__).read(); ksrc = open(KP.__file__).read()
check("T1 dflash_worker_v2 patched", "PATCHED (jmeadlock W3a" in src and "_build_verify_width_runtime" in src)
check("T1 kpool_plan patched", "PATCHED (jmeadlock W3a" in ksrc and "tail_extra_slots >= num_draft_tokens" in ksrc)

# ---------------- T2
from sglang.kernels.ops.speculative.dflash import _compute_dflash_accept_bonus_triton_unchecked as tri
from sglang.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus as eager

def run_tri(cand, top1, prefix):
    bs, w = cand.shape
    a = torch.empty(bs, dtype=torch.int32, device=dev); c = torch.empty_like(a)
    b = torch.empty(bs, dtype=torch.int64, device=dev); o = torch.empty((bs, w), dtype=torch.int64, device=dev)
    n = torch.empty(bs, dtype=torch.int64, device=dev)
    tri(candidates=cand, target_top1=top1, accept_lens_out=a, commit_lens_out=c, bonus_ids_out=b,
        out_tokens_out=o, prefix_lens=prefix, new_seq_lens_out=n)
    return a, c, b, o, n

g = torch.Generator(device="cpu").manual_seed(0)
bs, B, V = 20000, 7, 6  # tiny vocab -> many partial matches
cand = torch.randint(0, V, (bs, B), generator=g).to(dev)
top1 = torch.randint(0, V, (bs, B), generator=g).to(dev)
# force a spread of true accept lengths: copy a random-length prefix so drafts match the target
L = torch.randint(0, B, (bs,), generator=g)
for j in range(1, B):
    m = (L >= j).to(dev)
    cand[:, j] = torch.where(m.bool(), top1[:, j - 1], cand[:, j])
prefix = torch.randint(100, 5000, (bs,), generator=g).to(dev)
a7, c7, b7, o7, n7 = run_tri(cand, top1, prefix)
for K in (5, 4, 3, 2):
    cK = cand[:, :K].contiguous(); tK = top1[:, :K].contiguous()
    aK, commK, bK, oK, nK = run_tri(cK, tK, prefix)
    exp_a = torch.clamp(a7, max=K - 1)
    exp_b = torch.gather(top1, 1, exp_a.long()[:, None]).squeeze(1)
    pref_ok = all(bool((oK[i, : commK[i]] == o7[i, : commK[i]]).all()) if a7[i] < K - 1 else
                  bool((oK[i, : commK[i] - 1] == o7[i, : commK[i] - 1]).all())
                  for i in range(0, bs, 97))
    ea, eb = eager(candidates=cK, target_predict=tK)
    check(f"T2 K={K} accept==min(accept7,K-1)", bool((aK == exp_a).all()))
    check(f"T2 K={K} bonus==top1[accept]", bool((bK == exp_b).all()))
    check(f"T2 K={K} commit prefix of full-width commit", pref_ok)
    check(f"T2 K={K} new_seq_lens", bool((nK == prefix + commK.long()).all()))
    check(f"T2 K={K} triton==eager", bool((ea.to(torch.int32) == aK).all() and (eb == bK).all()))
hist = torch.bincount(a7.cpu(), minlength=B).tolist()
print(f"     T2 accept7 histogram {hist}")

# ---------------- T3
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    kpool_write_tail_and_maybe_compress, INDEX_HEAD_DIM, BLOCK_SIZE_K)
POOL = 4; SPP = BLOCK_SIZE_K // POOL  # pooled slots per page
BUF_PER_PAGE = SPP * INDEX_HEAD_DIM + SPP * 4
N = 5; STEPS = 40; NREQ = 3
ape = torch.randn(POOL, INDEX_HEAD_DIM, generator=torch.Generator().manual_seed(1)).float().to(dev)

def simulate(tail_extra):
    pool = types.SimpleNamespace(index_kpool=POOL, tail_extra_slots=tail_extra, slots_per_page=SPP,
                                 index_head_dim=INDEX_HEAD_DIM, page_size=BLOCK_SIZE_K)
    T = POOL + tail_extra
    tail_k = torch.zeros(NREQ + 1, T, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=dev)
    tail_s = torch.zeros_like(tail_k)
    pages = 64
    buf = torch.zeros(pages, BUF_PER_PAGE, dtype=torch.uint8, device=dev)
    rng = random.Random(7); gk = torch.Generator().manual_seed(11)
    seq = [rng.randint(0, 9) for _ in range(NREQ)]  # committed lengths
    for step in range(STEPS):
        key = torch.randn(NREQ * N, INDEX_HEAD_DIM, generator=gk).to(torch.bfloat16).to(dev)
        score = torch.randn(NREQ * N, INDEX_HEAD_DIM, generator=gk).to(torch.bfloat16).to(dev)
        ws = torch.tensor(seq, dtype=torch.int32, device=dev)
        base0 = (ws // POOL) * POOL
        maxp = (N + POOL - 1) // POOL
        # pooled-slot location for logical pool id p of request r: disjoint per request
        wl = torch.stack([(ws // POOL + p) + 100 * torch.arange(NREQ, device=dev, dtype=torch.int32) for p in range(maxp)], 1)
        ocl = torch.ones(NREQ * N, dtype=torch.int64, device=dev)  # nonzero -> row active
        kpool_write_tail_and_maybe_compress(
            pool, buf, key, score, tail_k, tail_s, ape,
            torch.arange(1, NREQ + 1, dtype=torch.int64, device=dev), ws, base0.to(torch.int32),
            wl.to(torch.int64).contiguous(), ocl, num_draft_tokens=N, round_scale=False)
        for r in range(NREQ):
            seq[r] += rng.randint(1, N)  # accept 0..N-1 drafts + bonus
    torch.cuda.synchronize()
    return buf.cpu()

b_ref = simulate(N)       # ring = POOL + 5  (stock-legal: tail_extra == N)
b_w3a = simulate(7)       # ring = POOL + 7  (W3a boot: sized for the full block)
check("T3 kpool compressed pages identical (ring POOL+5 vs POOL+7, N=5, 40 steps x 3 reqs)",
      torch.equal(b_ref, b_w3a), f"nonzero_bytes={int((b_ref != 0).sum())}")

# ---------------- T4
class FB:  # minimal forward_batch for the assert path only
    pass
def assert_path(extra, n):
    fb = FB(); fb.forward_mode = types.SimpleNamespace(is_target_verify=lambda: True, is_decode_or_idle=lambda: False,
                                                       is_draft_extend_v2=lambda: False)
    fb.token_to_kv_pool = types.SimpleNamespace(tail_extra_slots=extra)
    fb.seq_lens = torch.zeros(1, dtype=torch.int64, device=dev)
    try:
        KP.init_kpool_write_plan(None, fb, pool_size=4, real_page_size=64, real_page_table=None,
                                 num_draft_tokens=n, write_start=None, slots_per_page=SPP)
    except AssertionError:
        return "assert"
    except Exception:
        return "past-assert"
    return "past-assert"
check("T4 tail 7 accepts N=5", assert_path(7, 5) == "past-assert")
check("T4 tail 7 accepts N=7", assert_path(7, 7) == "past-assert")
check("T4 tail 7 rejects N=8", assert_path(7, 8) == "assert")

print("W3A_UNIT " + ("ALL_PASS" if ok else "FAILURES"))
sys.exit(0 if ok else 1)

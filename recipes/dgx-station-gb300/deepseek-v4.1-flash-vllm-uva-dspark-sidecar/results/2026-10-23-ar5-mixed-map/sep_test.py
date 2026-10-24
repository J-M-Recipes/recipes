#!/usr/bin/env python3
"""v25 adaptive feasibility: are the GB300 (DeepGEMM MegaMoE) and RTX (b12x) expert
layout transforms separable per expert?  If transforming one expert alone gives the
same bytes as its slice of a whole-bank transform, a hot<->cold swap can be done by
transforming a single expert and copying it into its row.

  side=gb300 : deep_gemm.transform_sf_into_required_layout + transform_weights_for_mega_moe
  side=rtx   : b12x fm.plan_weights + fm.prepare_weights (w4a8_mx, fp4_e8m0_k32, w31)
"""
import json
import sys

import torch

SIDE = sys.argv[1]
G, I, H = 6, 2304, 5120
torch.manual_seed(0)
dev = torch.device("cuda", 0)


def rand_bank(g):
    return {
        "w13": torch.randint(0, 256, (g, 2 * I, H // 2), dtype=torch.uint8, device=dev),
        "s13": torch.randint(112, 131, (g, 2 * I, H // 32), dtype=torch.uint8, device=dev),
        "w2": torch.randint(0, 256, (g, H, I // 2), dtype=torch.uint8, device=dev),
        "s2": torch.randint(112, 131, (g, H, I // 32), dtype=torch.uint8, device=dev),
    }


def desc(t):
    return {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype), "contig": t.is_contiguous()}


out = {"side": SIDE}
bank = rand_bank(G)
pick = 3

if SIDE == "gb300":
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts as M
    from vllm.utils.deep_gemm import _import_deep_gemm

    dg = _import_deep_gemm()
    ue = M._ue8m0_uint8_to_float

    def xf(b, g):
        a = dg.transform_sf_into_required_layout(ue(b["s13"]).contiguous(), 2 * I, H, (1, 32), g)
        c = dg.transform_sf_into_required_layout(ue(b["s2"]).contiguous(), H, I, (1, 32), g)
        l1, l2 = dg.transform_weights_for_mega_moe((b["w13"].view(torch.int8).contiguous(), a),
                                                   (b["w2"].view(torch.int8).contiguous(), c))
        return {"l1w": l1[0], "l1s": l1[1], "l2w": l2[0], "l2s": l2[1]}

    full = xf({k: v.clone() for k, v in bank.items()}, G)
    one = xf({k: v[pick:pick + 1].clone() for k, v in bank.items()}, 1)
    res = {}
    for k in full:
        f, o = full[k], one[k]
        same_lead = f.shape[0] == G and o.shape[0] == 1
        eq = bool(same_lead and torch.equal(f[pick].contiguous().view(torch.uint8), o[0].contiguous().view(torch.uint8)))
        res[k] = {"full": desc(f), "one": desc(o), "slice_equal": eq}
    out["tensors"] = res
    out["separable"] = all(v["slice_equal"] for v in res.values())

else:
    sys.path.insert(0, "/b12x")
    import b12x.moe.fused_moe as fm

    LIMIT = 10.0

    def prep(b, g):
        ones = torch.ones(g, dtype=torch.float32, device=dev)
        one = torch.ones((), dtype=torch.float32, device=dev)
        plan = fm.plan_weights(
            source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"), w13_layout=fm.W13Layout("w31")),
            activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16,
                                         swiglu_limit=LIMIT, swiglu_alpha=None, swiglu_beta=None),
            geometry=fm.MoEGeometry(num_experts=g, hidden_size=H, intermediate_size=I))
        ex = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(
            w13=b["w13"], w2=b["w2"],
            w13_block_scales=b["s13"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w2_block_scales=b["s2"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w13_global_scales=ones, w2_global_scales=ones, input_scale=one, intermediate_scale=one,
            immutable_input_scales=True))
        return ex

    bf = {k: v.clone() for k, v in bank.items()}
    b1 = {k: v[pick:pick + 1].clone() for k, v in bank.items()}
    exf = prep(bf, G)
    ex1 = prep(b1, 1)

    def tensors(obj, prefix="", seen=None, acc=None):
        seen = set() if seen is None else seen
        acc = {} if acc is None else acc
        if id(obj) in seen:
            return acc
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            acc[prefix] = obj
            return acc
        fields = []
        if hasattr(obj, "__dataclass_fields__"):
            fields = list(obj.__dataclass_fields__)
        elif hasattr(obj, "__dict__"):
            fields = list(vars(obj))
        elif isinstance(obj, (list, tuple)):
            for i, x in enumerate(obj):
                tensors(x, f"{prefix}[{i}]", seen, acc)
            return acc
        for f in fields:
            try:
                tensors(getattr(obj, f), f"{prefix}.{f}" if prefix else f, seen, acc)
            except Exception:
                pass
        return acc

    tf, t1 = tensors(exf), tensors(ex1)
    res = {}
    for k, f in tf.items():
        o = t1.get(k)
        if o is None:
            res[k] = {"full": desc(f), "one": None}
            continue
        r = {"full": desc(f), "one": desc(o)}
        if f.dim() >= 1 and f.shape[0] == G and o.shape[0] == 1:
            r["slice_equal"] = bool(torch.equal(f[pick].reshape(-1).contiguous().view(torch.uint8),
                                                o[0].reshape(-1).contiguous().view(torch.uint8)))
        elif f.numel() == o.numel():
            r["whole_equal"] = bool(torch.equal(f.reshape(-1).contiguous().view(torch.uint8),
                                                o.reshape(-1).contiguous().view(torch.uint8)))
        res[k] = r
    out["tensors"] = res
    out["inplace_storage"] = {k: int(bf[k].data_ptr()) for k in bf}
    out["prepared_ptrs"] = {k: int(v.data_ptr()) for k, v in tf.items()}
    out["separable"] = all(v.get("slice_equal", v.get("whole_equal", False)) for v in res.values() if v.get("one"))
    out["type"] = type(exf).__name__

print(json.dumps(out, indent=1))

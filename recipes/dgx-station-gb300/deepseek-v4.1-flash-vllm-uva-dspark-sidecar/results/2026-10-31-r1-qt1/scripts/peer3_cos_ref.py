#!/usr/bin/env python3
"""peer3_cos_ref.py — offline fp64 reference for the peer-tier-v3 cosine gate (Card R1, bars C1–C3).

Reads a peer3_cos_tap capture (<cap>.jsonl + <cap>.bin) and, per accepted call, recomputes the RTX's cold rows with
Garner's exact reference recipe (fastpath/peer3/tests/ref.py: MXFP8 input, fp64 GEMMs, bf16 rounding at MegaMoE's
points, fp32 top-k-order sum, bf16 out) and compares them to the bf16 rows the RTX wrote back.

Cold ids in the segment are cold-LOCAL indices (0xFF = no cold pick); local index i of layer L is
rowmap["layers"][str(L)]["cold"][i] (peer_server3.py:243 stacks experts in that order). Only the experts a sampled
call touches are loaded (a full layer is ~2.3 GB).

Run CPU-only inside the serving image (torch is not on the host), e.g.
  docker run --rm --network none --cpus 16 -v $MODEL:/model:ro -v $TREE/fastpath:/fp:ro -v $CAPDIR:/cap \
    --entrypoint python3 vllm/vllm-openai:v0.31.0 /cap/peer3_cos_ref.py --cap /cap/tap --model /model \
    --rowmap /fp/hook/rowmap-mix-v1-h255.json --ref /fp/peer3/tests --out /cap/cos.json
Bars (R1-PREREG): C1 40/40 layers with >= 4 samples and >= 48 cold (token,slot) pairs; C2 min per-layer cos >= 0.9990;
C3 max |delta| <= 2e-2. Also reported: rows exactly equal (bf16 bit-identical) fraction.
"""
import argparse, json, os, sys

HIDDEN, SCALES, TOPK = 5120, 160, 6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--rowmap", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max-per-layer", type=int, default=6)
    ap.add_argument("--selfcheck", action="store_true", help="verify ref math on one call twice (determinism)")
    a = ap.parse_args()
    sys.path.insert(0, a.ref)
    import torch, ref
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    cold = {int(k): v["cold"] for k, v in json.load(open(a.rowmap))["layers"].items()}
    samples = [json.loads(l) for l in open(a.cap + ".jsonl")]
    by_layer = {}
    for s in samples:
        by_layer.setdefault(s["layer"], []).append(s)
    raw = open(a.cap + ".bin", "rb")
    per_layer, worst = {}, {"cos": 1.0, "maxabs": 0.0}
    for L in sorted(by_layer):
        ss = by_layer[L][: a.max_per_layer]
        locs = sorted({e for s in ss for row in s["ids"] for e in row if e != 0xFF})
        if not locs:
            per_layer[L] = {"samples": len(ss), "pairs": 0, "note": "no cold picks"}; continue
        glob = [cold[L][i] for i in locs]
        W = ref.load_experts(a.model, L, glob)
        sub = {loc: j for j, loc in enumerate(locs)}
        num = den_a = den_b = 0.0; maxabs = 0.0; rows = eq = pairs = 0
        for s in ss:
            T = s["T"]; raw.seek(s["off"])
            xq = torch.frombuffer(bytearray(raw.read(T * HIDDEN)), dtype=torch.uint8).view(T, HIDDEN)
            xs = torch.frombuffer(bytearray(raw.read(T * SCALES)), dtype=torch.uint8).view(T, SCALES)
            got = torch.frombuffer(bytearray(raw.read(T * HIDDEN * 2)), dtype=torch.bfloat16).view(T, HIDDEN).double()
            ids = torch.tensor([[sub[e] if e != 0xFF else -1 for e in row] for row in s["ids"]], dtype=torch.long)
            w = torch.tensor(s["w"], dtype=torch.float32)
            exp = ref.cold_rows(W, xq, xs, ids, w)
            has = (ids >= 0).any(dim=1)
            pairs += int((ids >= 0).sum())
            g, e = got[has], exp[has]
            if g.numel() == 0:
                continue
            num += float((g * e).sum()); den_a += float((g * g).sum()); den_b += float((e * e).sum())
            maxabs = max(maxabs, float((g - e).abs().max()))
            rows += int(has.sum()); eq += int((g == e).all(dim=1).sum())
        cos = num / max((den_a * den_b) ** 0.5, 1e-300)
        per_layer[L] = {"samples": len(ss), "pairs": pairs, "rows": rows, "rows_bit_equal": eq,
                        "cos": cos, "maxabs": maxabs, "experts_loaded": len(glob)}
        worst["cos"] = min(worst["cos"], cos); worst["maxabs"] = max(worst["maxabs"], maxabs)
        print(f"layer {L}: samples {len(ss)} pairs {pairs} cos {cos:.6f} maxabs {maxabs:.3e} bit-equal {eq}/{rows}", flush=True)
    n_ok = sum(1 for v in per_layer.values() if v.get("samples", 0) >= 4 and v.get("pairs", 0) >= 48)
    gates = {
        "C1_coverage": ["PASS" if len(per_layer) == 40 and n_ok == 40 else "FAIL", f"{len(per_layer)}/40 layers, {n_ok} with >=4 samples & >=48 pairs"],
        "C2_min_cos": ["PASS" if worst["cos"] >= 0.9990 else "FAIL", f"{worst['cos']:.6f} >= 0.9990"],
        "C3_max_abs": ["PASS" if worst["maxabs"] <= 2e-2 else "FAIL", f"{worst['maxabs']:.3e} <= 2e-2"],
    }
    tot_rows = sum(v.get("rows", 0) for v in per_layer.values()); tot_eq = sum(v.get("rows_bit_equal", 0) for v in per_layer.values())
    json.dump({"gates": gates, "worst": worst, "rows_bit_equal": [tot_eq, tot_rows], "per_layer": per_layer},
              open(a.out, "w"), indent=1)
    print(json.dumps({"gates": gates, "rows_bit_equal": [tot_eq, tot_rows]}, indent=1))
    return 0 if all(v[0] == "PASS" for v in gates.values()) else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""ed_analyze.py — GLM-5.3-Flash expert-routing skew -> what an RTX cold-expert band would carry (Card ED, 2026-11-01).
Runs inside the serving image (torch available, CPU only):
  docker run --rm --network none -v <dir>:/w --entrypoint python3 <image> /w/ed_analyze.py /w/dump-A /w/dump-B /w/ed-analysis.json
Inputs: SGLang expert_distribution_recorder 'stat' dumps (.pt), buffer_size -1 => logical_count [steps, layers, experts].
A = synthetic fixtures (ranking set). B = Hermes transcript replay (holdout). If B's dump is cumulative (starts with A's rows),
A's rows are stripped from it.
Per N_cold (experts per MoE layer moved to the RTX; hot set = top 288-N by A's decode counts, per layer):
  cold_share      = fraction of decode token->expert assignments that land on cold experts (work the RTX does)
  layer_hit_rate  = fraction of (decode step, MoE layer) pairs with >= 1 cold assignment (=> a PCIe round trip that layer)
  step_any_rate   = fraction of decode steps touching any cold expert in any layer
Decode steps = steps whose per-layer assignment count <= 8 * DECODE_TOK_MAX (default 7-token verify x 16 running = 112 tokens).
"""
import glob, json, os, sys
import torch

DEC_TOK = int(os.getenv("DECODE_TOK_MAX", "112")); TOPK = 8


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "*.pt")))
    if not fs:
        raise SystemExit(f"no dumps in {d}")
    x = torch.load(fs[-1], map_location="cpu", weights_only=False)["logical_count"]
    return x.to(torch.int64), fs[-1]


def split_decode(x):
    per_layer = x.sum(-1)                      # [steps, layers]
    tok = per_layer.max(-1).values // TOPK     # tokens per step (MoE layers carry tokens*8)
    keep = (tok > 0) & (tok <= DEC_TOK)
    return x[keep], int(keep.sum()), int((tok > DEC_TOK).sum())


def analyze(rank_src, eval_src, moe_layers):
    rank_c = rank_src.sum(0)                   # [layers, experts]
    res = {}
    E = rank_c.shape[-1]
    for n in (16, 32, 64, 96, 128, 144):
        order = torch.argsort(rank_c, dim=-1, descending=True)
        cold = torch.zeros_like(rank_c, dtype=torch.bool)
        cold.scatter_(1, order[:, E - n:], True)
        ev = eval_src[:, moe_layers, :]
        cm = cold[moe_layers]
        cold_assign = (ev * cm).sum(-1)        # [steps, moe_layers]
        tot = ev.sum(-1).clamp(min=1)
        res[n] = {"cold_share": round(float(cold_assign.sum() / tot.sum()), 4),
                  "layer_hit_rate": round(float((cold_assign > 0).float().mean()), 4),
                  "step_any_rate": round(float((cold_assign.sum(-1) > 0).float().mean()), 4),
                  "worst_layer_cold_share": round(float((cold_assign.sum(0) / ev.sum((0, 2)).clamp(min=1)).max()), 4)}
    return res


def skew(c, moe_layers):
    s = c.sum(0)[moe_layers].float()
    srt, _ = torch.sort(s, dim=-1, descending=True)
    cum = srt.cumsum(-1) / srt.sum(-1, keepdim=True)
    return {f"top{k}_coverage_mean": round(float(cum[:, k - 1].mean()), 4) for k in (16, 32, 64, 128, 192, 224, 256)}


def main():
    A, fa = load(sys.argv[1]); B, fb = load(sys.argv[2]); out = sys.argv[3]
    cumulative = B.shape[0] >= A.shape[0] and torch.equal(B[: A.shape[0]], A)
    if cumulative:
        B = B[A.shape[0]:]
    moe_layers = [l for l in range(A.shape[1]) if int(A[:, l].sum()) > 0]
    if not moe_layers:
        raise SystemExit(f"ALL-ZERO counts in {fa} ({tuple(A.shape)}): recorder saw no routing (fused-routing MoE backend?)")
    Ad, a_n, a_pf = split_decode(A); Bd, b_n, b_pf = split_decode(B)
    r = {"files": [fa, fb], "B_was_cumulative": cumulative, "layers_total": A.shape[1], "moe_layers": len(moe_layers),
         "experts": A.shape[2], "A_decode_steps": a_n, "A_prefill_steps": a_pf, "B_decode_steps": b_n, "B_prefill_steps": b_pf,
         "decode_tok_max": DEC_TOK,
         "skew_A_decode": skew(Ad, moe_layers), "skew_B_decode": skew(Bd, moe_layers),
         "holdout_rankA_evalB": analyze(Ad, Bd, moe_layers),
         "insample_rankB_evalB": analyze(Bd, Bd, moe_layers),
         "insample_rankA_evalA": analyze(Ad, Ad, moe_layers)}
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("B_was_cumulative", "moe_layers", "A_decode_steps", "B_decode_steps", "skew_B_decode")}))
    print("N_cold | holdout cold_share layer_hit step_any | in-sample(B) cold_share layer_hit")
    for n, v in r["holdout_rankA_evalB"].items():
        w = r["insample_rankB_evalB"][n]
        print(f"{n:>6} | {v['cold_share']:.4f} {v['layer_hit_rate']:.4f} {v['step_any_rate']:.4f} | {w['cold_share']:.4f} {w['layer_hit_rate']:.4f}")


if __name__ == "__main__":
    main()

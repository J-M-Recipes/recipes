#!/usr/bin/env python3
"""ranking_portability.py — is the RL-derived hot list still right for the MOPD router?  (Q2 of harness/protocol-v2.yaml)

usage: python3 ranking_portability.py RANKING.json LIVE_MOPD.json [LIVE_RL.json] [budget_cells=8692] [--json OUT]
  RANKING   = the counts hotsplit booted with (expert_hist_mix.json, {"train":{"decode":{layer:[E]}}})
  LIVE_MOPD = live counter snapshot from the m-hot boot ({"live":{"decode":{layer:[E]}}})
  LIVE_RL   = optional: a live snapshot from an RL boot (live/counts-20260922.json) for the same-model noise floor

Reports, using the SAME greedy per-byte cell selection hotsplit.py uses (all MoE layers share one cell size, so this is
just top-K cells by normalised per-layer share):
  hot set from RANKING (K cells)         -> share of MOPD live decode traffic those cells serve  ("HBM-served share")
  hot set from LIVE_MOPD itself (K cells) -> oracle upper bound for this budget
  Jaccard overlap of the two hot sets, per-layer share correlation (Spearman-free: Pearson on normalised shares), and
  the same numbers for LIVE_RL vs RANKING so the MOPD number has a same-model reference.
Stdlib only (box host python has no numpy/torch)."""
import json, sys, math

def load(p, key):
    d = json.load(open(p)); d = d[key]["decode"]
    return {int(k): [float(x) for x in v] for k, v in d.items()}
def norm(v):
    s = sum(v) or 1.0; return [x / s for x in v]
def hotset(counts, K):
    cells = sorted(((s, l, e) for l, v in counts.items() for e, s in enumerate(norm(v))), reverse=True)[:K]
    return {(l, e) for _, l, e in cells}
def served(hot, traffic):
    tot = sum(sum(v) for v in traffic.values()) or 1.0
    return sum(traffic[l][e] for l, e in hot if l in traffic) / tot
def pearson(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else float("nan")

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; out = None
    if "--json" in sys.argv: out = sys.argv[sys.argv.index("--json") + 1]; args = [a for a in args if a != out]
    rank = load(args[0], "train"); mopd = load(args[1], "live")
    rl = load(args[2], "live") if len(args) > 2 and args[2].endswith(".json") else None
    K = int(args[3]) if len(args) > 3 else (int(args[2]) if len(args) > 2 and not args[2].endswith(".json") else 8692)
    layers = sorted(set(rank) & set(mopd))
    toks = min(sum(mopd[l]) for l in layers) / 8
    res = {"K": K, "layers": len(layers), "mopd_live_tokens_per_layer": round(toks)}
    hr = hotset({l: rank[l] for l in layers}, K); hm = hotset({l: mopd[l] for l in layers}, K)
    res["hbm_served_share_rl_ranking_on_mopd"] = served(hr, mopd)
    res["hbm_served_share_oracle_mopd"] = served(hm, mopd)
    res["jaccard_rlrank_vs_mopd_oracle"] = len(hr & hm) / len(hr | hm)
    res["pearson_share_rlrank_vs_mopd"] = sum(pearson(norm(rank[l]), norm(mopd[l])) for l in layers) / len(layers)
    res["stock_layer_order_share_on_mopd"] = served({(l, e) for l in layers if l >= 49 for e in range(len(mopd[l]))}, mopd)
    if rl:
        lr = sorted(set(rank) & set(rl)); hrl = hotset({l: rl[l] for l in lr}, K)
        res["rl_live_tokens_per_layer"] = round(min(sum(rl[l]) for l in lr) / 8)
        res["hbm_served_share_rl_ranking_on_rl_live"] = served(hotset({l: rank[l] for l in lr}, K), rl)
        res["hbm_served_share_oracle_rl_live"] = served(hrl, rl)
        res["jaccard_rlrank_vs_rl_oracle"] = len(hotset({l: rank[l] for l in lr}, K) & hrl) / len(hotset({l: rank[l] for l in lr}, K) | hrl)
        res["pearson_share_rlrank_vs_rl_live"] = sum(pearson(norm(rank[l]), norm(rl[l])) for l in lr) / len(lr)
        res["jaccard_mopd_oracle_vs_rl_oracle"] = len(hm & hrl) / len(hm | hrl)
    res["verdict"] = "PORTABLE" if res["hbm_served_share_rl_ranking_on_mopd"] >= 0.55 else "RE-RANK"
    if toks < 50000: res["verdict"] += " (UNDERPOWERED: <50k tokens/layer)"
    for k, v in res.items(): print(f"{k:45s} {v if not isinstance(v, float) else f'{v:.4f}'}")
    if out: json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__": main()

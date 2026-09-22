"""Analyse expert_hist2.json: does a train-set hot list generalise to held-out Hermes turns?
Reports holdout decode-traffic coverage of the hot set at several budgets, vs stock layer-granular placement."""
import json, sys, torch
r = json.load(open(sys.argv[1]))
E = 384
def load(split, phase):
    return {int(k): torch.tensor(v, dtype=torch.float64) for k, v in r[split][phase].items()}
tr_d, tr_p = load("train", "decode"), load("train", "prefill")
ho_d, ho_p = load("holdout", "decode"), load("holdout", "prefill")
L = sorted(tr_d)
print("meta", r["meta"])
print("decode routed tokens: train", int(sum(v.sum() for v in tr_d.values()) / 8 / len(L)),
      "holdout", int(sum(v.sum() for v in ho_d.values()) / 8 / len(L)))
def norm(d): return {k: v / v.sum() for k, v in d.items()}
def score(mix):  # per-layer normalised score
    return {l: mix[0] * norm(tr_d)[l] + mix[1] * norm(tr_p)[l] for l in L}
tot_ho = sum(ho_d[l].sum() for l in L)
# stock: HBM layers are the ones NOT offloaded; v18 log: layers 49..69 in HBM (21 of 69)
stock_hbm = [l for l in L if l >= 49]
stock_cov = sum(ho_d[l].sum() for l in stock_hbm) / tot_ho
print(f"stock placement (layers 49-69 resident, {len(stock_hbm)*E} cells): holdout decode coverage {stock_cov:.1%}")
for mix in [(1, 0), (1, 0.25), (1, 1), (0, 1)]:
    s = score(mix)
    cells = sorted(((s[l][e].item(), l, e) for l in L for e in range(E)), reverse=True)
    out = []
    for frac in [0.20, len(stock_hbm) / len(L), 0.40, 0.50]:
        k = int(frac * len(cells))
        cov = sum(ho_d[l][e].item() for _, l, e in cells[:k]) / tot_ho
        out.append(f"{frac:.0%}->{cov:.1%}")
    print(f"hot list from train mix decode={mix[0]} prefill={mix[1]}: holdout decode coverage  " + "  ".join(out))
# oracle (holdout ranks itself) at stock budget
k = len(stock_hbm) * E
oc = sorted((ho_d[l][e].item() for l in L for e in range(E)), reverse=True)
print(f"oracle (holdout-ranked) at stock budget: {sum(oc[:k]) / tot_ho:.1%}")

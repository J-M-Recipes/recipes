import json, sys, torch
h2=json.load(open("/w/dumps/expert_hist2.json")); h1=json.load(open("/w/dumps/expert_hist.json"))
L=sorted(int(k) for k in h2["train"]["decode"])
def n(t): t=torch.tensor(t,dtype=torch.float64); return t/t.sum()
ho={l:torch.tensor(h2["holdout"]["decode"][str(l)],dtype=torch.float64) for l in L}
tot=sum(v.sum() for v in ho.values())
stock_k=21*384
for w in (0.0,0.25,0.5,1.0):
    s={l: n(h2["train"]["decode"][str(l)]) + w*n(h1[str(l)]["counts"]) for l in L}
    cells=sorted(((s[l][e].item(),l,e) for l in L for e in range(384)),reverse=True)[:stock_k]
    cov=sum(ho[l][e].item() for _,l,e in cells)/tot
    # prose proxy coverage: hist1 counts themselves (in-sample for w>0, so only indicative)
    p={l:n(h1[str(l)]["counts"]) for l in L}; pc=sum(p[l][e].item() for _,l,e in cells)/len(L)
    print(f"w_prose={w}: holdout agent decode cov {cov:.1%}  hist1 (prose-ish) cov {pc:.1%}")
    if w==0.25:
        json.dump({"train":{"decode":{str(l):(s[l]*1e6).tolist() for l in L}}}, open("/w/dumps/expert_hist_mix.json","w"))
# stock prose coverage
p={l:n(h1[str(l)]["counts"]) for l in L}; print("stock hist1 cov", f"{sum(p[l].sum().item() for l in L if l>=49)/len(L):.1%}")

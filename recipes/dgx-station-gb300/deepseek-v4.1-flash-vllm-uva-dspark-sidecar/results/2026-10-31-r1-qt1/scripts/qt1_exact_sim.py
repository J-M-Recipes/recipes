# null + power check for qt1_exact: synthesize emitted tokens from the RECORDED distributions (T1.0a), with ties kept
import json, math, random, sys, os
sys.path.insert(0, ".")
import qt1_exact as qe
src = [json.loads(l) for l in open("run1/gen-T1.0a.jsonl")]
def synth(seed, eps=0.0, path="/tmp/qt1_syn.jsonl"):
    rng = random.Random(seed); f = open(path, "w")
    for r in src:
        ids, tls = [], []
        for j, tl in enumerate(r["top_logprobs"]):
            if not tl: continue
            items = sorted(((int(k), float(v["logprob"])) for k, v in tl.items()), key=lambda x: -x[1])
            em = r["token_ids"][j]
            top = [x for x in items if x[0] != em][:20] if len(items) > 20 else items[:20]
            ps = [math.exp(lp) for _, lp in top]; tail = max(0.0, 1 - sum(ps))
            u = rng.random(); acc = 0.0; pick = None
            if eps and rng.random() < eps: pick = top[0][0]           # argmax-substitution bias model
            else:
                for (t, _), p in zip(top, ps):
                    acc += p
                    if u < acc: pick = t; break
            if pick is None: pick = -1                                # tail token (not in top-20)
            d = {str(t): {"logprob": lp} for t, lp in top}
            if pick == -1: d["-1"] = {"logprob": math.log(max(tail, 1e-30))}
            ids.append(pick); tls.append(d)
        f.write(json.dumps({"cls": r["cls"], "token_ids": ids, "top_logprobs": tls}) + "\n")
    f.close(); return qe.analyse([path])
res = {"null": [], "eps": {}}
for s in range(40):
    o = synth(s); res["null"].append((o["R_rank_calibration"]["p"], o["M_midpit_mean"]["z"]))
for eps in (0.005, 0.01, 0.02):
    res["eps"][eps] = [synth(1000 + s, eps)["R_rank_calibration"]["p"] for s in range(10)]
nullR = [p for p, _ in res["null"]]; nullM = [z for _, z in res["null"]]
print("null R: frac p<0.01 =", sum(p < 0.01 for p in nullR) / len(nullR), " frac p<0.05 =", sum(p < 0.05 for p in nullR) / len(nullR))
print("null M: frac |z|>3 =", sum(abs(z) > 3 for z in nullM) / len(nullM), " sd(z) =", round((sum(z*z for z in nullM)/len(nullM))**0.5, 2))
for eps, ps in res["eps"].items(): print(f"power eps={eps}: frac p<0.01 = {sum(p < 0.01 for p in ps)/len(ps)}  median p = {sorted(ps)[len(ps)//2]:.3g}")
json.dump(res, open("run1-A1/exact_sim.json", "w"))

#!/usr/bin/env python3
"""tf_decode.py — decode-path divergence for speculative builds.
tf_kl.py scores prompt_logprobs (prefill only) and never touches the spec-decode path. This one does:
for each corpus piece take the first PREFIX tokens as a forced prefix, generate GEN tokens greedily with
logprobs, and record per-position (token, logprob, top-5). Compare two runs position-by-position over the
common prefix of generated tokens (once outputs diverge, later positions are conditioned on different
text and are not comparable). Metrics: tokens generated, positions compared, first-divergence position
per piece, mean/max |dlp| on compared positions, top1 disagreement on compared positions.

  tf_decode.py run <label> [--prefix 256] [--gen 64]     -> runs/decode/<label>.json
  tf_decode.py compare <A> <B_ref>
"""
import json, os, sys, time, urllib.request
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:30001/v1"); KEY = os.environ.get("API_KEY", ""); MODEL = os.environ.get("MODEL", "glm-5.3-big")
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "runs", "decode"); os.makedirs(OUT, exist_ok=True)
CORPUS = os.path.join(D, "runs", "kl", "corpus.json")

def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

SETTLE = float(os.environ.get("TF_DECODE_SETTLE", "0.3"))   # seconds between requests: strictly one in flight, scheduler drains

def run(label, prefix, gen, limit=None):
    pieces = json.load(open(CORPUS))["pieces"][:limit]
    out = []; t0 = time.time()
    for i, p in enumerate(pieces):
        ids = p["ids"][:prefix]
        time.sleep(SETTLE)
        r = post("/completions", {"model": MODEL, "prompt": ids, "max_tokens": gen, "temperature": 0, "logprobs": 5, "seed": 0})
        ch = r["choices"][0]; lp = ch["logprobs"]
        toks = lp["tokens"]; lps = lp["token_logprobs"]; tops = lp["top_logprobs"]
        out.append({"i": i, "cat": p["cat"], "n": len(toks), "tokens": toks, "lps": lps,
                    "top1": [max(t, key=t.get) if t else None for t in tops], "finish": ch.get("finish_reason")})
        if i % 10 == 0: print(f"  {i}/{len(pieces)} {time.time()-t0:.0f}s", file=sys.stderr)
    json.dump({"label": label, "prefix": prefix, "gen": gen, "pieces": out}, open(os.path.join(OUT, f"{label}.json"), "w"))
    print(f"RUN_DECODE {label} " + json.dumps({"pieces": len(out), "gen_tokens": sum(o["n"] for o in out), "sec": round(time.time()-t0)}))

def compare_data(a, b):
    A = json.load(open(os.path.join(OUT, f"{a}.json")))["pieces"]; B = json.load(open(os.path.join(OUT, f"{b}.json")))["pieces"]
    n_cmp = 0; dl = []; top1_dis = 0; first_div = []; identical = 0; per_cat = {}
    for x, y in zip(A, B):
        n = min(x["n"], y["n"]); k = 0
        while k < n and x["tokens"][k] == y["tokens"][k]: k += 1
        first_div.append(k if k < n else None)
        if k == n and x["n"] == y["n"]: identical += 1
        for j in range(k):  # compared positions: same forced history
            d = abs((x["lps"][j] or 0) - (y["lps"][j] or 0)); dl.append(d); n_cmp += 1
            if x["top1"][j] != y["top1"][j]: top1_dis += 1
            c = per_cat.setdefault(x["cat"], {"n": 0, "sum": 0.0, "dis": 0}); c["n"] += 1; c["sum"] += d; c["dis"] += x["top1"][j] != y["top1"][j]
    dl.sort()
    return {"A": a, "B_ref": b, "pieces": len(A), "identical_pieces": identical,
            "positions_compared": n_cmp, "mean_abs_dlp": sum(dl)/max(1, n_cmp), "p99_abs_dlp": dl[int(0.99*len(dl))] if dl else 0, "max_abs_dlp": dl[-1] if dl else 0,
            "top1_disagree_rate": top1_dis/max(1, n_cmp), "top1_disagree_n": top1_dis,
            "first_divergence_pos": {"none": sum(1 for f in first_div if f is None), "min": min([f for f in first_div if f is not None], default=None),
                                     "median": sorted([f for f in first_div if f is not None])[len([f for f in first_div if f is not None])//2] if any(f is not None for f in first_div) else None},
            "per_cat": {c: {"n": v["n"], "mean_abs_dlp": v["sum"]/max(1, v["n"]), "top1_disagree_rate": v["dis"]/max(1, v["n"])} for c, v in per_cat.items()}}

def compare(a, b):
    res = compare_data(a, b)
    print("COMPARE_DECODE " + json.dumps(res))
    return res

def gate(candidate, ref_a, ref_b):
    """Apply the approved self-repeat-floor decode gate.

    The reference floor is ref_a vs ref_b.  To avoid choosing the friendlier
    reference run, the candidate is compared with both and the worse value is
    used for each gate metric.  Coverage is reported separately because all
    metrics stop at the first generated-token divergence; it is not silently
    treated as evidence of equivalence.
    """
    floor = compare_data(ref_a, ref_b)
    crosses = [compare_data(candidate, ref_a), compare_data(candidate, ref_b)]
    worst_mean = max(r["mean_abs_dlp"] for r in crosses)
    worst_top1 = max(r["top1_disagree_rate"] for r in crosses)
    min_positions = min(r["positions_compared"] for r in crosses)
    floor_positions = floor["positions_compared"]
    result = {
        "candidate": candidate,
        "references": [ref_a, ref_b],
        "floor": {
            "mean_abs_dlp": floor["mean_abs_dlp"],
            "top1_disagree_rate": floor["top1_disagree_rate"],
            "positions_compared": floor_positions,
            "identical_pieces": floor["identical_pieces"],
        },
        "candidate_worst": {
            "mean_abs_dlp": worst_mean,
            "top1_disagree_rate": worst_top1,
            "min_positions_compared": min_positions,
            "coverage_vs_floor": min_positions / max(1, floor_positions),
            "identical_pieces_min": min(r["identical_pieces"] for r in crosses),
        },
        "metric_pass": worst_mean <= floor["mean_abs_dlp"] and worst_top1 <= floor["top1_disagree_rate"],
        "comparisons": crosses,
    }
    print("DECODE_NONINFERIORITY " + json.dumps(result))
    return result

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        a = sys.argv[3:]; pre = int(a[a.index("--prefix")+1]) if "--prefix" in a else 256; gen = int(a[a.index("--gen")+1]) if "--gen" in a else 64
        lim = int(a[a.index("--limit")+1]) if "--limit" in a else None
        run(sys.argv[2], pre, gen, lim)
    elif cmd == "compare": compare(sys.argv[2], sys.argv[3])
    elif cmd == "gate": gate(sys.argv[2], sys.argv[3], sys.argv[4])

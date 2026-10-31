#!/usr/bin/env python3
"""qt1_analyze.py — Card QT1 analysis (M4 side). stdlib only: runs on /usr/bin/python3.

Reads the box-side artifacts written by qt1_gen.py (gen-<arm>.jsonl, promptids.jsonl) and
qt1_rescore.py (rescore-<arm>.jsonl, tf20-*.jsonl) and decides the PREREG bars (see PREREG.md).

THE INSTRUMENT (QT1-A). For every emitted position j the server returned the decode-path top-20
(token ids + raw logprobs) plus the emitted token's own logprob. At T=1, top_p=1, top_k=-1 the
processed distribution equals the raw one, and the top-20 logprobs the rejection sampler returns are
computed from the TARGET logits at that verified position. So the recorded bins are the target
conditional's top-20, coarsened. Take
    bins = {top-1} .. {top-20} singleton, plus one tail bin with mass t = 1 - sum(top-20)
    u = F_{r-1} + V * p_r          if the emitted token is top-k at rank r (0-based)
    u = F_20   + V * t             if it is outside the top-20
with V ~ U(0,1) drawn fresh per position. Under H0 ("the emitted token is an exact sample of the
target conditional") u ~ U(0,1) EXACTLY, at any coarsening, at any temperature. That is the whole
trick: it needs no full vocab, and it is immune to how peaked the distribution is.

TESTS (alpha handling in PREREG.md):
  A1 KS on u                     pooled (primary) + per class (Holm)
  A2 chi-square, 10 equal bins   pooled diagnostic
  A3 dependence: lag-1 autocorrelation of (u-0.5) and a runs test. A #61161-class bug (one uniform
     shared by every position in a verified block) leaves the MARGINAL of u perfectly uniform, so A1
     cannot see it and this family is the only route to it.
  A4 NLL vs entropy: mean(-log p(x)) must sit in [mean H_partition - eps, mean H_partition + tail
     bound]; H_partition = -sum_i p_i log p_i - t log t is a LOWER bound on the true entropy.
  A5 top-1 calibration: fraction(emitted == argmax) vs mean(max prob), paired z.

NOT TESTED HERE: the accept/reject arithmetic itself (that is the G3 GPU unit test), speed, the
tail's internal SHAPE beyond its aggregate mass, and any temperature the run did not sample at.

Usage
  qt1_analyze.py --dir RUN --out OUT                       # full run
  qt1_analyze.py --tf-a tf20-Pg61.jsonl --tf-b tf20-h2REF.jsonl   # QT1-C only
  qt1_analyze.py --selftest                                # no lane, no files
"""
import argparse, hashlib, json, math, os, random, sys

# ---------------------------------------------------------------- PREREG constants (frozen)
# These MUST equal PREREG.md. VERDICT.txt prints them next to the measured value.
BARS = {
    "P1_pooled_ks_alpha": 0.01,            # pooled PIT KS must NOT reject at 1 %
    "P2_perclass_holm_fwer": 0.05,         # no class rejects after Holm across the 5 classes
    "P3_dependence_absz": 3.0,             # |z| of lag-1 autocorr and of the runs test
    "P4_top1_absz": 3.0,                   # |z| of (observed top-1 rate - mean max-prob)
    "P5_nll_lo": -0.02,                    # mean NLL - mean H_partition >= this (nats/token)
    "P5_nll_hi_slack": 0.02,               # ... and <= t*log(V) + this
    "P6_dp_mean_abs_dlp": 0.10,            # decode-vs-prefill admission bound
    "P6_dp_top1_agree": 0.95,
    "P7_tf_mean_abs_dlp": 0.0620,          # h2-REF repeat floor 0.0560 x 1.10
    "P7_tf_top1_flip": 0.0240,             # 2.181 % x 1.10
}
VOCAB = 129280                              # for the tail-mass bound in A4
CLASSES = ["prose", "structured", "code", "shell_ops", "tool_json"]

# ---------------------------------------------------------------- small stats library
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def ks_test(u):
    """Two-sided one-sample KS against U(0,1). Returns (D, p). Stephens-corrected asymptotic."""
    n = len(u)
    if n < 8:
        return float("nan"), float("nan")
    s = sorted(u)
    d = 0.0
    for i, v in enumerate(s):
        d = max(d, (i + 1) / n - v, v - i / n)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    return d, _kolmogorov_q(lam)


def _kolmogorov_q(lam):
    if lam <= 0:
        return 1.0
    if lam < 0.2:
        return 1.0
    s, sign = 0.0, 1.0
    for k in range(1, 101):
        term = 2.0 * sign * math.exp(-2.0 * k * k * lam * lam)
        s += term
        if abs(term) < 1e-13:
            break
        sign = -sign
    return min(1.0, max(0.0, s))


def _gammq(a, x):
    """Regularized upper incomplete gamma Q(a,x) (Numerical Recipes 6.2, series + CF)."""
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    if x < a + 1.0:                                  # series for P, Q = 1-P
        ap, s, dl = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            dl *= x / ap
            s += dl
            if abs(dl) < abs(s) * 1e-14:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b = x + 1.0 - a                                   # continued fraction for Q
    c, d, h = 1e300, 1.0 / b, 1.0 / b
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return min(1.0, max(0.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h))


def chi2_test(u, bins=10):
    """Equal-probability bin chi-square on u. Returns (chi2, dof, p)."""
    n = len(u)
    if n < 10 * bins:
        return float("nan"), bins - 1, float("nan")
    obs = [0] * bins
    for v in u:
        k = int(v * bins)
        obs[min(max(k, 0), bins - 1)] += 1
    e = n / bins
    x2 = sum((o - e) ** 2 / e for o in obs)
    return x2, bins - 1, _gammq((bins - 1) / 2.0, x2 / 2.0)


def autocorr1(x):
    """Lag-1 autocorrelation of x (already centred) and its z = r*sqrt(n)."""
    n = len(x)
    if n < 20:
        return float("nan"), float("nan")
    m = sum(x) / n
    d = [v - m for v in x]
    v0 = sum(v * v for v in d)
    if v0 <= 0:
        return float("nan"), float("nan")
    r = sum(d[i] * d[i + 1] for i in range(n - 1)) / v0
    return r, r * math.sqrt(n)


def runs_test(u):
    """Wald-Wolfowitz runs test on u>0.5 (ties broken at 0.5 exactly are dropped)."""
    s = [1 if v > 0.5 else (0 if v < 0.5 else None) for v in u]
    s = [v for v in s if v is not None]
    n1, n2 = sum(s), len(s) - sum(s)
    n = n1 + n2
    if n1 < 10 or n2 < 10:
        return float("nan"), float("nan")
    r = 1 + sum(1 for i in range(1, n) if s[i] != s[i - 1])
    mu = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    if var <= 0:
        return r, float("nan")
    return r, (r - mu) / math.sqrt(var)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, run = len(pvals), [1.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        out[i] = run
    return out


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))

# ---------------------------------------------------------------- distribution handling
def soften(lps, temp):
    """Apply a temperature to RAW logprobs: l/T, then renormalise over the given support.

    At T=1 this is the identity (no renormalisation error at all: raw==processed, per the pin).
    For T!=1 the server's normaliser over the full vocab is not observable (max_logprobs=20), so the
    renormalisation is over the observed support and the tail mass is recomputed to 0; PREREG.md
    records this as the one approximation in the T!=1 arms, and the measured top-20 mass is printed
    with every table so the reader can see how much support was in hand.
    """
    if abs(temp - 1.0) < 1e-12:
        return list(lps)
    sc = [lp / temp for lp in lps]
    m = max(sc)
    z = sum(math.exp(v - m) for v in sc)
    return [v - m - math.log(z) for v in sc]        # log-probs, normalised over support


def pit_from_bins(tids, lps, sampled_tid, sampled_lp, V, ptail_mode="auto", temp=1.0):
    """Coarsened randomized PIT for one position.

    tids/lps: the top-20 (descending by logprob, as returned). Returns dict(u, in_top, p_bin,
    ptail, mass_top, p_sampled, flag).
    """
    pairs = sorted(zip(lps, tids), reverse=True)[:20]           # enforce descending
    lps2 = soften([p[0] for p in pairs], temp)
    ids = [p[1] for p in pairs]
    ps = [math.exp(v) for v in lps2]
    s = sum(ps)
    if s > 1.0:                                                # rounding only
        ps = [v / s for v in ps]
        s = 1.0
    ptail = max(0.0, 1.0 - s)
    if ptail_mode == "top20" or (ptail_mode == "auto" and abs(temp - 1.0) > 1e-12):
        ptail = 0.0
    flag = ""
    if sampled_tid in ids:
        r = ids.index(sampled_tid)
        f = sum(ps[:r])
        u = f + V * ps[r]
        in_top, pbin = True, ps[r]
    else:
        f = s
        u = f + V * ptail
        in_top, pbin = False, ptail
        # the sampled token's own probability must not exceed the smallest reported top-20 member
        if ps and sampled_lp is not None and math.exp(soften([sampled_lp], temp)[0]) > ps[-1] * 1.0001:
            flag = "sampled>min_top20"
        if ptail <= 0.0:
            flag = (flag + ";tail0").strip(";")
    if not (0.0 <= u <= 1.0):
        u = min(max(u, 0.0), 1.0 - 1e-15)
        flag = (flag + ";clamped").strip(";")
    hpart = -sum(p_ * math.log(p_) for p_ in ps if p_ > 0.0)
    if ptail > 0.0:
        hpart -= ptail * math.log(ptail)
    return {"u": u, "in_top": in_top, "p_bin": pbin, "ptail": ptail, "mass_top": s, "h_part": hpart,
            "p_sampled": math.exp(soften([sampled_lp], temp)[0]) if sampled_lp is not None else None,
            "flag": flag}

# ---------------------------------------------------------------- artifact loading
def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def find(d, prefix):
    return os.path.join(d, prefix) if os.path.exists(os.path.join(d, prefix)) else None


def load_run(run_dir, arms):
    """Returns {arm: [record]} from gen-<arm>.jsonl merged with rescore-<arm>.jsonl if present."""
    out = {}
    for a in arms:
        g = find(run_dir, "gen-%s.jsonl" % a)
        if not g:
            continue
        recs = load_jsonl(g)
        r = find(run_dir, "rescore-%s.jsonl" % a)
        if r:
            rr = {x["prompt_id"]: x for x in load_jsonl(r)}
            for rec in recs:
                rec["_rescore"] = rr.get(rec["prompt_id"])
        out[a] = recs
    return out


def arm_temperature(recs):
    for r in recs:
        return float(r.get("temperature", 1.0))
    return 1.0

# ---------------------------------------------------------------- QT1-A
def analyze_arm(name, recs, temp_score=None, seed=20261031, ptail_mode="auto"):
    """Build the PIT vector for one arm and run the full test battery."""
    T = arm_temperature(recs)
    Ts = T if temp_score is None else temp_score
    rng = random.Random(seed ^ (hash(name) & 0xFFFF))
    rows = []            # per position
    for rec in recs:
        cls = rec.get("cls")
        tl = rec.get("top_logprobs") or []
        tids = rec.get("token_ids") or []
        tlps = rec.get("token_logprobs") or []
        for j, tid in enumerate(tids):
            if j >= len(tl) or not tl[j]:
                continue
            entry = tl[j]
            ids = [int(k) for k in entry.keys()]
            lps = [float(v["logprob"] if isinstance(v, dict) else v) for v in entry.values()]
            # drop a duplicate sampled-token entry if the server appended it (21 entries)
            if len(ids) > 20:
                order = sorted(range(len(ids)), key=lambda i: -lps[i])
                keep = [i for i in order if ids[i] != tid][:20]
                if len(keep) < 20:
                    keep = order[:20]
                ids = [ids[i] for i in keep]
                lps = [lps[i] for i in keep]
            slp = tlps[j] if j < len(tlps) else None
            rows.append(dict(cls=cls, prompt_id=rec.get("prompt_id"),
                             **pit_from_bins(ids, lps, tid, slp, rng.random(),
                                             ptail_mode=ptail_mode, temp=Ts)))
    U = [r["u"] for r in rows]
    res = {"arm": name, "score_temp": Ts, "sampled_temp": T, "positions": len(U),
           "mean_ptail": mean([r["ptail"] for r in rows]),
           "mean_mass_top": mean([r["mass_top"] for r in rows]),
           "frac_outside_top20": mean([0.0 if r["in_top"] else 1.0 for r in rows]) if rows else None,
           "flags": sum(1 for r in rows if r["flag"])}
    if len(U) < 8:
        res["error"] = "too few positions"
        return res, rows
    d, p = ks_test(U)
    x2, dof, p2 = chi2_test(U)
    r1, z1 = autocorr1([v - 0.5 for v in U])
    nr, zr = runs_test(U)
    res.update({"ks_D": d, "ks_p": p, "chi2": x2, "chi2_dof": dof, "chi2_p": p2,
                "autocorr1": r1, "autocorr_z": z1, "runs": nr, "runs_z": zr,
                "mean_u": mean(U), "sd_u": (sum((v - mean(U)) ** 2 for v in U) / max(len(U) - 1, 1)) ** 0.5})
    res["per_class"] = {}
    for c in CLASSES:
        Uc = [r["u"] for r in rows if r["cls"] == c]
        if len(Uc) < 8:
            continue
        dc, pc = ks_test(Uc)
        x2c, dofc, p2c = chi2_test(Uc)
        rc, zc = autocorr1([v - 0.5 for v in Uc])
        nrc, zrc = runs_test(Uc)
        res["per_class"][c] = {"n": len(Uc), "ks_D": dc, "ks_p": pc, "chi2_p": p2c,
                               "autocorr_z": zc, "runs_z": zrc,
                               "mean_ptail": mean([r["ptail"] for r in rows if r["cls"] == c])}
    return res, rows


def analyze_pit_extras(rows):
    """A4 (NLL vs entropy bounds) and A5 (top-1 calibration) need bin masses; computed from rows."""
    nll, hpart, tmass = [], [], []
    for r in rows:
        if not r.get("p_sampled"):
            continue
        nll.append(-math.log(max(r["p_sampled"], 1e-300)))
        t = r["ptail"]
        # H_partition over the reported bins: sum p_bin log p_bin over the 20 singletons + tail
        hpart.append(r.get("h_part", float("nan")))
        tmass.append(t)
    if not nll:
        return {}
    ok = [(a, b, t) for a, b, t in zip(nll, hpart, tmass) if b == b]
    gap = mean([a - b for a, b, _ in ok]) if ok else float("nan")
    bound = mean([t * math.log(VOCAB) for _, _, t in ok]) if ok else float("nan")
    return {"mean_nll": mean(nll), "mean_h_partition": mean([b for _, b, _ in ok]),
            "mean_nll_minus_h": gap, "tail_bound": bound, "positions": len(ok)}


def top1_calibration(recs):
    """A5: fraction(emitted == argmax of the reported top-20) vs mean(max p)."""
    obs, exp, pv = 0, 0.0, []
    for rec in recs:
        tl = rec.get("top_logprobs") or []
        tids = rec.get("token_ids") or []
        for j, tid in enumerate(tids):
            if j >= len(tl) or not tl[j]:
                continue
            entry = tl[j]
            ids = [int(k) for k in entry.keys()]
            lps = [float(v["logprob"] if isinstance(v, dict) else v) for v in entry.values()]
            if len(ids) > 20:
                order = sorted(range(len(ids)), key=lambda i: -lps[i])
                keep = [i for i in order if ids[i] != tid][:20] or order[:20]
                ids = [ids[i] for i in keep]
                lps = [lps[i] for i in keep]
            pm = math.exp(max(lps))
            pv.append(pm)
            exp += pm
            obs += int(tid == ids[lps.index(max(lps))])
    n = len(pv)
    if n < 30:
        return {}
    var = sum(p * (1 - p) for p in pv)
    se = math.sqrt(var) / n if var > 0 else float("nan")
    return {"n": n, "observed_top1_rate": obs / n, "expected_top1_rate": exp / n,
            "diff": obs / n - exp / n, "z": (obs / n - exp / n) / se if se else float("nan")}

# ---------------------------------------------------------------- QT1-B
def decode_vs_prefill(recs):
    """Per-position compare of the decode-path logprob with the prefill (prompt_logprobs) one."""
    dl, agree, n, ranks_moved, jac = [], 0, 0, [], []
    for rec in recs:
        rs = rec.get("_rescore")
        if not rs:
            continue
        tl = rec.get("top_logprobs") or []
        tids = rec.get("token_ids") or []
        tlps = rec.get("token_logprobs") or []
        pos = rs.get("positions") or []
        if len(pos) != len(tids):
            continue
        for j, tid in enumerate(tids):
            if j >= len(tl) or not tl[j]:
                continue
            p = pos[j]
            if p.get("lp") is None or tlps[j] is None:
                continue
            dl.append(abs(float(tlps[j]) - float(p["lp"])))
            ids = [int(k) for k in tl[j].keys()]
            lps = [float(v["logprob"] if isinstance(v, dict) else v) for v in tl[j].values()]
            if len(ids) > 20:
                order = sorted(range(len(ids)), key=lambda i: -lps[i])
                keep = [i for i in order if ids[i] != tid][:20] or order[:20]
                ids = [ids[i] for i in keep]
            arg = ids[lps.index(max(lps))]
            r_ids = [int(x[0]) for x in (p.get("top20") or [])]
            agree += int(bool(r_ids) and arg == r_ids[0])
            if r_ids:
                sa, sb = set(ids), set(r_ids)
                jac.append(len(sa & sb) / max(1, len(sa | sb)))
            if p.get("rank") is not None:
                ranks_moved.append(int(p["rank"]))
            n += 1
    if not n:
        return {}
    return {"positions": n, "mean_abs_dlp": mean(dl),
            "p99_abs_dlp": sorted(dl)[int(0.99 * (len(dl) - 1))],
            "max_abs_dlp": max(dl), "top1_agree": agree / n,
            "mean_jaccard_top20": mean(jac) if jac else None,
            "mean_prefill_rank_of_emitted": mean(ranks_moved) if ranks_moved else None}

# ---------------------------------------------------------------- QT1-C
def tf_compare(a_path, b_path, max_pos=200000):
    """Top-20 forced-token compare of two tf20 captures (same corpus, same positions)."""
    A = {r["doc"]: r for r in load_jsonl(a_path)}
    B = {r["doc"]: r for r in load_jsonl(b_path)}
    dl, n, flips, jac = [], 0, 0, []
    for doc in sorted(set(A) & set(B)):
        pa, pb = A[doc].get("positions") or [], B[doc].get("positions") or []
        for x, y in zip(pa, pb):
            if x.get("lp") is None or y.get("lp") is None:
                continue
            dl.append(abs(float(x["lp"]) - float(y["lp"])))
            ta = [int(t) for t, _ in (x.get("top20") or [])]
            tb = [int(t) for t, _ in (y.get("top20") or [])]
            if ta and tb:
                flips += int(ta[0] != tb[0])
                sa, sb = set(ta), set(tb)
                jac.append(len(sa & sb) / max(1, len(sa | sb)))
            n += 1
            if n >= max_pos:
                break
    if not n:
        return {}
    return {"positions": n, "docs": len(set(A) & set(B)), "mean_abs_dlp": mean(dl),
            "p99_abs_dlp": sorted(dl)[int(0.99 * (len(dl) - 1))], "max_abs_dlp": max(dl),
            "top1_flips": flips, "top1_flip_rate": flips / n,
            "mean_jaccard_top20": mean(jac) if jac else None}

# ---------------------------------------------------------------- verdict
def judge(armres, extras, dp, tf=None):
    bars = []

    def add(name, ok, got, want):
        bars.append({"bar": name, "pass": bool(ok), "measured": got, "prereg": want})

    p = armres.get("ks_p")
    add("P1 pooled PIT KS p >= 0.01", p is not None and p >= BARS["P1_pooled_ks_alpha"],
        p, ">= %s" % BARS["P1_pooled_ks_alpha"])
    pc = armres.get("per_class") or {}
    names = list(pc)
    if names:
        adj = holm([pc[c]["ks_p"] for c in names])
        worst = min(zip(adj, names))
        add("P2 no per-class KS rejects after Holm (FWER 5%)",
            worst[0] >= BARS["P2_perclass_holm_fwer"],
            {"min_holm_p": worst[0], "class": worst[1]}, ">= 0.05")
    z1, zr = armres.get("autocorr_z"), armres.get("runs_z")
    ok3 = z1 is not None and zr is not None and abs(z1) < BARS["P3_dependence_absz"] and abs(zr) < BARS["P3_dependence_absz"]
    add("P3 dependence: |z(autocorr1)|, |z(runs)| < 3", ok3, {"autocorr_z": z1, "runs_z": zr}, "< 3")
    t1 = extras.get("top1") or {}
    ok4 = t1.get("z") is not None and abs(t1["z"]) < BARS["P4_top1_absz"]
    add("P4 top-1 calibration |z| < 3", ok4, t1.get("z"), "< 3")
    e = extras.get("pit") or {}
    gap, bound = e.get("mean_nll_minus_h"), e.get("tail_bound")
    ok5 = gap is not None and gap >= BARS["P5_nll_lo"] and gap <= (bound or 0) + BARS["P5_nll_hi_slack"]
    add("P5 mean NLL - H_partition in [-0.02, tail_bound+0.02]", ok5,
        {"gap": gap, "bound": bound}, "in band")
    if dp:
        add("P6 decode-vs-prefill mean|dlp| <= 0.10 and top-1 agree >= 0.95",
            dp["mean_abs_dlp"] <= BARS["P6_dp_mean_abs_dlp"] and dp["top1_agree"] >= BARS["P6_dp_top1_agree"],
            {"mean_abs_dlp": dp["mean_abs_dlp"], "top1_agree": dp["top1_agree"]}, "0.10 / 0.95")
    if tf:
        add("P7 Pg61-vs-h2REF top-20 TF mean|dlp| <= 0.062 and flips <= 2.40%",
            tf["mean_abs_dlp"] <= BARS["P7_tf_mean_abs_dlp"] and tf["top1_flip_rate"] <= BARS["P7_tf_top1_flip"],
            {"mean_abs_dlp": tf["mean_abs_dlp"], "flip_rate": tf["top1_flip_rate"]}, "0.062 / 0.0240")
    return bars

# ---------------------------------------------------------------- selftest
def selftest():
    """Proof of the PIT construction with no lane: synthetic bins, exact draws, and the
    pathological cases. Also checks the stats library against closed forms."""
    ok = True
    msg = []

    # 1. stats library
    d, p = ks_test([(i + 0.5) / 1000 for i in range(1000)])          # perfectly uniform
    msg.append("ks(uniform grid) D=%.4f p=%.3f" % (d, p))
    ok &= (d < 0.002 and p > 0.5)
    # a perfectly even grid has x2 = 0 by construction, so the chi-square is calibrated on a random
    # uniform draw instead (bar: x2/dof within [0.5, 2.0] and p > 0.01 — a wrong gammq fails both)
    rng2 = random.Random(3)
    ur = [rng2.random() for _ in range(100000)]
    x2, dof, p2 = chi2_test(ur)
    msg.append("chi2(random uniform) x2=%.2f dof=%d p=%.3f" % (x2, dof, p2))
    ok &= (0.5 < x2 / dof < 2.0 and p2 > 0.01)
    msg.append("gammq(1,1)=%.6f (want 0.367879)" % _gammq(1, 1))
    ok &= abs(_gammq(1, 1) - 0.367879441) < 1e-6
    msg.append("kolmogorov_q(1.36)=%.4f (want ~0.0495)" % _kolmogorov_q(1.36))
    ok &= abs(_kolmogorov_q(1.36) - 0.0495) < 0.005
    r, zz = autocorr1([1.0 if i % 2 else -1.0 for i in range(200)])
    msg.append("autocorr1(alternating) r=%.3f z=%.1f (want r~-1)" % (r, zz))
    ok &= r < -0.9

    # 2. PIT exactness on a peaked + flat mixture, scalar path, with coarsening
    rng = random.Random(7)
    for trial in range(3):
        U = []
        for _ in range(4000):
            V = 64
            lps = sorted([rng.gauss(0, 2.0) for _ in range(V)], reverse=True)
            m = max(lps)
            z_ = sum(math.exp(v - m) for v in lps)
            ps = [math.exp(v - m) / z_ for v in lps]
            tids = list(range(V))
            # exact draw from the FULL distribution
            x, acc, pick = rng.random(), 0.0, V - 1
            for i, p_ in enumerate(ps):
                acc += p_
                if x <= acc:
                    pick = i
                    break
            top = 20
            r = pit_from_bins(tids[:top], [math.log(max(p_, 1e-300)) for p_ in ps[:top]],
                              pick if pick < top else tids[-1], math.log(max(ps[pick], 1e-300)),
                              rng.random())
            U.append(r["u"])
        d, p = ks_test(U)
        msg.append("PIT null trial %d: KS D=%.4f p=%.3f mean=%.4f" % (trial, d, p, mean(U)))
        ok &= (p > 0.01 and abs(mean(U) - 0.5) < 0.02)

    # 3. a wrong sampler must be caught: tokens drawn at T=1.3, scored at T=1
    U = []
    for _ in range(4000):
        V = 64
        lps = [rng.gauss(0, 2.0) for _ in range(V)]
        m = max(lps)
        p1 = [math.exp(v - m) for v in lps]
        z1 = sum(p1)
        p1 = [v / z1 for v in p1]
        p13 = [v ** (1 / 1.3) for v in p1]
        z13 = sum(p13)
        p13 = [v / z13 for v in p13]
        x, acc, pick = rng.random(), 0.0, V - 1
        for i, p_ in enumerate(p13):
            acc += p_
            if x <= acc:
                pick = i
                break
        order = sorted(range(V), key=lambda i: -lps[i])          # top-20 of the T=1 distribution
        r = pit_from_bins([order[i] for i in range(20)],
                          [math.log(max(p1[order[i]], 1e-300)) for i in range(20)],
                          pick, math.log(max(p1[pick], 1e-300)), rng.random())
        U.append(r["u"])
    d, p = ks_test(U)
    msg.append("PIT T-mismatch 1.3: KS D=%.4f p=%.2e (must reject)" % (d, p))
    ok &= (p < 1e-6)

    # 4. the #61161 shape: ONE uniform shared by every position of a verified block decides the token
    #    (the server-side reuse), while the analyst still randomises its own V. That is the faithful
    #    model: the emitted tokens inside a block are comonotone. Each position stays marginally
    #    uniform, so the KS marginal test is nearly blind and the dependence family is the route in.
    #    (An earlier version of this selftest shared the analyst's V but sampled each token from a
    #    fresh uniform — that is not the bug, and it is only marginally detectable: z ~ 2.7 at any N.)
    for k in (2, 5):
        U = []
        for _ in range(4000 // k):
            x = rng.random()                                  # ONE uniform per block
            for _j in range(k):
                Vv = 64
                lps = sorted([rng.gauss(0, 2.0) for _ in range(Vv)], reverse=True)
                m = max(lps)
                z_ = sum(math.exp(v - m) for v in lps)
                ps = [math.exp(v - m) / z_ for v in lps]
                acc, pick = 0.0, Vv - 1
                for i, p_ in enumerate(ps):
                    acc += p_
                    if x <= acc:
                        pick = i
                        break
                r = pit_from_bins(list(range(20)), [math.log(max(p_, 1e-300)) for p_ in ps[:20]],
                                  pick, math.log(max(ps[pick], 1e-300)), rng.random())
                U.append(r["u"])
        d, p = ks_test(U)
        _, z1 = autocorr1([v - 0.5 for v in U])
        msg.append("PIT shared-x block k=%d, n=4000: KS p=%.2g  D=%.4f  z(autocorr1)=%.1f  mean(u)=%.3f"
                   % (k, p, d, z1, mean(U)))
        ok &= (abs(z1) > 3.0)

    print(json.dumps({"selftest": "PASS" if ok else "FAIL", "checks": msg}, indent=1))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="run directory with gen-*.jsonl / rescore-*.jsonl")
    ap.add_argument("--out", default=None, help="output directory (default: --dir)")
    ap.add_argument("--arms", default="T1.0a,T1.0b,T0.95,T1.05")
    ap.add_argument("--primary", default="T1.0a,T1.0b", help="arms pooled for the primary test")
    ap.add_argument("--serve-temp", default="", help="arm=T overrides (default: record's temperature)")
    ap.add_argument("--controls", default="T0.95=1.0,T1.05=1.0",
                    help="arm=scoreT extra scorings for the positive controls")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--ptail-mode", default="auto", choices=["auto", "clamp", "top20"])
    ap.add_argument("--tf-a"), ap.add_argument("--tf-b")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.tf_a and a.tf_b:
        r = tf_compare(a.tf_a, a.tf_b)
        bars = judge({}, {}, None, r)
        out = {"tf": r, "bars": bars, "pass": all(b["pass"] for b in bars)}
        print(json.dumps(out, indent=1))
        return 0 if out["pass"] else 1
    if not a.dir:
        ap.error("--dir or --tf-a/--tf-b or --selftest required")

    arms = [x for x in a.arms.split(",") if x]
    run = load_run(a.dir, arms)
    missing = [x for x in arms if x not in run]
    over = {}
    for spec in [s for s in a.serve_temp.split(",") if s]:
        k, v = spec.split("=")
        over[k] = float(v)
    ctl = {}
    for spec in [s for s in a.controls.split(",") if s]:
        k, v = spec.split("=")
        ctl[k] = float(v)

    verdict = {"prereg_bars": BARS, "arms": {}, "controls": {}, "pass": None, "warnings": []}
    if missing:
        verdict["warnings"].append("missing arms: %s" % missing)
    per_arm = {}
    for arm in arms:
        if arm not in run:
            continue
        res, rows = analyze_arm(arm, run[arm], temp_score=over.get(arm), seed=a.seed,
                                ptail_mode=a.ptail_mode)
        res["pit"] = analyze_pit_extras(rows)
        res["top1"] = top1_calibration(run[arm])
        res["decode_vs_prefill"] = decode_vs_prefill(run[arm])
        # per-class NLL/entropy
        res["per_class_pit"] = {}
        for c in CLASSES:
            sub = [r for r in rows if r["cls"] == c]
            if sub:
                res["per_class_pit"][c] = analyze_pit_extras(sub)
        verdict["arms"][arm] = res
        per_arm[arm] = res
        if ctl.get(arm) is not None:
            cres, crows = analyze_arm(arm + "@T%.2f(control)" % ctl[arm], run[arm],
                                      temp_score=ctl[arm], seed=a.seed, ptail_mode=a.ptail_mode)
            cres["pit"] = analyze_pit_extras(crows)
            verdict["controls"][cres["arm"]] = cres

    primary = [x for x in a.primary.split(",") if x in per_arm]
    if primary:
        # pool the primary arms' u vectors for the P1 test: rebuild by re-running with a shared seed
        pooled_rows = []
        for arm in primary:
            _, rows = analyze_arm(arm, run[arm], temp_score=over.get(arm), seed=a.seed,
                                  ptail_mode=a.ptail_mode)
            pooled_rows.extend(rows)
        U = [r["u"] for r in pooled_rows]
        d, p = ks_test(U)
        x2, dof, p2 = chi2_test(U)
        r1, z1 = autocorr1([v - 0.5 for v in U])
        nr, zr = runs_test(U)
        pooled = {"arms": primary, "positions": len(U), "ks_D": d, "ks_p": p, "chi2_p": p2,
                  "autocorr_z": z1, "runs_z": zr,
                  "per_class": {}}
        for c in CLASSES:
            Uc = [r["u"] for r in pooled_rows if r["cls"] == c]
            if len(Uc) > 8:
                dc, pc = ks_test(Uc)
                pooled["per_class"][c] = {"n": len(Uc), "ks_D": dc, "ks_p": pc}
        pooled["pit"] = analyze_pit_extras(pooled_rows)
        pooled["top1"] = top1_calibration([r for arm in primary for r in run[arm]])
        pooled["decode_vs_prefill"] = decode_vs_prefill([r for arm in primary for r in run[arm]])
        verdict["primary"] = pooled
        verdict["bars"] = judge(pooled, pooled, pooled.get("decode_vs_prefill"))
        verdict["pass"] = all(b["pass"] for b in verdict["bars"])

    outdir = a.out or a.dir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    txt = render_txt(verdict)
    with open(os.path.join(outdir, "VERDICT.txt"), "w") as f:
        f.write(txt)
    print(txt)
    return 0 if verdict.get("pass") else 1


def render_txt(v):
    L = []
    L.append("QT1 VERDICT — temperature>0 sampler calibration (PIT), DSV4.1-Flash Pg61 :30006")
    L.append("overall: %s" % ("PASS" if v.get("pass") else ("FAIL" if v.get("pass") is False else "INCOMPLETE")))
    if v.get("warnings"):
        L.append("warnings: %s" % v["warnings"])
    p = v.get("primary") or {}
    if p:
        L.append("")
        L.append("QT1-A primary (pooled %s) N=%d positions" % (p.get("arms"), p.get("positions")))
        L.append("  PIT KS      D=%.5f  p=%.4g" % (p["ks_D"], p["ks_p"]))
        L.append("  chi2 10-bin p=%.4g" % p["chi2_p"])
        L.append("  dependence  z(autocorr1)=%.2f  z(runs)=%.2f" % (p["autocorr_z"], p["runs_z"]))
        t1 = p.get("top1") or {}
        if t1:
            L.append("  top-1 rate  obs=%.4f  exp(max p)=%.4f  z=%.2f"
                     % (t1["observed_top1_rate"], t1["expected_top1_rate"], t1["z"]))
        pe = p.get("pit") or {}
        if pe:
            L.append("  NLL %.4f nats vs H_partition %.4f -> gap %+.4f (tail bound %.4f)"
                     % (pe["mean_nll"], pe["mean_h_partition"], pe["mean_nll_minus_h"], pe["tail_bound"]))
        L.append("  per class:")
        for c, d in (p.get("per_class") or {}).items():
            L.append("    %-11s n=%5d  KS D=%.5f p=%.3g" % (c, d["n"], d["ks_D"], d["ks_p"]))
        dp = p.get("decode_vs_prefill") or {}
        if dp:
            L.append("  QT1-B decode-vs-prefill: n=%d mean|dlp|=%.4f p99=%.3f top1_agree=%.4f "
                     "jac20=%.3f" % (dp["positions"], dp["mean_abs_dlp"], dp["p99_abs_dlp"],
                                     dp["top1_agree"], dp.get("mean_jaccard_top20") or -1))
    if v.get("controls"):
        L.append("")
        L.append("positive controls (a real distribution change must be caught):")
        for k, c in v["controls"].items():
            L.append("  %-22s KS D=%.4f p=%.3g  N=%d" % (k, c["ks_D"], c["ks_p"], c["positions"]))
    if v.get("arms"):
        L.append("")
        L.append("arms:")
        for k, r in v["arms"].items():
            L.append("  %-6s N=%6d  KS p=%.3g  autocorr_z=%.2f  runs_z=%.2f  mean_u=%.4f  "
                     "mass_top20=%.4f  ptail=%.4f  flags=%d"
                     % (k, r["positions"], r["ks_p"], r["autocorr_z"], r["runs_z"], r["mean_u"],
                        r["mean_mass_top"], r["mean_ptail"], r["flags"]))
    if v.get("bars"):
        L.append("")
        L.append("pre-registered bars:")
        for b in v["bars"]:
            got = b["measured"]
            if isinstance(got, float):
                got = "%.5g" % got
            elif isinstance(got, dict):
                got = json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in got.items()})
            L.append("  [%s] %-62s got=%s want=%s" % ("PASS" if b["pass"] else "FAIL",
                                                      b["bar"], got, b["prereg"]))
    L.append("")
    L.append("Do not read this as: the accept/reject arithmetic is correct (G3 GPU unit test), the")
    L.append("distribution is right at temperatures not sampled here, or anything about speed.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())

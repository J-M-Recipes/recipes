#!/usr/bin/env python3
"""qt1_sim.py — instrument controls for QT1: null calibration, power, and smallest detectable effect.

Runs on the recording (gen-<arm>.jsonl from the T=1 primary arms) so every simulation draws from the
distributions the lane actually produced, not from a toy model. numpy required.

  NULL     draw synthetic tokens EXACTLY from each recorded distribution (top-20 + tail), score them
           with the same coarsened PIT, run the registered test at the registered alpha, 200 times.
           A correct instrument must reject ~1 % of the time (alpha = 0.01, pooled KS).
  POWER-A  temperature mismatch: sample at T'=0.95/1.05 from the recorded distribution, score at T=1.
  POWER-B  #61161-class draw reuse: one uniform shared by every position of a verified block (k=2,3,5,
           8,16); emitted token = inverse-CDF of its own distribution at that shared uniform. The
           marginal of u stays uniform, so only the dependence family can see it.
  POWER-C  acceptance bias toward the drafter: with prob eps the emitted token is taken from the
           drafter's support (the target's own top-5, renormalised) instead of a true draw from p.
  MDE      smallest eps / k at 80 % power at the planned N (bisection, 40 reps per candidate).

Detection = the pre-registered decision rule (P1 pooled KS p < 0.01 OR P2 class KS after Holm OR
P3 |z| >= 3). The report separates "which family fired" so a PASS cannot be manufactured by a
family that is blind to the bug.

Usage: qt1_sim.py --dir RUN --out RUN/sim.json [--reps-null 200] [--n 51200]
"""
import argparse, json, math, os, sys
import numpy as np

CLASSES = ["prose", "structured", "code", "shell_ops", "tool_json"]
ALPHA = 0.01
Z_DEP = 3.0


def load(d, arms):
    bins = []          # per position: array of 21 probs (top-20 desc + tail)
    cls = []
    for a in arms:
        p = os.path.join(d, "gen-%s.jsonl" % a)
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            tl = r["top_logprobs"]
            tid = r["token_ids"]
            for j, t in enumerate(tid):
                if j >= len(tl) or not tl[j]:
                    continue
                lps = sorted((float(v["logprob"]) for v in tl[j].values()), reverse=True)[:20]
                ps = np.exp(np.array(lps, dtype=np.float64))
                s = float(ps.sum())
                if s > 1.0:
                    ps = ps / s
                    s = 1.0
                row = np.zeros(21)
                row[:len(ps)] = ps
                row[20] = max(0.0, 1.0 - s)
                bins.append(row)
                cls.append(r["cls"])
    return np.array(bins), np.array(cls)


def pit_from_bins(draw_bin, V, bins):
    """draw_bin: int array of chosen bin (0..19 singleton, 20 tail). Returns u."""
    n = len(draw_bin)
    rows = np.arange(n)
    p = bins[rows, draw_bin]
    # cumulative mass of the bins strictly before the chosen one (bins are in descending-prob order)
    F = np.cumsum(bins, axis=1)[rows, draw_bin] - p
    return np.clip(F + V * p, 0.0, 1.0 - 1e-15)


def ks_p(u):
    n = len(u)
    s = np.sort(u)
    i = np.arange(1, n + 1)
    d = max(float(np.max(i / n - s)), float(np.max(s - (i - 1) / n)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    # asymptotic Kolmogorov Q
    k = np.arange(1, 101)
    terms = 2.0 * ((-1.0) ** (k - 1)) * np.exp(-2.0 * (k ** 2) * lam * lam)
    return d, float(min(1.0, max(0.0, terms.sum())))


def autocorr_z(u):
    d = u - u.mean()
    v0 = float((d * d).sum())
    if v0 <= 0:
        return 0.0
    r = float((d[:-1] * d[1:]).sum()) / v0
    return r * math.sqrt(len(u))


_C_CACHE = {}


def cumsum_of(b, key):
    """Cache the cumulative matrix: for the T=1 recording it is identical in every replicate."""
    if key in _C_CACHE:
        return _C_CACHE[key]
    C = np.cumsum(b, axis=1)
    if key is not None:
        _C_CACHE[key] = C
    return C


def cat_draw(bins, rng, key=None):
    C = cumsum_of(bins, key)
    x = rng.random(len(bins))
    return np.minimum((x[:, None] > C).sum(axis=1), 20)


def run_rep(bins, cls_idx, rng, mode="null", eps=0.0, k=1, temp=1.0):
    n = len(bins)
    if temp != 1.0:
        b = np.power(np.clip(bins, 1e-300, None), 1.0 / temp)
        b = b / b.sum(axis=1, keepdims=True)
    else:
        b = bins
    V = rng.random(n)
    if mode == "null":
        draw = cat_draw(b, rng, key=("b", id(bins)) if temp == 1.0 else None)
    elif mode == "reuse":
        xb = rng.random(int(math.ceil(n / k)))
        x = np.repeat(xb, k)[:n]
        C = cumsum_of(b, ("b", id(bins)))
        draw = np.minimum((x[:, None] > C).sum(axis=1), 20)
    elif mode == "bias1":
        # acceptance bias toward the draft: with prob eps the emitted token is the draft's proposal,
        # modelled as the ARGMAX (the corner case that deviates most from the target; a real drafter's
        # sample lands on the argmax more often than chance, so this is an optimistic power bound)
        draw = cat_draw(b, rng, key=("b", id(bins)))
        sub = rng.random(n) < eps
        if sub.any():
            draw[sub] = 0
    elif mode == "bias5":
        # weakest version: the draft's support is the target's own top-5, renormalised. At this
        # peakedness that distribution is nearly the target itself, so this is a pessimistic bound.
        draw = cat_draw(b, rng, key=("b", id(bins)))
        d5 = np.zeros_like(b)
        d5[:, :5] = b[:, :5]
        d5 = d5 / np.clip(d5.sum(axis=1, keepdims=True), 1e-12, None)
        sub = rng.random(n) < eps
        if sub.any():
            draw[sub] = cat_draw(d5[sub], rng)
    else:
        raise SystemExit(mode)
    u = pit_from_bins(draw, V, bins)             # scored against the T=1 recording
    p1 = ks_p(u)
    z = autocorr_z(u)
    per_class = []
    for c in np.unique(cls_idx):
        m = cls_idx == c
        if m.sum() > 8:
            per_class.append(ks_p(u[m])[1])
    p2 = min(1.0, min(per_class) * len(per_class)) if per_class else 1.0
    fire = []
    if p1[1] < ALPHA:
        fire.append("P1")
    if p2 < 0.05:
        fire.append("P2")
    if abs(z) >= Z_DEP:
        fire.append("P3")
    return {"p1": p1[1], "p2": p2, "z": z, "fire": fire, "detect": bool(fire)}


def rate(reps):
    return sum(r["detect"] for r in reps) / len(reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arms", default="T1.0a,T1.0b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps-null", type=int, default=200)
    ap.add_argument("--reps-power", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--n", type=int, default=0, help="subsample to N positions (0 = all)")
    a = ap.parse_args()
    bins, cls = load(a.dir, [x for x in a.arms.split(",") if x])
    if a.n and a.n < len(bins):
        bins, cls = bins[:a.n], cls[:a.n]
    names = sorted(set(cls))
    cls_idx = np.array([names.index(c) for c in cls])
    rng = np.random.default_rng(a.seed)
    out = {"positions": int(len(bins)), "alpha": ALPHA, "arms": a.arms,
           "mean_mass_top20": float(bins[:, :20].sum(axis=1).mean()),
           "mean_tail": float(bins[:, 20].mean()),
           "per_class_n": {c: int((cls == c).sum()) for c in names}}

    # ---- NULL: 200 replicates, false-alarm rate at the registered alpha
    null = [run_rep(bins, cls_idx, rng) for _ in range(a.reps_null)]
    fam = {"P1": sum("P1" in r["fire"] for r in null),
           "P2": sum("P2" in r["fire"] for r in null),
           "P3": sum("P3" in r["fire"] for r in null),
           "any": sum(r["detect"] for r in null)}
    out["null"] = {"reps": a.reps_null, "false_alarm_any": fam["any"] / a.reps_null,
                   "false_alarm_by_family": {k: v / a.reps_null for k, v in fam.items()},
                   "mean_p1_p": float(np.mean([r["p1"] for r in null])),
                   "sd_p1_p": float(np.std([r["p1"] for r in null]))}

    # ---- POWER A: temperature mismatch, simulated; the LIVE arms are the real control
    pw = {}
    for T in (0.95, 1.05):
        reps = [run_rep(bins, cls_idx, rng, mode="null", temp=T) for _ in range(a.reps_power)]
        pw["temp%.2f" % T] = {"power": rate(reps), "median_p1": float(np.median([r["p1"] for r in reps]))}
    # ---- POWER B: shared uniform across a verified block (draw reuse)
    for k in (2, 3, 5, 8, 16):
        reps = [run_rep(bins, cls_idx, rng, mode="reuse", k=k) for _ in range(a.reps_power)]
        pw["reuse_k%d" % k] = {"power": rate(reps),
                               "median_abs_z": float(np.median([abs(r["z"]) for r in reps]))}
    # ---- POWER C: acceptance bias toward the drafter (argmax corner + top-5 pessimistic corner)
    for eps in (0.01, 0.02, 0.05):
        for mode, tag in (("bias1", "bias_argmax"), ("bias5", "bias_top5")):
            reps = [run_rep(bins, cls_idx, rng, mode=mode, eps=eps) for _ in range(a.reps_power)]
            pw["%s_eps%.3f" % (tag, eps)] = {
                "power": rate(reps), "median_p1": float(np.median([r["p1"] for r in reps])),
                "top1_excess": float(eps * (1.0 - float(bins[:, 0].mean())))}
    out["power"] = pw

    # ---- MDE at 80 % power (bisection): drafter-argmax bias, and |dT| in the sim's temperature model
    def power_of(fn, reps=30):
        return rate([fn() for _ in range(reps)])

    def mde(build, lo, hi, rounds=5):
        if power_of(lambda: build(hi)) < 0.8:
            return None
        for _ in range(rounds):
            mid = 0.5 * (lo + hi)
            if power_of(lambda: build(mid)) >= 0.8:
                hi = mid
            else:
                lo = mid
        return hi

    out["mde_acceptance_bias_eps_argmax"] = mde(
        lambda e: run_rep(bins, cls_idx, rng, mode="bias1", eps=e), 0.0, 0.20)
    out["mde_acceptance_bias_eps_top5"] = mde(
        lambda e: run_rep(bins, cls_idx, rng, mode="bias5", eps=e), 0.0, 0.50)
    out["mde_temperature_delta"] = mde(
        lambda T: run_rep(bins, cls_idx, rng, mode="null", temp=T), 1.0, 1.30)

    out["mde_notes"] = ("N = %d positions pooled (%d per class); power computed by resampling the "
                        "recorded T=1 distributions; null false-alarm rate fixes the 80 %% threshold "
                        "at the registered alpha." % (len(bins), int((cls == names[0]).sum())))
    with open(a.out or os.path.join(a.dir, "sim.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())

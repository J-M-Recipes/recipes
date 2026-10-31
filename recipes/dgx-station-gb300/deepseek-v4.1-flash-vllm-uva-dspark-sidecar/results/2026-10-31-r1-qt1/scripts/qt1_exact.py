#!/usr/bin/env python3
"""qt1_exact.py — deterministic (no randomisation) calibration tests for QT1-A. (Milo, 2026-10-31, amendment A3)

For each emitted position, with the reported top-20 (sampled token appended if outside) + tail mass:
  R  rank calibration: categories = rank 1, 2, 3, 4-5, 6-10, 11-20, tail. Expected count per category = sum of the
     reported probabilities of the tokens in it; observed = fractional tie-fair count (k tokens tied at one value
     share credit 1/k across the categories they span). Pearson chi-square, 6 dof.
  M  mid-PIT mean: m = F(x-) + p(x)/2 with ties split; under H0 E[m] = 1/2 exactly, Var(m) known per position.
Both are exact under H0 (emitted token ~ reported distribution) and need no random draw.
"""
import json, math, sys, collections

CATS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20)]
def cat_of(rank):
    for i, (a, b) in enumerate(CATS):
        if a <= rank <= b:
            return i
    return len(CATS)

def chi2_sf(x, k):
    # regularized upper gamma Q(k/2, x/2) via series/continued fraction
    a, x = k / 2.0, x / 2.0
    if x <= 0: return 1.0
    if x < a + 1:
        s = term = 1.0 / a; n = a
        while abs(term) > 1e-15 * abs(s):
            n += 1; term *= x / n; s += term
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2; d = an * d + b; d = 1 / d if d else 1e300; c = b + an / c; h *= d * c
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h)

def analyse(paths):
    obs = [0.0] * (len(CATS) + 1); exp = [0.0] * (len(CATS) + 1)
    msum = mvar = 0.0; n = 0
    by_cls = collections.defaultdict(lambda: [[0.0] * 7, [0.0] * 7])
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            for j, tid in enumerate(r.get("token_ids") or []):
                tl = r["top_logprobs"][j] if j < len(r["top_logprobs"]) else None
                if not tl: continue
                items = [(int(k), float(v["logprob"] if isinstance(v, dict) else v)) for k, v in tl.items()]
                items.sort(key=lambda x: -x[1])
                if len(items) > 20:                       # sampled token appended outside the top-20
                    top = [x for x in items if x[0] != tid][:20]
                else:
                    top = items[:20]
                ps = [math.exp(lp) for _, lp in top]; tail = max(0.0, 1.0 - sum(ps))
                # group ties into value-classes in rank order
                groups = []; i = 0
                while i < len(top):
                    k = i
                    while k + 1 < len(top) and top[k + 1][1] == top[i][1]: k += 1
                    groups.append((i + 1, k + 1, [t for t, _ in top[i:k + 1]], ps[i])); i = k + 1
                # expected per category
                F = 0.0; m_emit = None; Em2 = 0.0
                cls = r.get("cls", "?")
                for (r0, r1, toks, p) in groups:
                    for rk in range(r0, r1 + 1):
                        exp[cat_of(rk)] += p; by_cls[cls][1][cat_of(rk)] += p
                    gm = F + len(toks) * p / 2.0         # mid-PIT of the whole tie group (ties split)
                    Em2 += len(toks) * p * gm * gm
                    if tid in toks:
                        m_emit = gm
                        for rk in range(r0, r1 + 1):     # tie-fair fractional credit
                            obs[cat_of(rk)] += 1.0 / len(toks); by_cls[cls][0][cat_of(rk)] += 1.0 / len(toks)
                    F += len(toks) * p
                exp[-1] += tail; by_cls[cls][1][-1] += tail
                gm_tail = F + tail / 2.0; Em2 += tail * gm_tail * gm_tail
                if m_emit is None:
                    obs[-1] += 1.0; by_cls[cls][0][-1] += 1.0; m_emit = gm_tail
                msum += m_emit - 0.5; mvar += max(Em2 - 0.25, 0.0); n += 1
    def chi(o, e):
        keep = [(a, b) for a, b in zip(o, e) if b > 5]
        x = sum((a - b) ** 2 / b for a, b in keep); dof = len(keep) - 1
        return x, dof, chi2_sf(x, dof)
    x, dof, p = chi(obs, exp)
    out = {"positions": n,
           "R_rank_calibration": {"chi2": x, "dof": dof, "p": p,
                                  "obs": [round(v, 1) for v in obs], "exp": [round(v, 1) for v in exp],
                                  "cats": ["1", "2", "3", "4-5", "6-10", "11-20", "tail"]},
           "M_midpit_mean": {"mean_minus_half": msum / n, "z": msum / math.sqrt(mvar)},
           "R_per_class_p": {c: chi(o, e)[2] for c, (o, e) in sorted(by_cls.items())}}
    return out

if __name__ == "__main__":
    print(json.dumps(analyse(sys.argv[1:]), indent=1))

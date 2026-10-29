#!/usr/bin/env python3
"""tr2_verdict.py — Card TR2 verdict from receipts in the card run dir.

Usage: python3 tr2_verdict.py <results-cardTR2>
Reads: <R>/<arm>/facts.json, <R>/<arm>/perclass-t1|t0/*.json, <R>/<arm>/l2-verify-stats.txt,
       <R>/tf-<ref>-vs-<cand>.log, <R>/gauntlet-<arm>/... and the L2 stats line in the boot log
       (dumped to <R>/<arm>/l2-boot-stats.txt by the runner or by hand).
Writes: <R>/verdict.json + a human table on stdout. Bars are the ones frozen in PREREG-TR2.md.
"""
import json
import os
import re
import sys

CLASSES = ["prose", "structured", "code", "shell_ops", "tool_json"]


def load(p, default=None):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return default


def perclass(R, arm, tag, mode):
    """{'class': {'c1': tok/s, 'c8': agg tok/s, 'accept': x, 'accepted': n, 'drafted': n}}"""
    base = os.path.join(R, arm, tag)
    out = {}
    for fn in (sorted(os.listdir(base)) if os.path.isdir(base) else []):
        if not fn.endswith(".json"):
            continue
        d = load(os.path.join(base, fn))
        if not isinstance(d, dict) or "res" not in d:
            continue
        for cls, r in (d.get("res") or {}).items():
            e = out.setdefault(cls, {})
            if d.get("mode") == "c1" and r.get("decode_tok_s_pooled"):
                e.setdefault("c1_reps", []).append(r["decode_tok_s_pooled"])
                e["accept"] = r.get("accept")
                e["accepted"] = r.get("accepted")
                e["drafted"] = r.get("drafted")
                e["draft_steps"] = r.get("draft_steps")
            if d.get("mode") == "c8" and r.get("agg_decode_tok_s"):
                e.setdefault("c8_reps", []).append(r["agg_decode_tok_s"])
    for e in out.values():
        for k in ("c1", "c8"):
            reps = e.pop(k + "_reps", [])
            if reps:
                e[k] = round(sum(reps) / len(reps), 2)
                e[k + "_n"] = len(reps)
    return out


def l2_stats_line(R, arm):
    p = os.path.join(R, arm, "l2-boot-stats.txt")
    if os.path.exists(p):
        txt = open(p).read()
        m = re.search(r"total=\{([^}]*)\}", txt)
        if m:
            d = {}
            for part in m.group(1).split(","):
                k, _, v = part.partition(":")
                try:
                    d[k.strip().strip("'")] = int(v)
                except ValueError:
                    pass
            return d
    return {}


def tf_flips(R, ref, cand):
    p = os.path.join(R, "tf-%s-vs-%s.log" % (ref, cand))
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    out = {}
    m = re.search(r"top-1 flips=(\d+)\s*\(([0-9.]+)%\)", txt)
    if m:
        out["flips"] = int(m.group(1))
        out["flip_pct"] = float(m.group(2))
    m = re.search(r"mean\|Δ\|=([0-9.]+).*?p99=([0-9.]+)", txt)
    if m:
        out["mean_abs_dlogp"] = float(m.group(1))
        out["p99_dlogp"] = float(m.group(2))
    return out or {"raw_tail": txt.strip().splitlines()[-1][:160]}


def gauntlet(R, arm):
    for name in ("summary.json", "gauntlet.json", "result.json"):
        d = load(os.path.join(R, "gauntlet-%s" % arm, name))
        if isinstance(d, dict):
            for k in ("passed", "pass", "n_pass", "passed_count"):
                if k in d:
                    return d.get(k), d.get("total") or d.get("n") or d.get("n_tasks")
            if "results" in d and isinstance(d["results"], list):
                ok = sum(1 for r in d["results"] if r.get("pass") or r.get("ok") or r.get("passed"))
                return ok, len(d["results"])
    return None, None


def ratio(b, a):
    try:
        return round(b / a, 4) if a else None
    except TypeError:
        return None


def main():
    R = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/dsv41/m3/tr2/results-cardTR2")
    arms = ["A", "B", "A2"]
    facts = {a: load(os.path.join(R, a, "facts.json"), {}) for a in arms}
    t1 = {a: perclass(R, a, "perclass-t1", "c1") for a in arms}
    t0 = {a: perclass(R, a, "perclass-t0", "c1") for a in arms}
    l2 = {a: l2_stats_line(R, a) for a in arms}
    gates = {}

    # G0 boot
    kvs = {a: (facts[a] or {}).get("kv_tokens") for a in arms}
    have_kv = all(kvs.values())
    gates["G0_kv_identical"] = ((len(set(kvs.values())) == 1) if have_kv else None, kvs)
    hh = {a: facts[a].get("hash_hit") for a in arms if facts.get(a)}
    gates["G0_hash_hit"] = ((all(v == 1 for v in hh.values()) if hh else None), hh)
    li = {a: (facts[a] or {}).get("l2_installed") for a in arms if facts.get(a)}
    gates["G0_l2_installed"] = ((all((v or 0) >= 1 for v in li.values()) if li else None), li)
    # G0b lever engaged
    b_hit = (l2.get("B") or {}).get("hit", 0)
    a_hit = (l2.get("A") or {}).get("hit", 0)
    gates["G0b_lever"] = (((b_hit > 0 and a_hit == 0) if l2.get("B") and l2.get("A") else None),
                          {"B_hit": b_hit, "A_hit": a_hit,
                                                     "B_miss": (l2.get("B") or {}).get("miss"),
                                                     "A_noskip": (l2.get("A") or {}).get("hit_noskip")})

    # G1 speed (temp 1.0, C1 primary + C8)
    rows = {}
    for cls in CLASSES:
        a1, b1, c1 = (t1[a].get(cls, {}).get("c1") for a in arms)
        a8, b8, c8 = (t1[a].get(cls, {}).get("c8") for a in arms)
        m1 = (a1 + c1) / 2 if a1 and c1 else None
        m8 = (a8 + c8) / 2 if a8 and c8 else None
        rows[cls] = {"c1": [a1, b1, c1], "c1_ratio": ratio(b1, m1) if b1 and m1 else None,
                     "c8": [a8, b8, c8], "c8_ratio": ratio(b8, m8) if b8 and m8 else None}
    r1 = [v["c1_ratio"] for v in rows.values() if v["c1_ratio"]]
    r8 = [v["c8_ratio"] for v in rows.values() if v["c8_ratio"]]
    headline = round(sum(r1) / len(r1), 4) if r1 else None
    gate_ok = (min(r1) >= 0.98) if r1 else None
    gates["G1_speed"] = (gate_ok, {"headline_c1_ratio": headline,
                                   "per_class_c1": {k: v["c1_ratio"] for k, v in rows.items()},
                                   "per_class_c8": {k: v["c8_ratio"] for k, v in rows.items()},
                                   "mean_c8_ratio": round(sum(r8) / len(r8), 4) if r8 else None})
    verdict_speed = ("WORKS" if headline and headline >= 1.005 else
                     "NULL" if headline and headline >= 0.995 else
                     "FAIL" if headline else "UNMEASURED")

    # G2 in-process identity: prefer the dedicated verify boot (BV/verify-stats.txt), else the B window
    vb = vo = None
    bvp = os.path.join(R, "BV", "verify-stats.txt")
    if os.path.exists(bvp):
        txt = open(bvp).read()
        for m in re.finditer(r"total=\{([^}]*)\}", txt):
            d = {}
            for part in m.group(1).split(","):
                k, _, v = part.partition(":")
                try:
                    d[k.strip().strip("'")] = int(v)
                except ValueError:
                    pass
            if d:
                vb, vo = d.get("verify_bad"), d.get("verify_ok")
    if vo is None:
        vb = (l2.get("B") or {}).get("verify_bad")
        vo = (l2.get("B") or {}).get("verify_ok")
    if vo is not None and vb is not None:
        gates["G2_identity_inproc"] = (vb == 0 and vo >= 100, {"verify_ok": vo, "verify_bad": vb})
    else:
        gates["G2_identity_inproc"] = (None, {"verify_ok": vo, "verify_bad": vb})

    # G3 teacher-forced
    flips = {p: tf_flips(R, *p.split("-vs-")) for p in ("A-vs-B", "A-vs-A2", "A2-vs-B")}
    fb = (flips.get("A-vs-B") or {}).get("flip_pct")
    fa = (flips.get("A-vs-A2") or {}).get("flip_pct")
    gates["G3_tf"] = ((fb <= fa + 0.05) if (fb is not None and fa is not None) else None, flips)

    # G4 temp-0 parity
    rows0 = {}
    for cls in CLASSES:
        a0, b0, c0 = (t0[a].get(cls, {}).get("c1") for a in arms)
        m0 = (a0 + c0) / 2 if a0 and c0 else None
        acc = [t0[a].get(cls, {}).get("accepted") for a in arms]
        drf = [t0[a].get(cls, {}).get("drafted") for a in arms]
        rows0[cls] = {"c1": [a0, b0, c0], "ratio": ratio(b0, m0) if b0 and m0 else None,
                      "accepted": acc, "drafted": drf}
    rr = [v["ratio"] for v in rows0.values() if v["ratio"]]
    gates["G4_parity_t0"] = ((min(rr) >= 0.97) if rr else None,
                             {"per_class": {k: v["ratio"] for k, v in rows0.items()},
                              "accepted": {k: v["accepted"] for k, v in rows0.items()}})

    # G5 gauntlet
    ga = {a: gauntlet(R, a) for a in arms}
    gates["G5_gauntlet"] = (((all(v[0] == 12 for v in ga.values()))
                             if all(v[0] is not None for v in ga.values()) and ga else None), ga)
    # G6 hygiene
    er = {a: (facts[a] or {}).get("errors") for a in arms if facts.get(a)}
    gates["G6_errors"] = ((all(v == 0 for v in er.values()) if er else None), er)

    hard = ["G0_kv_identical", "G0_hash_hit", "G0_l2_installed", "G0b_lever", "G2_identity_inproc",
            "G5_gauntlet", "G6_errors"]
    miss = [g for g in hard if gates[g][0] is False]
    if verdict_speed == "FAIL" or miss:
        verdict = "FAIL"
    elif verdict_speed == "NULL":
        verdict = "NULL"
    elif verdict_speed == "WORKS" and not miss:
        verdict = "PASS"
    else:
        verdict = "INCOMPLETE"
    out = {"verdict": verdict, "speed": verdict_speed, "headline_c1_ratio": headline,
           "gates": {k: v[0] for k, v in gates.items()}, "detail": {k: v[1] for k, v in gates.items()},
           "per_class_t1": rows, "per_class_t0": rows0, "facts": facts, "l2": l2}
    with open(os.path.join(R, "verdict.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("VERDICT:", verdict, "| speed:", verdict_speed, "| headline C1 ratio:", headline)
    for k, (ok, det) in gates.items():
        print("  %-18s %s" % (k, {True: "PASS", False: "FAIL", None: "UNMEASURED"}[ok]))
    print("\nper-class temp-1.0 C1 (A / B / A2, ratio B/mean):")
    for cls, v in rows.items():
        print("  %-11s %-28s %s" % (cls, v["c1"], v["c1_ratio"]))
    print("\nper-class temp-0 C1 (A / B / A2, ratio):")
    for cls, v in rows0.items():
        print("  %-11s %-28s %s  accepted=%s" % (cls, v["c1"], v["ratio"], v["accepted"]))


if __name__ == "__main__":
    main()
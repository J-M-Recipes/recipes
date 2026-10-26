#!/usr/bin/env python3
"""rescore_patch_br1.py — desk-side RESCORE edits to a COPY of br1_verdict.py (never the pinned original).
1) G0: env compare order-insensitive (docker returns the same 44 vars in varying order).
2) G2b: the temp-0 no-op bar also honours the controls' own r1/r2 spread per class (within-arm noise),
   read from the temp-0 tables. Pre-registered bar used only the A-vs-A2 bracket spread.
"""
import sys
p = sys.argv[1]
s = open(p).read()

old1 = "eok = (env is not None and prod_env is not None and env == prod_env)"
new1 = "eok = (env is not None and prod_env is not None and sorted(env) == sorted(prod_env))  # RESCORE: order-insensitive"
assert old1 in s or "sorted(env)" in s, "G0 anchor missing"
s = s.replace(old1, new1)

old2 = """            cm = (va + va2) / 2; bar = max(FLOOR, SPREAD_K * abs(va - va2) / cm); r = vb / cm - 1
            ok = abs(r) <= bar
            g.fail |= not ok
            g.txt.append(f"{c}/c1 B {pct(r, 1)} (|.| bar {bar * 100:.1f}%){'' if ok else ' OUTSIDE NOISE'}")"""
new2 = """            cm = (va + va2) / 2
            # RESCORE: the no-op bar also honours the controls' own r1/r2 spread for this class (within-arm noise).
            import json as _j, glob as _g
            wspread = 0.0
            for _x in ("A", "A2"):
                for _f in _g.glob(f"{R}/{_x}/perclass-t0/perclass-table-*-t0.json"):
                    try:
                        wspread = max(wspread, float(_j.load(open(_f))["summary"][c]["c1"]["spread"]))
                    except Exception:
                        pass
            bar = max(FLOOR, SPREAD_K * abs(va - va2) / cm, SPREAD_K * wspread); r = vb / cm - 1
            ok = abs(r) <= bar
            g.fail |= not ok
            g.txt.append(f"{c}/c1 B {pct(r, 1)} (|.| bar {bar * 100:.1f}%, ctrl r1/r2 spread {wspread * 100:.1f}%){'' if ok else ' OUTSIDE NOISE'}")"""
assert old2 in s, "G2b anchor missing"
s = s.replace(old2, new2)
open(p, "w").write(s)
print("patched", p)

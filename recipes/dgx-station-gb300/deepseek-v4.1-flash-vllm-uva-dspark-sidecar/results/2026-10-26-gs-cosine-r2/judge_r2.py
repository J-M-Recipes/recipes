#!/usr/bin/env python3
"""judge_r2.py — cosine-gate re-registration R2 for the DSV4.1 RTX-sidecar lane. REGISTERED 2026-10-24 before any re-judging.

Why: the R1 gate (min cos >= 0.999 over >= 32-row checks) is noise-bound where the cold experts carry ~0 of the layer output
(cosine of two near-zero vectors). Card GV2 showed this at layer 39 on a split identical to v25 static.

R2 (all on post-"Application startup complete" `MEGA_PEER check` lines, rows >= 32, as R1):
  e = rel_diff * cold_share = ||cold_peer - cold_trt|| / ||y_trt||   (mega_peer_hook._log_check: the sidecar's error as a
      fraction of the whole layer output; y_peer - y_trt = cold_peer - cold_trt because y_hot is shared)
  coverage: every one of the 40 layers has >= 1 check with >= 32 rows (unchanged)
  timeouts: 0 peer + 0 hook timeout lines (unchanged)
  bar B = 1.5 x max over the REFERENCE boots (v25 static map, rowmap-static-v1) of that boot's max e
  candidate PASS iff coverage AND timeouts AND max e <= B
Reported, not gated: R1 verdict, min cos over checks with cold share >= 0.01, p99 e.
Precision note: the log prints rel_diff to 4 and cold share to 3 decimals; e is computed from the printed values.
The rule, the reference set and the 1.5x factor are fixed here; they are not to be changed after reading the result.

usage: judge_r2.py OUT_DIR ref:LABEL=LOG [ref:...] cand:LABEL=LOG [cand:...]   (peer logs optional: LOG,PEERLOG)
"""
import json
import re
import sys
from pathlib import Path

RX = re.compile(r"check layer=(\d+) T=(\d+) rows=(\d+) cold rel_diff=([\d.]+) cos=([\d.]+) \(cold share of output norm ([\d.]+)\)")


def load(log, peer=None):
    if not Path(log).exists():
        return None
    t = Path(log).read_text(errors="replace")
    i = t.find("Application startup complete")
    if i < 0:
        return None
    rows = [dict(L=int(a), T=int(b), rows=int(c), rel=float(d), cos=float(e), share=float(f))
            for a, b, c, d, e, f in RX.findall(t[i:])]
    g = [r for r in rows if r["rows"] >= 32]
    for r in g:
        r["e"] = r["rel"] * r["share"]
    hook_to = sum(1 for l in t[i:].splitlines() if ("MEGA" in l or "peer" in l) and "timeout" in l.lower())
    peer_to = 0
    if peer and Path(peer).exists():
        peer_to = sum(1 for l in Path(peer).read_text(errors="replace").splitlines() if "timeout" in l.lower())
    cov = {r["L"] for r in g}
    es = sorted(r["e"] for r in g)
    worst = max(g, key=lambda r: r["e"]) if g else None
    big = [r["cos"] for r in g if r["share"] >= 0.01]
    return dict(n=len(g), layers_covered=len(cov), max_e=es[-1] if es else None, p99_e=es[int(0.99 * (len(es) - 1))] if es else None,
                worst=worst, r1_min_cos=min((r["cos"] for r in g), default=None),
                r1_pass=bool(g) and len(cov) == 40 and min(r["cos"] for r in g) >= 0.999 and hook_to == 0 and peer_to == 0,
                min_cos_share_ge_1pct=min(big) if big else None, hook_timeouts=hook_to, peer_timeouts=peer_to)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    arms = []
    for a in sys.argv[2:]:
        kind, rest = a.split(":", 1); label, paths = rest.split("=", 1)
        log, *peer = paths.split(",")
        arms.append((kind, label, log, peer[0] if peer else None))
    res = {label: dict(kind=kind, log=log, **(load(log, peer) or {"error": "no post-startup checks"})) for kind, label, log, peer in arms}
    refs = [v for v in res.values() if v["kind"] == "ref" and v.get("max_e") is not None]
    B = 1.5 * max(v["max_e"] for v in refs) if refs else None
    for v in res.values():
        if v.get("max_e") is None or B is None:
            v["r2"] = "INVALID"
        else:
            v["r2"] = "PASS" if (v["layers_covered"] == 40 and v["hook_timeouts"] == 0 and v["peer_timeouts"] == 0 and v["max_e"] <= B) else "FAIL"
    json.dump({"bar_B": B, "n_refs": len(refs), "arms": res}, open(out / "r2.json", "w"), indent=1, default=str)
    L = [f"# Cosine gate R2 re-judge — bar B = 1.5 x max reference e = **{B:.5f}** ({len(refs)} v25-static reference boots)\n" if B else "# R2: no valid reference\n",
         "| boot | kind | gated checks | layers | max e (err / layer output) | p99 e | worst-e check | R2 | R1 (old) min cos | min cos, share >= 1% |",
         "|---|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for k, v in res.items():
        if v.get("max_e") is None:
            L.append(f"| {k} | {v['kind']} | — | — | — | — | {v.get('error')} | INVALID | — | — |"); continue
        w = v["worst"]
        L.append(f"| {k} | {v['kind']} | {v['n']} | {v['layers_covered']} | {v['max_e']:.5f} | {v['p99_e']:.5f} | L{w['L']} T={w['T']} rows={w['rows']} cos {w['cos']:.5f} share {w['share']:.3f} | **{v['r2']}** | {v['r1_min_cos']:.5f} ({'PASS' if v['r1_pass'] else 'FAIL'}) | {v['min_cos_share_ge_1pct']} |")
    (out / "R2-VERDICT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

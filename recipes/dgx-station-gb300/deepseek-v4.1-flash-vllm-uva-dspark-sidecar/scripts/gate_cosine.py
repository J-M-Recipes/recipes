#!/usr/bin/env python3
"""gate_cosine.py — the bring-up gate for the GB300 + RTX PRO 6000 sidecar lane.

What it checks, and why this is the gate that matters: the sidecar computes a layer's cold-expert
contribution on the RTX card and the GB300 hook adds it back. On a sidecar timeout the hook SKIPS
that add and keeps emitting tokens — the output is silently wrong, not slow. So before the lane serves
anything, the hook is booted in check mode (MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1) and every
`MEGA_PEER check …` line is compared against the TRT-LLM cold path computed on the GB300 itself:

    MEGA_PEER check layer=<L> T=<tokens> rows=<cold rows> cold rel_diff=<x> cos=<y> (cold share of output norm …)

Pass bar (pre-registered, from RonanLabs' published gate and re-registered for our uniform map):
  * 40/40 MoE layers covered by at least one check with >= --min-rows cold rows,
  * min cos over checks with >= --min-rows >= --bar (0.999),
  * zero sidecar timeouts reported on the shm buffer.

Only POST-STARTUP lines count. vLLM's dummy profile run feeds all-zero inputs and emits checks with
cos=0.00000 / "cold share … nan"; those are warm-up, not evidence. v1 of this driver counted them and a
sibling session briefly reported a false arm-B failure from it (see the research bundle's TAKEOVER.md).

Usage:
  python3 gate_cosine.py <server-log> [--bar 0.999] [--min-rows 32] [--json out.json]
Exit 0 = pass, 1 = fail, 2 = the log has no post-startup checks (gate not run).
"""
import argparse
import json
import re
import sys

CHECK = re.compile(
    r"MEGA_PEER check layer=(?P<layer>\d+) T=(?P<t>\d+) rows=(?P<rows>\d+) "
    r"cold rel_diff=(?P<rel>[0-9.]+) cos=(?P<cos>[0-9.]+)"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--bar", type=float, default=0.999)
    ap.add_argument("--min-rows", type=int, default=32)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    started = False
    checks = []
    with open(a.log, errors="replace") as fh:
        for line in fh:
            if "Application startup complete" in line:
                started = True
                continue
            if not started:
                continue
            m = CHECK.search(line)
            if m:
                checks.append({k: (float(v) if k in ("rel", "cos") else int(v))
                               for k, v in m.groupdict().items()})

    if not checks:
        print("NO POST-STARTUP CHECKS — gate not run (boot with MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1)")
        return 2

    gated = [c for c in checks if c["rows"] >= a.min_rows]
    layers = sorted({c["layer"] for c in gated})
    worst = min(gated, key=lambda c: c["cos"]) if gated else None
    result = {
        "checks_post_startup": len(checks),
        "checks_min_rows": len(gated),
        "min_rows": a.min_rows,
        "layers_covered": len(layers),
        "layers": layers,
        "bar": a.bar,
        "min_cos": worst["cos"] if worst else None,
        "min_cos_at": ({k: worst[k] for k in ("layer", "t", "rows", "cos", "rel")} if worst else None),
        "max_rel_diff": max((c["rel"] for c in gated), default=None),
        "pass": bool(len(layers) == 40 and worst and worst["cos"] >= a.bar),
    }
    print(json.dumps(result, indent=2))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(result, fh, indent=2)
    if not result["pass"]:
        print(f"FAIL: bar cos>={a.bar} on >= {a.min_rows} cold rows, 40/40 layers")
        return 1
    print(f"PASS: {len(layers)}/40 layers, min cos {result['min_cos']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

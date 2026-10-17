#!/usr/bin/env python3
"""gauntlet_reps.py <lane> <provider> <model> <n>  — n Hermes 20-hop tool-chain runs; prints one JSON line per rep and a MEDIAN line."""
import json, sys, statistics, pathlib
lane, prov, model, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
here = pathlib.Path(__file__).resolve().parent
sys.argv = ["run_lane.py", lane, prov, model]
exec(open(here / "run_lane.py").read().split("rows=[]; LAST={}")[0])
LAST = {}; walls = []; oks = 0
for i in range(n):
    nonce, prompt = gauntlet(); w, final, tools, rc, so, se = hermes(prompt, 1200)
    ok = nonce in (final or ""); oks += ok; walls.append(w)
    print(json.dumps({"id": f"gauntlet20-rep{i+1}", "wall_s": round(w, 1), "agent_s": round((LAST["agent_ms"] or 0) / 1000, 1), "out_tok": LAST["tokens"].get("output"), "ok": ok, "tools": tools}), flush=True)
print(json.dumps({"MEDIAN": round(statistics.median(walls), 1), "ok": f"{oks}/{n}"}), flush=True)

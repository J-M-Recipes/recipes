#!/usr/bin/env python3
"""Four-tier plan v1: the eight Spark targets from plan-uniform-62 unchanged (their banks are staged and
verified), plus an RTX PRO 6000 target `rtx0` with RTX_PER experts per layer drawn (seeded) from the experts
no Spark holds. Whatever is left per layer is the Station's (HBM + Grace), listed under `station` for the
engine hook; the builder never builds it.

Usage: make_plan_4tier.py [RTX_PER=56] [seed=20261031]
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTX_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 56
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261031
src = json.load(open(os.path.join(HERE, "plans/plan-uniform-62.json")))
out = dict(src)
out["scheme"] = "four-tier-v1: sparks uniform-62 (seed 20261030) + rtx uniform from remainder"
out["seed_rtx"] = SEED
out["targets"] = {k: v for k, v in src["targets"].items() if k.startswith("spark")}
rng = random.Random(SEED)
rtx, station = {}, {}
for L in src["moe_layers"]:
    held = set()
    for t in range(8):
        held.update(src["targets"][f"spark{t}"]["layers"][str(L)])
    rest = sorted(set(range(896)) - held)
    pick = sorted(rng.sample(rest, RTX_PER))
    rtx[str(L)] = pick
    station[str(L)] = sorted(set(rest) - set(pick))
out["targets"]["rtx0"] = {"layers": rtx}
out["station"] = {"layers": station}
# sanity: every (layer, expert) owned exactly once across all tiers
for L in src["moe_layers"]:
    owners = []
    for t, v in out["targets"].items():
        owners += v["layers"][str(L)]
    owners += station[str(L)]
    assert sorted(owners) == list(range(896)), L
n_sp = sum(len(v) for t, v in out["targets"].items() if t.startswith("spark") for v in v["layers"].values())
n_rtx = sum(len(v) for v in rtx.values()); n_st = sum(len(v) for v in station.values())
REC = 18_579_480
print(f"sparks {n_sp:,}  rtx {n_rtx:,} ({n_rtx*REC/2**30:.2f} GiB)  station {n_st:,} ({n_st*REC/2**30:.2f} GiB)  "
      f"total {n_sp+n_rtx+n_st:,}")
p = os.path.join(HERE, "plans/plan-4tier-v1.json")
json.dump(out, open(p, "w"))
print("wrote", p)

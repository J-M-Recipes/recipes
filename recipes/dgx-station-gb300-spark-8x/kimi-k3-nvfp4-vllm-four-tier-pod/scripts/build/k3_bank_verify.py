#!/usr/bin/env python3
"""Prove a built expert bank matches the checkpoint, field by field.

The builder's own --verify compares only the 24 scalar bytes. This checks all five fields,
including un-swizzling the E4M3 scale blocks back to the checkpoint's linear order — which is
also the reference for the inverse operation every consumer (CPU reference, GPU kernel) needs.

Nothing in the checkpoint's NVFP4 payload is transformed except a byte permutation of the scales,
so every comparison here is exact equality: any mismatch is a real layout bug, not tolerance.

Usage (on the Station, which has both the checkpoint and the bank):
    python3 k3_bank_verify.py --bank /models/kimi-k3/banks/spark0/k3bank-spark0.ldb \
        --plan /tmp/plan-uniform-62.json --target spark0 --samples 24
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDER = os.path.join(HERE, "k3_bank_build.py")


def load_builder():
    spec = importlib.util.spec_from_file_location("k3_bank_build", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["k3_bank_build"] = mod          # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


def unswizzle(data: bytes, m: int, k: int) -> bytes:
    """Inverse of the builder's scale swizzle.

    The forward map is x.reshape(m//128, 4, 32, k//4, 4).transpose(0,3,2,1,4). The inverse must
    reshape to the SWIZZLED dimension order (m//128, k//4, 32, 4, 4) before transposing — a plain
    second call to the forward function is NOT the inverse (verified false on both shapes).
    """
    import numpy as np
    a = np.frombuffer(data, dtype=np.uint8)
    if a.size != m * k:
        raise ValueError(f"unswizzle: {a.size} B, want {m * k}")
    return a.reshape(m // 128, k // 4, 32, 4, 4).transpose(0, 3, 2, 1, 4).copy().tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    kb = load_builder()

    layers, _fp = kb.discover(kb.CKPT_DEFAULT, verbose=False)
    per = kb.load_plan(args.plan, layers, args.target, False)

    with open(args.bank, "rb") as f:
        head = f.read(kb.HEADER_BYTES)
    assert head[:8] == kb.MAGIC, f"bad magic {head[:8]!r}"
    ver, rec, E, n_layers, plan_cut, lay, ntargets = struct.unpack("<IIIIQ8sI", head[8:8 + 36])
    bank_sha = head[44:76].hex()
    counts = struct.unpack(f"<{ntargets}I", head[76:76 + 4 * ntargets])
    print(f"bank {args.bank}")
    print(f"  version {ver} rec {rec:,} E {E} layers {n_layers} layout {lay.rstrip(chr(0).encode()).decode()}")
    print(f"  plan {bank_sha[:16]} per-layer counts {sorted(set(counts))}")

    lids = sorted(per)
    rng = random.Random(args.seed)
    picks = []
    for _ in range(args.samples):
        lid = rng.choice(lids)
        picks.append((lid, rng.choice(per[lid])))
    picks = sorted(set(picks))

    fd = os.open(args.bank, os.O_RDONLY)
    bad = 0
    checked = {"w13": 0, "w2": 0, "sf13": 0, "sf2": 0, "scalars": 0}
    try:
        for lid, gid in picks:
            L = lids.index(lid)
            slot = sorted(per[lid]).index(gid)
            off = kb.HEADER_BYTES + (L * E + slot) * rec
            buf = os.pread(fd, rec, off)
            assert len(buf) == rec, f"short record read at {off}"
            geom = layers[lid]

            def ck(key, n):
                g = os.open(geom.shard, os.O_RDONLY)
                try:
                    return os.pread(g, n, geom.offs[key][gid])
                finally:
                    os.close(g)

            errs = []
            w1 = ck(("w1", "weight"), kb.M_W)
            w2 = ck(("w2", "weight"), kb.M_W2)
            w3 = ck(("w3", "weight"), kb.M_W)
            if buf[0:kb.M_W] != w3:
                errs.append("w13[up] != w3")
            if buf[kb.M_W:2 * kb.M_W] != w1:
                errs.append("w13[gate] != w1")
            if buf[kb.FIELD_OFF["w2"]:kb.FIELD_OFF["w2"] + kb.M_W2] != w2:
                errs.append("w2 field")
            s1 = ck(("w1", "weight_scale"), kb.M_S)
            s2 = ck(("w2", "weight_scale"), kb.M_S2)
            s3 = ck(("w3", "weight_scale"), kb.M_S)
            sf13 = buf[kb.FIELD_OFF["sf13"]:kb.FIELD_OFF["sf13"] + 2 * kb.M_S]
            if unswizzle(sf13[0:kb.M_S], 3072, 224) != s3:
                errs.append("sf13[0] != sw(w3 scale)")
            if unswizzle(sf13[kb.M_S:2 * kb.M_S], 3072, 224) != s1:
                errs.append("sf13[1] != sw(w1 scale)")
            sf2 = buf[kb.FIELD_OFF["sf2"]:kb.FIELD_OFF["sf2"] + kb.M_S2]
            if unswizzle(sf2, 3584, 192) != s2:
                errs.append("sf2 != sw(w2 scale)")
            ref24 = b"".join(ck(tuple(n.split(".")[:2]), 4) for n in kb.SCALARS)
            if buf[kb.FIELD_OFF["scalars"]:kb.FIELD_OFF["scalars"] + 24] != ref24:
                errs.append("scalars")
            if errs:
                bad += 1
                print(f"  FAIL layer {lid} gid {gid} off {off:,}: {', '.join(errs)}")
            else:
                checked["w13"] += 1
                checked["w2"] += 1
                checked["sf13"] += 1
                checked["sf2"] += 1
                checked["scalars"] += 1
                print(f"  ok   layer {lid:>2} gid {gid:>3} off {off:>15,}")
    finally:
        os.close(fd)

    print(f"\n{len(picks) - bad}/{len(picks)} records exact; fields checked {checked}")
    print("VERDICT:", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

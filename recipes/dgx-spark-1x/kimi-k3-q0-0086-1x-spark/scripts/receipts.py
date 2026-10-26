#!/usr/bin/env python3
"""receipts.py — the two numbers in every "X bits per parameter" claim.

  declared bpw = file_bytes * 8 / advertised_parameter_count
  real     bpw = file_bytes * 8 / parameters_actually_in_the_file

Also reports how much of the original weight volume survived, which is the
part nobody puts in the screenshot.

Usage: python3 receipts.py k3-q01.gguf [--nominal 2.8e12] [--orig-experts 896]
"""
import argparse
import struct


def parse(path):
    with open(path, "rb") as f:
        buf = f.read(512 << 20)          # header + tensor table only
    assert buf[:4] == b"GGUF"
    o = 4
    ver = struct.unpack_from("<I", buf, o)[0]; o += 4
    n_t = struct.unpack_from("<Q", buf, o)[0]; o += 8
    n_kv = struct.unpack_from("<Q", buf, o)[0]; o += 8

    def s():
        nonlocal o
        n = struct.unpack_from("<Q", buf, o)[0]; o += 8
        v = buf[o:o + n].decode("utf-8", "replace"); o += n
        return v

    def val(t):
        nonlocal o
        simple = {0: ("<B", 1), 1: ("<b", 1), 7: ("<B", 1), 2: ("<H", 2), 3: ("<h", 2),
                  4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 10: ("<Q", 8),
                  11: ("<q", 8), 12: ("<d", 8)}
        if t in simple:
            fmt, n = simple[t]
            v = struct.unpack_from(fmt, buf, o)[0]; o += n
            return v
        if t == 8:
            return s()
        if t == 9:
            at = struct.unpack_from("<I", buf, o)[0]; o += 4
            n = struct.unpack_from("<Q", buf, o)[0]; o += 8
            return [val(at) for _ in range(n)]
        raise ValueError(t)

    kv = {}
    for _ in range(n_kv):
        k = s(); t = struct.unpack_from("<I", buf, o)[0]; o += 4
        kv[k] = val(t)
    tensors = []
    for _ in range(n_t):
        name = s()
        nd = struct.unpack_from("<I", buf, o)[0]; o += 4
        dims = [struct.unpack_from("<Q", buf, o + 8 * i)[0] for i in range(nd)]
        o += 8 * nd
        tt = struct.unpack_from("<I", buf, o)[0]; o += 4
        off = struct.unpack_from("<Q", buf, o)[0]; o += 8
        tensors.append((name, dims, tt))
    return kv, tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--nominal", type=float, default=2.8e12)
    ap.add_argument("--orig-experts", type=int, default=896)
    a = ap.parse_args()

    import os
    size = os.path.getsize(a.gguf)
    kv, tensors = parse(a.gguf)

    kept = 0
    for name, dims, _ in tensors:
        p = 1
        for d in dims:
            p *= d
        kept += p

    declared = size * 8 / a.nominal
    real = size * 8 / kept if kept else float("nan")
    experts = kv.get("kimi-k3.expert_count", "?")
    layers = kv.get("kimi-k3.block_count", "?")
    used = kv.get("kimi-k3.expert_used_count", "?")

    print(f"file                 : {a.gguf}  {size/1e9:.2f} GB")
    print(f"advertised params    : {a.nominal/1e12:.2f} T   (the model card)")
    print(f"params in this file  : {kept/1e9:.2f} B   ({100*kept/a.nominal:.2f}% of nominal)")
    print(f"layers / experts     : {layers} / {experts} of {a.orig_experts}  (top-{used} per token)")
    print()
    print(f"DECLARED bits/param  : {declared:.4f}   <- the tweet")
    print(f"REAL bits/param      : {real:.3f}   <- divide by what you actually shipped")
    print(f"weight volume missing: {100*(1-kept/a.nominal):.1f}%")


if __name__ == "__main__":
    main()

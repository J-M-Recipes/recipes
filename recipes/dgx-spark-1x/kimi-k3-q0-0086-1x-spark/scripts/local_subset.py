#!/usr/bin/env python3
"""local_subset.py — cut a K3 GGUF down to N layers + fewer experts, offline.

Operates on an already-assembled local GGUF (no network). Used to trade depth
and active experts for decode speed when quality is explicitly out of scope.

Usage:
  python3 local_subset.py IN.gguf OUT.gguf --layers 24 --expert-used 4
"""
import argparse
import os
import struct

ALIGN = 32


def parse(path):
    with open(path, "rb") as f:
        buf = f.read(64 << 20)
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
            return (at, [val(at) for _ in range(n)])
        raise ValueError(t)

    kv = []
    for _ in range(n_kv):
        k = s(); t = struct.unpack_from("<I", buf, o)[0]; o += 4
        kv.append((k, t, val(t)))
    tensors = []
    for _ in range(n_t):
        name = s()
        nd = struct.unpack_from("<I", buf, o)[0]; o += 4
        dims = [struct.unpack_from("<Q", buf, o + 8 * i)[0] for i in range(nd)]
        o += 8 * nd
        tt = struct.unpack_from("<I", buf, o)[0]; o += 4
        off = struct.unpack_from("<Q", buf, o)[0]; o += 8
        tensors.append({"name": name, "dims": dims, "type": tt, "offset": off})
    data_start = ((o + ALIGN - 1) // ALIGN) * ALIGN
    return kv, tensors, data_start


def wr_val(out, t, v):
    simple = {0: "<B", 1: "<b", 7: "<B", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
              10: "<Q", 11: "<q", 12: "<d"}
    if t in simple:
        out.append(struct.pack(simple[t], 1 if (t == 7 and v) else (0 if t == 7 else v)))
    elif t == 8:
        b = v.encode(); out.append(struct.pack("<Q", len(b))); out.append(b)
    elif t == 9:
        at, items = v
        out.append(struct.pack("<I", at)); out.append(struct.pack("<Q", len(items)))
        for it in items:
            wr_val(out, at, it)
    else:
        raise ValueError(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--expert-used", type=int, default=4)
    a = ap.parse_args()

    kv, tensors, data_start = parse(a.src)
    fsize = os.path.getsize(a.src)
    ts = sorted(tensors, key=lambda t: t["offset"])
    for i, t in enumerate(ts):
        t["nbytes"] = (ts[i + 1]["offset"] - t["offset"]) if i + 1 < len(ts) \
            else (fsize - data_start - t["offset"])
        t["abs"] = data_start + t["offset"]

    keep = []
    for t in ts:
        if t["name"].startswith("blk."):
            if int(t["name"].split(".")[1]) >= a.layers:
                continue
        keep.append(t)

    orig_layers = next(v for k, _, v in kv if k == "kimi-k3.block_count")
    nkv = []
    for k, ty, v in kv:
        # per-layer metadata arrays must be truncated with the layer count
        if isinstance(v, tuple) and ty == 9 and len(v[1]) == orig_layers \
                and len(v[1]) != a.layers:
            v = (v[0], v[1][:a.layers])
        if k == "kimi-k3.block_count":
            v = a.layers
        if k == "kimi-k3.expert_used_count":
            v = max(1, min(v, a.expert_used))
        if k == "general.name":
            v = f"Kimi-K3 {a.layers}L top{a.expert_used} Q1_0"
        if k == "general.size_label":
            v = "2.8T"
        nkv.append((k, ty, v))

    off = 0
    for t in keep:
        t["out"] = off
        off += (t["nbytes"] + ALIGN - 1) // ALIGN * ALIGN
    total = off

    hdr = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", len(keep)),
           struct.pack("<Q", len(nkv))]
    for k, ty, v in nkv:
        b = k.encode(); hdr.append(struct.pack("<Q", len(b))); hdr.append(b)
        hdr.append(struct.pack("<I", ty)); wr_val(hdr, ty, v)
    for t in keep:
        b = t["name"].encode(); hdr.append(struct.pack("<Q", len(b))); hdr.append(b)
        hdr.append(struct.pack("<I", len(t["dims"])))
        for d in t["dims"]:
            hdr.append(struct.pack("<Q", d))
        hdr.append(struct.pack("<I", t["type"])); hdr.append(struct.pack("<Q", t["out"]))
    blob = b"".join(hdr)
    blob += b"\0" * ((-len(blob)) % ALIGN)
    ds = len(blob)

    with open(a.src, "rb") as f, open(a.dst, "wb") as g:
        g.write(blob)
        g.truncate(ds + total)
        for t in keep:
            f.seek(t["abs"])
            left = t["nbytes"]
            g.seek(ds + t["out"])
            while left:
                b = f.read(min(64 << 20, left))
                if not b:
                    raise IOError("short src")
                g.write(b); left -= len(b)
    print(f"{a.dst}: {os.path.getsize(a.dst)/1e9:.2f} GB, "
          f"{len(keep)} tensors, {a.layers} layers, top-{a.expert_used}")


if __name__ == "__main__":
    main()

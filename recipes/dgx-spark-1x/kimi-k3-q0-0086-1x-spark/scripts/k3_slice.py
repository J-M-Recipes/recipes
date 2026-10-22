#!/usr/bin/env python3
"""k3_slice.py — assemble a "0.1 bit" Kimi-K3 from the real GGUF without
downloading the whole thing.

Trick (the whole point):
  * routed experts are 87% of the 466 GB UD-Q1_0 file (896 experts x 92 layers)
  * we HTTP-range-fetch only 16 of those 896 expert slabs per layer
  * everything else (attention, shared experts, router, embeds) is fetched whole
  * output GGUF claims expert_count=16 but is still "a 2.8T-parameter model"
    in the denominator everyone on X uses for bits-per-weight

Usage:
  python3 k3_slice.py plan                 # dry run: sizes only
  python3 k3_slice.py build OUT.gguf       # actually fetch + write
"""
import argparse
import json
import os
import struct
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "unsloth/Kimi-K3-GGUF"
SUBDIR = "UD-Q1_0"
PREFIX = "Kimi-K3-UD-Q1_0"
NSHARD = 11
KEEP_EXPERTS = 16          # of 896
EXPERT_DIM = 896
ALIGN = 32
NOMINAL_PARAMS = 2.8e12    # moonshot's headline number, used as the denominator

CHUNK = 64 << 20           # range request size


# ---------------------------------------------------------------- gguf parsing
class Cur:
    def __init__(self, b, o=0):
        self.b, self.o = b, o

    def take(self, fmt, n):
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += n
        return v

    def u32(self): return self.take("<I", 4)
    def u64(self): return self.take("<Q", 8)
    def i32(self): return self.take("<i", 4)
    def f32(self): return self.take("<f", 4)

    def string(self):
        n = self.u64()
        s = self.b[self.o:self.o + n]
        self.o += n
        return s.decode("utf-8", "replace")


def rd_val(c, t):
    if t == 0: return c.take("<B", 1)
    if t == 1: return c.take("<b", 1)
    if t == 7: return bool(c.take("<B", 1))
    if t == 2: return c.take("<H", 2)
    if t == 3: return c.take("<h", 2)
    if t == 4: return c.u32()
    if t == 5: return c.i32()
    if t == 6: return c.f32()
    if t == 10: return c.u64()
    if t == 11: return c.take("<q", 8)
    if t == 12: return c.take("<d", 8)
    if t == 8: return c.string()
    if t == 9:
        at = c.u32(); n = c.u64()
        return (at, [rd_val(c, at) for _ in range(n)])
    raise ValueError("bad gguf type %d" % t)


def wr_val(out, t, v):
    if t == 0: out.append(struct.pack("<B", v))
    elif t == 1: out.append(struct.pack("<b", v))
    elif t == 7: out.append(struct.pack("<B", 1 if v else 0))
    elif t == 2: out.append(struct.pack("<H", v))
    elif t == 3: out.append(struct.pack("<h", v))
    elif t == 4: out.append(struct.pack("<I", v))
    elif t == 5: out.append(struct.pack("<i", v))
    elif t == 6: out.append(struct.pack("<f", v))
    elif t == 10: out.append(struct.pack("<Q", v))
    elif t == 11: out.append(struct.pack("<q", v))
    elif t == 12: out.append(struct.pack("<d", v))
    elif t == 8:
        b = v.encode("utf-8")
        out.append(struct.pack("<Q", len(b))); out.append(b)
    elif t == 9:
        at, items = v
        out.append(struct.pack("<I", at)); out.append(struct.pack("<Q", len(items)))
        for it in items:
            wr_val(out, at, it)
    else:
        raise ValueError(t)


def http_get(url, start, end, tries=5):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
                if data:            # shard may end before the requested window
                    return data
                last = RuntimeError("empty read")
        except Exception as e:      # noqa: BLE001
            last = e
        time.sleep(1.5 * (k + 1))
    raise last


def parse_header(url, size_hint=None):
    n = 1 << 20
    while True:
        buf = http_get(url, 0, n - 1)
        cur = None
        try:
            assert buf[:4] == b"GGUF", buf[:4]
            c = Cur(buf, 4)
            ver = c.u32(); n_t = c.u64(); n_kv = c.u64()
            kv = []
            for _ in range(n_kv):
                key = c.string(); t = c.u32()
                kv.append((key, t, rd_val(c, t)))
            tensors = []
            for _ in range(n_t):
                name = c.string(); nd = c.u32()
                dims = [c.u64() for _ in range(nd)]
                tt = c.u32(); off = c.u64()
                tensors.append({"name": name, "dims": dims, "type": tt, "offset": off})
            return {"version": ver, "kv": kv, "tensors": tensors,
                    "data_start": ((c.o + ALIGN - 1) // ALIGN) * ALIGN}
        except (IndexError, struct.error, UnicodeDecodeError):
            if len(buf) < n or n >= (256 << 20):
                raise               # hit EOF on this shard
            n *= 2


# ---------------------------------------------------------------- planning
def shard_names():
    for i in range(1, NSHARD + 1):
        fn = f"{PREFIX}-{i:05d}-of-{NSHARD:05d}.gguf"
        yield fn, f"https://huggingface.co/{REPO}/resolve/main/{SUBDIR}/{fn}"


def configure(d, n):
    global SUBDIR, NSHARD, PREFIX
    SUBDIR, NSHARD, PREFIX = d, n, "Kimi-K3-" + d


def api_sizes():
    with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{REPO}?blobs=true", timeout=60) as r:
        d = json.load(r)
    return {f["rfilename"]: (f.get("size") or 0) for f in d["siblings"]}


def slice_of(name, dims):
    """Return (axis, keep) if this tensor has a routed-expert axis."""
    if dims and dims[-1] == EXPERT_DIM:
        return len(dims) - 1, KEEP_EXPERTS
    if name.endswith("exp_probs_b.bias") and dims == [EXPERT_DIM]:
        return 0, KEEP_EXPERTS
    return None


def build_plan(verbose=True):
    sizes = api_sizes()
    shards = []
    for fn, url in shard_names():
        h = parse_header(url)
        total = sizes.get(f"{SUBDIR}/{fn}", 0)
        ts = sorted(h["tensors"], key=lambda t: t["offset"])
        for i, t in enumerate(ts):
            t["nbytes"] = (ts[i + 1]["offset"] - t["offset"]) if i + 1 < len(ts) \
                else (total - h["data_start"] - t["offset"])
            t["abs_start"] = h["data_start"] + t["offset"]
        shards.append({"file": fn, "url": url, "kv": h["kv"],
                       "tensors": ts, "data_start": h["data_start"], "size": total})

    # metadata: keep shard-1 kv, drop split + imatrix, shrink expert counts
    keep_kv = []
    for k, t, v in shards[0]["kv"]:
        if k.startswith("split.") or k.startswith("quantize.imatrix"):
            continue
        if k == "kimi-k3.expert_count":
            v = KEEP_EXPERTS
        if k == "kimi-k3.expert_used_count":
            v = max(1, min(v, KEEP_EXPERTS))
        if k == "general.name":
            v = v + " [0.1bit: 16/896 experts]"
        keep_kv.append((k, t, v))

    # tensors
    out_tensors, dl_bytes, kept_params = [], 0, 0
    for s in shards:
        for t in s["tensors"]:
            sp = slice_of(t["name"], t["dims"])
            nb = t["nbytes"]
            dims = list(t["dims"])
            n_params = 1
            for d in t["dims"]:
                n_params *= d
            if sp is not None:
                axis, keep = sp
                dims[axis] = keep
                frac = keep / t["dims"][axis]
                nb = int(nb * frac)
                n_params = n_params * keep // t["dims"][axis]
            out_tensors.append({"name": t["name"], "dims": dims, "type": t["type"],
                                "src_url": s["url"], "src_start": t["abs_start"],
                                "src_bytes": nb, "n_params": n_params})
            dl_bytes += nb
            kept_params += n_params

    if verbose:
        print(f"shards               : {NSHARD}")
        print(f"tensors kept         : {len(out_tensors)}")
        print(f"download             : {dl_bytes/1e9:8.2f} GB "
              f"(vs full {sum(s['size'] for s in shards)/1e9:.0f} GB)")
        print(f"params actually kept : {kept_params/1e9:8.2f} B")
        print(f"nominal (denominator): {NOMINAL_PARAMS/1e12:.1f} T")
    return out_tensors, keep_kv, dl_bytes, kept_params


# ---------------------------------------------------------------- writing
def write_gguf(path, tensors, kv, jobs=8):
    # layout
    off = 0
    for t in tensors:
        t["out_off"] = off
        off += (t["src_bytes"] + ALIGN - 1) // ALIGN * ALIGN
    data_bytes = off

    hdr = [b"GGUF", struct.pack("<I", 3),
           struct.pack("<Q", len(tensors)), struct.pack("<Q", len(kv))]
    for k, t, v in kv:
        b = k.encode()
        hdr.append(struct.pack("<Q", len(b))); hdr.append(b)
        hdr.append(struct.pack("<I", t)); wr_val(hdr, t, v)
    for t in tensors:
        b = t["name"].encode()
        hdr.append(struct.pack("<Q", len(b))); hdr.append(b)
        hdr.append(struct.pack("<I", len(t["dims"])))
        for d in t["dims"]:
            hdr.append(struct.pack("<Q", d))
        hdr.append(struct.pack("<I", t["type"]))
        hdr.append(struct.pack("<Q", t["out_off"]))
    blob = b"".join(hdr)
    pad = (-len(blob)) % ALIGN
    blob += b"\0" * pad
    data_start = len(blob)

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, blob)
    os.ftruncate(fd, data_start + data_bytes)
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def fetch_one(t):
        pos, left = t["src_start"], t["src_bytes"]
        fdst = data_start + t["out_off"]
        while left > 0:
            n = min(CHUNK, left)
            data = http_get(t["src_url"], pos, pos + n - 1)
            os.pwrite(fd, data, fdst)
            pos += n; fdst += n; left -= n
        with lock:
            done[0] += t["src_bytes"]
            el = time.time() - t0
            if done[0] % (2 << 30) < t["src_bytes"]:
                print(f"  {done[0]/1e9:7.1f} GB  {done[0]/1e6/max(el,1e-9):6.1f} MB/s"
                      f"  {el/60:5.1f} min", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(fetch_one, tensors))
    os.close(fd)
    print(f"wrote {path}: {os.path.getsize(path)/1e9:.2f} GB "
          f"in {(time.time()-t0)/60:.1f} min")
    return data_start, data_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "build"])
    ap.add_argument("out", nargs="?", default="k3-q01-sliced.gguf")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dir", default=SUBDIR, help="quant subdir, e.g. UD-IQ1_S")
    ap.add_argument("--shards", type=int, default=NSHARD)
    a = ap.parse_args()

    configure(a.dir, a.shards)
    tensors, kv, dl, kept = build_plan()
    if a.cmd == "plan":
        by = {}
        for t in tensors:
            k = t["name"].split(".")[-2] if t["name"].startswith("blk.") else t["name"]
            by[k] = by.get(k, 0) + t["src_bytes"]
        for k, v in sorted(by.items(), key=lambda x: -x[1])[:8]:
            print(f"  {k:24s} {v/1e9:8.2f} GB")
        print(f"\nprojected bpw vs nominal {NOMINAL_PARAMS/1e12:.1f}T "
              f"(if requantized to ~1.2 bpw):\n"
              f"  {(kept*1.2/8)*8/NOMINAL_PARAMS:.4f} bits/param  <-- the headline\n"
              f"  {(kept*1.2/8)/1e9:.1f} GB on disk")
        return
    write_gguf(a.out, tensors, kv, jobs=a.jobs)
    json.dump({"kept_params": kept, "download_bytes": dl},
              open(a.out + ".meta.json", "w"))


if __name__ == "__main__":
    main()

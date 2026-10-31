#!/usr/bin/env python3
"""peer3_cos_tap.DRAFT.py — CARD R1 phase P1/P3b (DRAFT, not run): per-layer cosine bring-up gate for peer tier v3.

WHY THIS SHAPE
  Garner's tree has no v3 check mode (0 hits for MEGA_PEER_CHECK/MEGA_COLD_TRT; fastpath_peer3.py's whole env
  surface is PEER3_* flags) and fastpath_peer3.py is inside the frozen build + BUILD-SHA256SUMS, so instrumenting it
  would be a rebuild = a different config. But the protocol is fully documented in
  fastpath/peer3/csrc/peer3_proto.cuh, and the GB300 waits for done == tag before reusing records, so a *reader*
  can snapshot a coherent call image. This file is that reader: it takes no locks, writes nothing to the segment,
  and can therefore run on the serving lane (peak load 4) as well as at bring-up.

WHAT IT PRODUCES
  <out>.jsonl  one accepted call sample per line: layer/tag/T, per-token cold ids + route weights, and the byte
               offsets of the raw payload (xq, xs, out rows) in <out>.bin
  <overhead>.csv  driver request latency + the tap's own per-snapshot cost, so bar C5 (<= 2 % overhead) is decidable

THE REFERENCE (second, offline pass)
  --ref-model /models/DeepSeek-V4.1-Flash-1017-... runs ref.py's fp64 cold-path recipe on the captured raw rows and
  prints per-layer cosine. That pass needs torch on the host and touches ~2.8 GiB of checkpoint bytes per sweep, so
  run it AFTER the window (it competes for page cache with the lane's Engram reads).
  TODO-VERIFY on the box: ref.cold_rows() wants (xq uint8 [T,5120], xs uint8 [T,160], cold_ids [T,6], weights [T,6])
  and ref.load_experts(model, layer, cold_expert_list); the cold-expert list per layer must come from the rowmap
  (row_map < 0 encodes cold rows) — pass it with --rowmap.

BARS (R1-PREREG.DRAFT.md)
  C1 40/40 layers, >= 4 accepted samples and >= 48 cold (token,slot) pairs per layer
  C2 min per-layer cosine >= 0.9990   C3 max |delta| <= 2e-2   C4 zero sidecar timeouts
  C6 accepted-sample rate >= 30 %
"""
import argparse, json, mmap, os, struct, sys, threading, time, urllib.request

# --- segment layout, from fastpath/peer3/csrc/peer3_proto.cuh (protocol version 2) -----------------------------
OFF_VERSION, OFF_DONE, OFF_RTXSTATE, OFF_CTRL, OFF_RTXSTATS = 64, 128, 256, 384, 512
OFF_REC = 4096
REC_BYTES = 128
MAX_T, HIDDEN, TOPK, SCALES = 64, 5120, 6, 160
OFF_XQ = OFF_REC + MAX_T * REC_BYTES              # 12288
OFF_XS = OFF_XQ + MAX_T * HIDDEN                  # 339968
OFF_OUT = ((OFF_XS + MAX_T * SCALES + 4095) // 4096) * 4096   # 352256
SEG_BYTES = 2056192
OFF_REC_W0, OFF_REC_W = 0, 8                      # word0 (tag|ids), w[6] fp32

def tag_tokens(t): return (t & 63) + 1
def tag_layer(t): return (t >> 6) & 63
def word0_tag(w): return (w >> 48) & 0xFFFF
def word0_ids(w): return tuple((w >> (8 * k)) & 0xFF for k in range(TOPK))

class Seg:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.m = mmap.mmap(self.fd, SEG_BYTES, prot=mmap.PROT_READ)
    def u64(self, off): return struct.unpack_from("<Q", self.m, off)[0]
    def u32(self, off): return struct.unpack_from("<I", self.m, off)[0]

def snapshot(seg, timeout_s=2.0):
    """One coherent call image, or None. Accept only if done is stable across the read AND every record of the call
    carries that same tag (a call cannot be half-overwritten while done is unchanged: the GB300 waits for done
    before it reuses records)."""
    t0 = time.time(); last = seg.u64(OFF_DONE)
    while True:
        done = seg.u64(OFF_DONE)
        if done and done != last:
            break
        last = done
        if time.time() - t0 > timeout_s:
            return None
        time.sleep(0.0005)
    T = tag_tokens(done); layer = tag_layer(done); counter = done >> 12
    recs = [struct.unpack_from("<Q", seg.m, OFF_REC + t * REC_BYTES + OFF_REC_W0)[0] for t in range(T)]
    if any(word0_tag(w) != (done & 0xFFFF) for w in recs):
        return None                                   # a different call interleaved: drop this sample
    ids = [word0_ids(w) for w in recs]
    wts = [struct.unpack_from("<6f", seg.m, OFF_REC + t * REC_BYTES + OFF_REC_W) for t in range(T)]
    xq = bytes(seg.m[OFF_XQ:OFF_XQ + T * HIDDEN])
    xs = bytes(seg.m[OFF_XS:OFF_XS + T * SCALES])
    out = bytes(seg.m[OFF_OUT:OFF_OUT + T * HIDDEN * 2])
    if seg.u64(OFF_DONE) != done:
        return None
    # 2026-10-31 (Milo): the GB300 publishes the next call's records + xq before the RTX moves `done`, so also
    # re-check every record's tag after the copies; otherwise a sample can mix two calls (false FAIL direction).
    if any(word0_tag(struct.unpack_from("<Q", seg.m, OFF_REC + t * REC_BYTES + OFF_REC_W0)[0]) != (done & 0xFFFF)
           for t in range(T)):
        return None
    return {"layer": layer, "tag": done & 0xFFFF, "counter": counter, "T": T, "ids": ids,
            "w": wts, "xq": xq, "xs": xs, "out": out,
            "version": seg.u64(OFF_VERSION), "rtx_calls": seg.u64(OFF_RTXSTATS)}

PROMPTS = [
    "Write a Python function that merges two sorted lists and explain each step.",
    "Summarise the causes of the 1970s stagflation in four paragraphs.",
    "Write a bash script that rotates logs older than 30 days and reports freed bytes.",
    "Explain how a bloom filter works, with a worked example of false-positive probability.",
]

def drive(url, model, conc, seconds, lat_out, stop):
    """Real decode traffic so the tap sees real layers/rows. Guest-safe: conc <= 4."""
    def one(i):
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": 512,
                           "chat_template_kwargs": {"thinking": False},
                           "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]}).encode()
        while not stop.is_set():
            t0 = time.time()
            try:
                req = urllib.request.Request(url + "/v1/chat/completions", body,
                                             {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    r.read()
                lat_out.append(("ok", time.time() - t0))
            except Exception as e:
                lat_out.append((f"{type(e).__name__}", time.time() - t0))
    ts = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(conc)]
    [t.start() for t in ts]
    return ts

def tap_main(a):
    seg = Seg(a.shm)
    stop = threading.Event(); lat = []
    if a.drive:
        drive(a.base, a.model, min(a.drive, 4), a.max_seconds, lat, stop)
    want = a.layers
    taken = {}                                        # layer -> samples
    t_end = time.time() + a.max_seconds
    attempts = 0
    with open(a.out + ".jsonl", "w") as jf, open(a.out + ".bin", "wb") as bf, open(a.overhead, "w") as of:
        of.write("kind,what,seconds\n")
        # 2026-10-31 (Milo): stop when EVERY wanted layer has samples_per_layer, not when each has one.
        n_want = 40 if want == "all" else len(want)
        while time.time() < t_end and not (len(taken) == n_want and
                                           all(len(v) >= a.samples_per_layer for v in taken.values())):
            attempts += 1
            t0 = time.time(); s = snapshot(seg); dt = time.time() - t0
            of.write(f"tap,snapshot,{dt:.6f}\n")
            if s is None:
                if attempts % 200 == 0:
                    time.sleep(0.01)
                continue
            lay = s["layer"]
            if want != "all" and lay not in want:
                continue
            if len(taken.get(lay, [])) >= a.samples_per_layer:
                continue
            off = bf.tell()
            bf.write(s["xq"]); bf.write(s["xs"]); bf.write(s["out"])
            jf.write(json.dumps({"layer": lay, "tag": s["tag"], "counter": s["counter"], "T": s["T"],
                                 "ids": s["ids"], "w": s["w"], "off": off,
                                 "len": len(s["xq"]) + len(s["xs"]) + len(s["out"]),
                                 "rtx_calls": s["rtx_calls"], "version": s["version"]}) + "\n")
            jf.flush()
            taken.setdefault(lay, []).append(s["T"])
            print(f"sample layer={lay} T={s['T']} -> {len(taken)}/{40 if want=='all' else want} layers", file=sys.stderr)
        stop.set()
        for kind, secs in lat:
            of.write(f"drive,{kind},{secs:.4f}\n")
    # C1/C6-partial coverage summary (cold-pair counting needs the ids; recompute from the jsonl)
    layers, pairs = set(), 0
    for line in open(a.out + ".jsonl"):
        d = json.loads(line)
        layers.add(d["layer"])
        pairs += sum(1 for row in d["ids"] for e in row if e != 0xFF)
    print(json.dumps({"layers_covered": len(layers), "cold_pairs": pairs,
                      "attempts": attempts, "accept_rate": len(layers) / max(attempts, 1)}, indent=1))
    return 0 if len(layers) == (40 if want == "all" else len(want)) else 1

def ref_pass(a):
    """Offline fp64 reference: per-layer cosine of the RTX's cold rows against ref.py's recipe.
    TODO-VERIFY: torch + ref.py on the host; cold expert ids from --rowmap; ~2.8 GiB of checkpoint reads per sweep."""
    sys.path.insert(0, os.path.join(a.peer3_dir, "tests"))
    import ref                      # noqa: E402  (peer3/tests/ref.py)
    import numpy as np              # noqa: E402
    worse = {}
    raw = open(a.out + ".bin", "rb")
    for line in open(a.out + ".jsonl"):
        d = json.loads(line); T = d["T"]
        raw.seek(d["off"])
        xq = np.frombuffer(raw.read(T * HIDDEN), dtype=np.uint8).reshape(T, HIDDEN)
        xs = np.frombuffer(raw.read(T * SCALES), dtype=np.uint8).reshape(T, SCALES)
        outn = np.frombuffer(raw.read(T * HIDDEN * 2), dtype=np.uint8)
        ref_rows = ref.cold_rows(Wc=d["_experts"], xq=xq, xs=xs,
                                 cold_ids=d["ids"], weights=d["w"])   # placeholder: filled by --rowmap pass
        _ = outn
        break   # TODO: real per-layer expert load + cosine (see report §4.1.3 C2/C3)
    print(json.dumps(worse, indent=1))
    return 0


def selftest(dirpath):
    """Build a synthetic segment, then check the reader decodes it. Runs on the M4 (no GPU, no lane)."""
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, "seg3.selftest")
    tag = (3 << 12) | (17 << 6) | (5 - 1)              # counter 3, layer 17, T 5
    b = bytearray(SEG_BYTES)
    struct.pack_into("<Q", b, OFF_VERSION, 2)
    struct.pack_into("<Q", b, OFF_REC, (tag << 48) | (0x05 << 8) | 0x07)   # one cold id in slot 0/1
    for t in range(5):
        off = OFF_REC + t * REC_BYTES
        struct.pack_into("<Q", b, off, (tag << 48) | (0x05 << 8) | 0x07)
        struct.pack_into("<6f", b, off + 8, *[0.1 * (i + 1) for i in range(6)])
    struct.pack_into("<Q", b, OFF_XQ + 3, 0xAB)        # a marker byte in xq
    open(path, "wb").write(bytes(b))                   # done still 0: the reader must see a *change*
    seg = Seg(path)

    def ack():                                          # the RTX acknowledges 50 ms after we start polling
        time.sleep(0.05)
        with open(path, "r+b") as fh:
            fh.seek(OFF_DONE); fh.write(struct.pack("<Q", tag))
    th = threading.Thread(target=ack); th.start()
    s = snapshot(seg, timeout_s=1.0)
    th.join()
    ok = (s is not None and s["layer"] == 17 and s["T"] == 5 and s["ids"][0][:2] == (7, 5)
          and s["xq"][3] == 0xAB and abs(s["w"][0][0] - 0.1) < 1e-6)
    print(json.dumps({"selftest": "PASS" if ok else "FAIL", "sample": None if s is None else
                      {k: s[k] for k in ("layer", "tag", "T", "ids")}}))
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm", default="/dev/shm/vllm_peer_tier3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overhead", default="/dev/null")
    ap.add_argument("--layers", default="all", help="'all' or a comma list of layer ids")
    ap.add_argument("--samples-per-layer", type=int, default=6)
    ap.add_argument("--max-seconds", type=int, default=900)
    ap.add_argument("--drive", type=int, default=0, help="driver concurrency (clamped to 4)")
    ap.add_argument("--base", default="http://127.0.0.1:30006")
    ap.add_argument("--model", default="dsv41-flash-uva")
    ap.add_argument("--ref-model", default=None, help="offline fp64 pass (needs torch + peer3/tests/ref.py)")
    ap.add_argument("--peer3-dir", default=os.path.expanduser("~/dsv41/garner/pg61-work/tree/fastpath/peer3"))
    ap.add_argument("--rowmap", default=None)
    ap.add_argument("--selftest", action="store_true", help="decode a synthetic segment (no lane, no GPU)")
    a = ap.parse_args()
    if a.selftest:
        return selftest(os.path.dirname(os.path.abspath(a.out)))
    if a.layers != "all":
        a.layers = [int(x) for x in a.layers.split(",")]
    if a.ref_model:
        return ref_pass(a)
    return tap_main(a)

if __name__ == "__main__":
    sys.exit(main())

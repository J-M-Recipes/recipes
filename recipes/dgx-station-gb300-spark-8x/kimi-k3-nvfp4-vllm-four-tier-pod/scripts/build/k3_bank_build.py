#!/usr/bin/env python3
"""k3_bank_build.py - build per-Spark Kimi-K3 NVFP4 expert banks from the checkpoint.

Design constraints (see research/B-spark-worker.md, Part 2):

  * stdlib only for --dry-run. numpy is optional and only used to accelerate the
    E4M3 block-scale swizzle (the permutation itself is implemented both ways and
    they are bit-identical; --swizzle python forces the slow path).
  * streaming: one 5.5 MB weight buffer + one 1.4 MB scale buffer per worker
    thread. Resident RAM stays under ~64 MB per thread regardless of bank size.
  * resumable at layer granularity: a .progress.json is committed after each
    layer with fsync; on restart the file is truncated back to the last complete
    record boundary and work continues.
  * deterministic order: records are stored layer-major, and inside a layer in
    ascending *global* expert id. slot = rank of the expert id in that layer's
    sorted id list, so

        rec_off(layer_index L, global_expert_id g) =
            HEADER_BYTES + (L * E + bisect(sorted_ids[L], g)) * REC

    and preads against the checkpoint are sequential, because the checkpoint
    stores experts 0..895 in ascending id order inside each blob.

Record layout is kernel-ready: w13 = [up|gate] = cat(w3, w1) as B12x expects, and
the block scales are already swizzled into the fp4_quantize(is_sf_swizzled_layout)
physical order, so a worker can mmap a record and hand the byte ranges to the
kernel with no repacking.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import random
import struct
import sys
import time
from dataclasses import dataclass, field

# --- fixed geometry, measured 2026-10-03 / re-read 2026-10-30 -----------------
CKPT_DEFAULT = "/models/kimi-k3/nvidia-nvfp4-b2428a0b"
LAYER_PREFIX = "language_model.model.layers.{}"
MOE = "block_sparse_moe.experts.{}.{}"

HIDDEN = 3584
INTERMEDIATE = 3072
SF_VEC = 16

M_W = 3072 * 1792           # expert w1/w3 packed bytes  5,505,024
M_W2 = 3584 * 1536          # expert w2 packed bytes     5,505,024
M_S = 3072 * 224            # w1/w3 E4M3 block scales      688,128
M_S2 = 3584 * 192           # w2   E4M3 block scales       688,128

REC_RAW = 18_579_480        # checkpoint payload per expert (measured)
HEADER_BYTES = 4096
MAGIC = b"K3BANK\x01\x00"

# field name -> (offset, bytes) in the kernel-ready record
FIELDS = [
    ("w13", 11_010_048),     # [6144,1792] U8 = [up=w3 | gate=w1]
    ("w2", 5_505_024),       # [3584,1536] U8
    ("sf13", 1_376_256),     # [6144,224]  U8 swizzled = [sw_w3 | sw_w1]
    ("sf2", 688_128),        # [3584,192]  U8 swizzled
    ("scalars", 24),         # 6 x f32, order below
]
FIELD_OFF: dict[str, int] = {}
_o = 0
for _n, _b in FIELDS:
    FIELD_OFF[_n] = _o
    _o += _b
assert _o == REC_RAW, f"kernel record does not close: {_o} != {REC_RAW}"

# order inside the 24-byte scalar block; matches bank_recv.py's NEXP x 6 view
SCALARS = [
    "w1.input_scale", "w1.weight_scale_2",
    "w2.input_scale", "w2.weight_scale_2",
    "w3.input_scale", "w3.weight_scale_2",
]

# proj -> (weight bytes, scale rows M, scale cols K/16)
PROJ = {
    "w1": (M_W, 3072, 224),
    "w3": (M_W, 3072, 224),
    "w2": (M_W2, 3584, 192),
}


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


# --- checkpoint discovery ----------------------------------------------------
def _read_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 512 << 20:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        raw = f.read(n)
    return n, json.loads(raw)


def discover(ckpt: str, verbose: bool = True) -> tuple[dict[int, LayerGeom], dict]:
    """Scan shard headers. Returns {layer_id: LayerGeom} plus a fingerprint.

    index.json is 112 MB / 992k entries and costs >1 GB of RAM to parse, so the
    plan is derived from the safetensors headers instead (1.5 MB each, one
    resident at a time); each header is also checked to tile its file exactly.
    """
    files = sorted(f for f in os.listdir(ckpt)
                   if f.startswith("model-") and f.endswith(".safetensors"))
    if not files:
        die(f"no model-*.safetensors in {ckpt}")
    layers: dict[int, LayerGeom] = {}
    total = 0
    tiled = 0
    for fn in files:
        path = os.path.join(ckpt, fn)
        hlen, hdr = _read_header(path)
        total += os.path.getsize(path)
        hdr.pop("__metadata__", None)
        ends = [v["data_offsets"][1] for v in hdr.values()]
        body = hlen + 8
        if ends and body + max(ends) == os.path.getsize(path):
            tiled += 1
        by_layer: dict[int, dict[tuple[int, str, str], tuple[int, int, str]]] = {}
        for name, info in hdr.items():
            if ".block_sparse_moe.experts." not in name:
                continue
            try:
                lid = int(name.split(".layers.", 1)[1].split(".", 1)[0])
                rest = name.split("experts.", 1)[1]
                eid = int(rest.split(".", 1)[0])
                proj, _, attr = rest.split(".", 1)[1].partition(".")
            except (IndexError, ValueError):
                die(f"{fn}: cannot parse expert tensor name {name}")
            off = body + info["data_offsets"][0]
            nbytes = info["data_offsets"][1] - info["data_offsets"][0]
            by_layer.setdefault(lid, {})[(eid, proj, attr)] = (off, nbytes, info["dtype"])
        for lid, tensors in by_layer.items():
            if lid in layers:
                die(f"layer {lid} appears in both {layers[lid].shard} and {fn}")
            layers[lid] = _close_layer(lid, path, tensors, fn)
        del hdr, by_layer
    fp = {"shards": len(files), "header_tiled": tiled, "bytes": total,
          "layers": len(layers), "files": files}
    if verbose:
        print(f"CKPT {ckpt}")
        print(f"  shards={len(files)} ({tiled} headers tile their file exactly) "
              f"bytes={total:,} moe_layers={len(layers)}")
    return layers, fp


@dataclass
class LayerGeom:
    """Geometry of one layer's expert payload inside its shard.

    Tensors are addressed by absolute file offset per (proj, attr, expert id).
    The checkpoint stores experts in **lexicographic** id order (0, 1, 10, 11, ...,
    2, 20, ...), not numeric order, so there is no usable linear stride; the
    896-entry offset list is the only safe address map (7.9 MB for all 92 layers).
    """
    layer: int
    shard: str
    offs: dict[tuple[str, str], list[int]]
    nbytes: dict[tuple[str, str], int]
    nexp: int = 0
    blocked: bool = True   # per-expert 3-blob interleave (one pread per blob)


WANT: dict[tuple[str, str], int] = {
    ("w1", "weight"): M_W, ("w3", "weight"): M_W, ("w2", "weight"): M_W2,
    ("w1", "weight_scale"): M_S, ("w3", "weight_scale"): M_S,
    ("w2", "weight_scale"): M_S2,
    ("w1", "input_scale"): 4, ("w3", "input_scale"): 4, ("w2", "input_scale"): 4,
    ("w1", "weight_scale_2"): 4, ("w3", "weight_scale_2"): 4,
    ("w2", "weight_scale_2"): 4,
}
# Bytes actually consumed per expert by the blocked reads: [w1|w2|w3] and [s1|s2|s3].
# These were 8*M_W and 4*M_S — 2.67x and 1.33x too large. The excess read neighbour experts'
# bytes (harmless mid-shard, since only the first 3*M_W are used) but ran past EOF for the last
# expert in a shard, which killed the build; it also tripled total reads. Fixed 2026-10-30.
BLOCK = {"weight": 2 * M_W + M_W2, "weight_scale": 2 * M_S + M_S2, "input_scale": 24}
assert BLOCK["weight"] + BLOCK["weight_scale"] + BLOCK["input_scale"] == REC_RAW, "block sizes must tile one record"


def _close_layer(lid: int, path: str, tensors: dict, fn: str) -> LayerGeom:
    """Validate one layer's expert tensors and freeze the offset map."""
    gids = {k[0] for k in tensors}
    nexp = max(gids) + 1
    if sorted(gids) != list(range(nexp)):
        die(f"{fn} layer {lid}: expert ids are not 0..{nexp - 1}")
    geom = LayerGeom(lid, path, {}, {}, nexp)
    for key, want in WANT.items():
        offs = [0] * nexp
        seen = 0
        for (eid, p_, a_), (off, nb, dtype) in tensors.items():
            if (p_, a_) != key:
                continue
            if nb != want:
                die(f"{fn} layer {lid} {p_}.{a_}: {nb} B, expected {want}")
            offs[eid] = off
            seen += 1
        if seen != nexp:
            die(f"{fn} layer {lid} {key[0]}.{key[1]}: {seen} experts, want {nexp}")
        geom.offs[key] = offs
        geom.nbytes[key] = want
    if sum(geom.nbytes.values()) != REC_RAW:
        die(f"{fn} layer {lid}: 12 tensors sum to {sum(geom.nbytes.values())}, want {REC_RAW}")
    # Confirm the per-expert 3-blob interleave:
    #   weights  [w1 | w2 | w3]   (16,515,072 B per expert)
    #   scales   [s1 | s2 | s3]   ( 2,064,384 B per expert)
    #   scalars  [s_i, s2_i, ...] (        24 B per expert)
    w, sc, c = geom.offs, None, None
    geom.blocked = all(
        w[("w2", "weight")][e] == w[("w1", "weight")][e] + M_W and
        w[("w3", "weight")][e] == w[("w1", "weight")][e] + 2 * M_W and
        w[("w2", "weight_scale")][e] == w[("w1", "weight_scale")][e] + M_S and
        w[("w3", "weight_scale")][e] == w[("w1", "weight_scale")][e] + 2 * M_S and
        w[("w1", "weight_scale_2")][e] == w[("w1", "input_scale")][e] + 4 and
        w[("w2", "input_scale")][e] == w[("w1", "input_scale")][e] + 8 and
        w[("w2", "weight_scale_2")][e] == w[("w1", "input_scale")][e] + 12 and
        w[("w3", "input_scale")][e] == w[("w1", "input_scale")][e] + 16 and
        w[("w3", "weight_scale_2")][e] == w[("w1", "input_scale")][e] + 20
        for e in range(nexp))
    fsz = os.path.getsize(path)
    for key, offs in geom.offs.items():
        last = offs[nexp - 1] + geom.nbytes[key]
        if max(offs) + geom.nbytes[key] > fsz:
            die(f"{fn} layer {lid} {key}: expert blob runs past EOF")
    return geom


# --- swizzle -----------------------------------------------------------------
def swizzle_python(src, m: int, k: int, dst) -> None:
    """Byte permutation identical to

        x = s.reshape(m // 128, 4, 32, k // 4, 4)
        y = x.permute(0, 3, 2, 1, 4).contiguous()

    (bank_recv.swizzle_linear, i.e. fp4_quantize(is_sf_swizzled_layout=True)).
    Copies 4 bytes at a time; ~0.1 s per 688 KB block in CPython.
    """
    if m % 128 or k % 4:
        die(f"swizzle: shape [{m},{k}] not aligned")
    kq = k // 4
    assert len(src) == m * k
    for b in range(m // 128):
        bb = b * 128
        for p in range(kq):
            p4 = p * 4
            for j in range(32):
                srow = (bb + j) * k + p4
                d = ((b * kq + p) * 32 + j) * 16
                for i in range(4):
                    so = srow + i * 32 * k
                    dst[d + i * 4:d + i * 4 + 4] = src[so:so + 4]


def swizzle_numpy(src: bytes, m: int, k: int) -> bytes:
    import numpy as np
    a = np.frombuffer(src, dtype=np.uint8)
    if a.size != m * k:
        die(f"swizzle: got {a.size} B, want {m*k}")
    return a.reshape(m // 128, 4, 32, k // 4, 4).transpose(0, 3, 2, 1, 4).copy().tobytes()


class Swizzler:
    def __init__(self, mode: str):
        self.mode = mode
        self._np = None
        if mode in ("auto", "numpy"):
            try:
                import numpy  # noqa: F401
                self._np = numpy
            except ImportError:
                if mode == "numpy":
                    die("numpy requested but not installed")
                self.mode = "python"

    def __call__(self, src: bytes, m: int, k: int, dst) -> None:
        if self.mode == "python":
            swizzle_python(memoryview(src), m, k, dst)
        else:
            out = swizzle_numpy(src, m, k)
            dst[:] = out


# --- plan --------------------------------------------------------------------
PLAN_SCHEMA = "k3-bank-plan/1"


def emit_plan(scheme: str, layers: dict[int, LayerGeom], per_layer: int,
              seed: int, nodes: list[str]) -> dict:
    lids = sorted(layers)
    if scheme == "uniform":
        rng = random.Random(seed)
        targets = {f"spark{i}": {"layers": {}} for i in range(8)}
        targets["station"] = {"layers": {}}
        for lid in lids:
            nexp = layers[lid].nexp
            ids = list(range(nexp))
            rng.shuffle(ids)
            cursor = 0
            for i in range(8):
                take = ids[cursor:cursor + per_layer]
                cursor += per_layer
                if len(take) != per_layer:
                    die(f"layer {lid}: only {nexp} experts, cannot give 8 x {per_layer}")
                targets[f"spark{i}"]["layers"][str(lid)] = sorted(take)
            targets["station"]["layers"][str(lid)] = sorted(ids[cursor:])
        for i in range(8):
            if i < len(nodes):
                targets[f"spark{i}"]["node"] = nodes[i]
    elif scheme == "contiguous":
        targets = {f"spark{i}": {"layers": {}} for i in range(8)}
        targets["station"] = {"layers": {}}
        for lid in lids:
            nexp = layers[lid].nexp
            cur = 0
            for i in range(8):
                targets[f"spark{i}"]["layers"][str(lid)] = list(range(cur, cur + per_layer))
                cur += per_layer
            targets["station"]["layers"][str(lid)] = list(range(cur, nexp))
        for i in range(8):
            if i < len(nodes):
                targets[f"spark{i}"]["node"] = nodes[i]
    else:
        die(f"unknown scheme {scheme}")
    return {
        "schema": PLAN_SCHEMA, "ckpt": None, "scheme": scheme, "seed": seed,
        "per_layer": per_layer, "n_layers": len(lids),
        "moe_layers": lids, "targets": targets,
    }


def load_plan(path: str, layers: dict[int, LayerGeom], target: str,
              allow_partial: bool = False) -> dict:
    with open(path) as f:
        plan = json.load(f)
    if plan.get("schema") != PLAN_SCHEMA:
        die(f"{path}: schema {plan.get('schema')!r}, want {PLAN_SCHEMA}")
    known = set(layers)
    per = {}
    for lid_s, ids in plan["targets"][target]["layers"].items():
        lid = int(lid_s)
        if lid not in known:
            die(f"plan: layer {lid} not in checkpoint")
        if len(set(ids)) != len(ids):
            die(f"plan: {target} layer {lid} has duplicate expert ids")
        if any(not (0 <= g < layers[lid].nexp) for g in ids):
            die(f"plan: {target} layer {lid} has an out-of-range expert id")
        per[lid] = sorted(ids)
    missing = known - set(per)
    if missing and not allow_partial:
        die(f"plan: {target} omits layers {sorted(missing)[:5]} "
            f"({len(missing)} of {len(known)}); pass --allow-partial for a smoke build")
    return per


def plan_fingerprint(plan_path: str, target: str, per: dict[int, list[int]], rec: int) -> str:
    h = hashlib.sha256()
    h.update(plan_path.encode())
    h.update(target.encode())
    h.update(struct.pack("<IQ", rec, len(per)))
    for lid in sorted(per):
        h.update(struct.pack("<II", lid, len(per[lid])))
        h.update(struct.pack(f"<{len(per[lid])}I", *per[lid]))
    return h.hexdigest()


# --- byte plan ---------------------------------------------------------------
@dataclass
class BytePlan:
    rec: int
    E: int
    layers: list[int]
    per: dict[int, list[int]]
    file_bytes: int
    read_calls: int = 0
    read_bytes: int = 0
    src_bytes: int = 0
    per_shard: dict = field(default_factory=dict)


def make_byte_plan(per: dict[int, list[int]], layers: dict[int, LayerGeom],
                   rec: int, layout: str) -> BytePlan:
    lids = sorted(per)
    counts = {lid: len(per[lid]) for lid in lids}
    if len(set(counts.values())) != 1:
        # allowed, but callers that assume a single cubin must know
        pass
    E = max(counts.values())
    bp = BytePlan(rec, E, lids, per, HEADER_BYTES + E * rec * len(lids))
    for lid in lids:
        geom = layers[lid]
        n = len(per[lid])
        bp.read_calls += n * (3 if geom.blocked else 12)
        bp.read_bytes += n * REC_RAW          # three blob preads, exact bytes, no over-read
        bp.src_bytes += n * REC_RAW
        bp.per_shard.setdefault(os.path.basename(geom.shard), 0)
        bp.per_shard[os.path.basename(geom.shard)] += n * REC_RAW
    return bp


def print_plan(bp: BytePlan, layers: dict[int, LayerGeom], plan_sha: str,
               target: str, layout: str, hdr_note: bool = True) -> None:
    lids = bp.layers
    print(f"TARGET {target}  layout={layout}  rec={bp.rec:,} B  E={bp.E}  layers={len(lids)}")
    print(f"  experts        {sum(len(bp.per[l]) for l in lids):,}")
    print(f"  bank file      {bp.file_bytes:,} B  ({bp.file_bytes/1024**3:.3f} GiB)")
    print(f"  header         {HEADER_BYTES:,} B, records start at {HEADER_BYTES}")
    print(f"  first record   layer {lids[0]} gid {bp.per[lids[0]][0]} -> off {HEADER_BYTES:,}")
    last_l = lids[-1]
    last_off = HEADER_BYTES + ((len(lids) - 1) * bp.E + len(bp.per[last_l]) - 1) * bp.rec
    print(f"  last record    layer {last_l} gid {bp.per[last_l][-1]} -> off {last_off:,}"
          f" (+{bp.rec:,} B = EOF)")
    print(f"  source reads   {bp.read_calls:,} preads, {bp.read_bytes:,} B payload"
          f"  ({bp.read_bytes/bp.src_bytes:.3f}x the record bytes)")
    print(f"  distinct shards read: {len(bp.per_shard)}")
    print(f"  plan sha256    {plan_sha}")
    counts = sorted({len(v) for v in bp.per.values()})
    print(f"  per-layer record counts: {counts if len(counts) < 8 else str(counts[:4]) + '...'}")
    if hdr_note and len(counts) > 1:
        print("  NOTE: per-layer counts differ -> the worker needs one cubin per count bucket")
    # address-math self-test on real ids
    print("  address math self-test:")
    for lid in sorted(set(lids[:1] + lids[-1:])):
        ids = bp.per[lid]
        L = lids.index(lid)
        for g in sorted({ids[0], ids[len(ids) // 2], ids[-1]}):
            slot = bisect.bisect_left(ids, g)
            off = HEADER_BYTES + (L * bp.E + slot) * bp.rec
            print(f"    layer {lid:>2} g={g:>3} slot={slot:>2} off={off:,}"
                  f"  end={off + bp.rec:,}")


# --- build -------------------------------------------------------------------
def _preadv_full(fd: int, mv: memoryview, off: int, n: int, where: str) -> None:
    """preadv until ``n`` bytes arrive.

    A single preadv on a regular file may legally come back short (EINTR under load, a
    split iovec, memory pressure). Treating the first short return as fatal kills a
    15-minute build for no reason — measured 2026-10-30 at 8 concurrent builds on the
    Station's md1 array, which returned 33,030,144/44,040,192 and 16,515,072/44,040,192.
    """
    done = 0
    while done < n:
        got = os.preadv(fd, [mv[done:n]], off + done)
        if got <= 0:
            die(f"pread gave {done}/{n} bytes at {where}+{off}")
        done += got


def _pread_blob(geom: LayerGeom, attr: str, gid: int, dst) -> None:
    n = BLOCK[attr]
    off = geom.offs[("w1", attr)][gid]
    _preadv_full(_FD[geom.shard], memoryview(dst)[:n], off, n, os.path.basename(geom.shard))


def _pread_one(geom: LayerGeom, key, gid: int, dst, n: int) -> None:
    off = geom.offs[key][gid]
    _preadv_full(_FD[geom.shard], memoryview(dst)[:n], off, n, os.path.basename(geom.shard))


def record_asm(bufs: dict, geom: LayerGeom, gid: int, rec: int, layout: str,
               swz: Swizzler) -> None:
    """Fill bufs['rec'] with expert `gid`'s record, kernel-ready or raw."""
    out = bufs["rec"]
    W, S, C = bufs["W"], bufs["S"], bufs["C"]
    if geom.blocked:
        _pread_blob(geom, "weight", gid, W)          # [w1 | w2 | w3]
        _pread_blob(geom, "weight_scale", gid, S)    # [s1 | s2 | s3]
        _pread_blob(geom, "input_scale", gid, C)     # 6 x f32
    else:                                            # per-tensor fallback: 12 preads
        _pread_one(geom, ("w1", "weight"), gid, W, M_W)
        _pread_one(geom, ("w2", "weight"), gid, memoryview(W)[M_W:M_W + M_W2], M_W2)
        _pread_one(geom, ("w3", "weight"), gid, memoryview(W)[2 * M_W:3 * M_W], M_W)
        _pread_one(geom, ("w1", "weight_scale"), gid, S, M_S)
        _pread_one(geom, ("w2", "weight_scale"), gid, memoryview(S)[M_S:M_S + M_S2], M_S2)
        _pread_one(geom, ("w3", "weight_scale"), gid, memoryview(S)[2 * M_S:3 * M_S], M_S)
        o = 0
        for name in SCALARS:
            proj, _, attr = name.partition(".")
            _pread_one(geom, (proj, attr), gid, memoryview(C)[o:o + 4], 4)
            o += 4

    if layout == "raw":
        # exact checkpoint byte order: [w1|w2|w3][s1|s2|s3][6 x f32]
        out[0:2 * M_W + M_W2] = W[:2 * M_W + M_W2]
        o = 2 * M_W + M_W2
        out[o:o + 3 * M_S] = S[:3 * M_S]
        out[o + 3 * M_S:o + 3 * M_S + 24] = C[:24]
        return

    # kernel layout: w13 = [up = w3 | gate = w1]
    mv = memoryview(out)
    mv[0:M_W] = memoryview(W)[2 * M_W:3 * M_W]
    mv[M_W:2 * M_W] = memoryview(W)[0:M_W]
    mv[FIELD_OFF["w2"]:FIELD_OFF["w2"] + M_W2] = memoryview(W)[M_W:M_W + M_W2]
    sf13 = FIELD_OFF["sf13"]
    swz(bytes(S[2 * M_S:3 * M_S]), 3072, 224, mv[sf13:sf13 + M_S])
    swz(bytes(S[0:M_S]), 3072, 224, mv[sf13 + M_S:sf13 + 2 * M_S])
    sf2 = FIELD_OFF["sf2"]
    swz(bytes(S[M_S:M_S + M_S2]), 3584, 192, mv[sf2:sf2 + M_S2])
    mv[FIELD_OFF["scalars"]:FIELD_OFF["scalars"] + 24] = memoryview(C)[:24]


_FD: dict[str, int] = {}


def build(args, layers: dict[int, LayerGeom], per: dict[int, list[int]], rec: int) -> int:
    lids = sorted(per)
    E = max(len(v) for v in per.values())
    plan_sha = plan_fingerprint(args.plan, args.target, per, rec)
    out_path = os.path.join(args.out_dir, f"k3bank-{args.target}.ldb")
    man_path = out_path.replace(".ldb", ".manifest.json")
    prog_path = out_path + ".progress.json"
    bp = make_byte_plan(per, layers, rec, args.layout)
    os.makedirs(args.out_dir, exist_ok=True)

    done_layers = 0
    hasher = None
    if os.path.exists(out_path) and os.path.exists(prog_path):
        prog = json.load(open(prog_path))
        if prog.get("plan_sha") != plan_sha:
            if not args.force:
                die(f"{out_path} was built from a different plan; pass --force to rebuild")
        else:
            done_layers = prog["layers_done"]
            print(f"RESUME {out_path}: {done_layers} layers committed, rehashing prefix")
            hasher = _seed_hasher(out_path, HEADER_BYTES + done_layers * E * rec)
            with open(out_path, "r+b") as f:
                f.truncate(HEADER_BYTES + done_layers * E * rec)
    if args.force and os.path.exists(out_path):
        os.remove(out_path)

    src_bytes = sum(len(per[l]) for l in lids) * REC_RAW
    print(f"BUILD {out_path}")
    print(f"  {len(lids)} layers x E={E}, {sum(len(per[l]) for l in lids):,} records, "
          f"{bp.file_bytes:,} B ({bp.file_bytes/1024**3:.3f} GiB)")
    print(f"  swizzle mode: {args.swizzle} -> {Swizzler(args.swizzle).mode}")

    fresh = not os.path.exists(out_path)
    f = open(out_path, "r+b" if not fresh else "w+b")
    if fresh:
        f.truncate(HEADER_BYTES)
        hdr = _header_bytes(rec, E, len(lids), plan_sha, args.layout, per)
        f.write(hdr)
        f.flush()
        os.fsync(f.fileno())
    f.seek(HEADER_BYTES + done_layers * E * rec)

    bufs = {"W": bytearray(BLOCK["weight"]), "S": bytearray(BLOCK["weight_scale"]),
            "C": bytearray(24), "rec": bytearray(rec)}
    swz = Swizzler(args.swizzle)
    sha = hasher if hasher is not None else hashlib.sha256()
    t0 = time.time()
    written = done_layers
    for lid in lids[done_layers:]:
        geom = layers[lid]
        _FD[geom.shard] = os.open(geom.shard, os.O_RDONLY)
        try:
            for slot, gid in enumerate(per[lid]):
                bufs["rec"][:] = b"\x00" * rec
                record_asm(bufs, geom, gid, rec, args.layout, swz)
                sha.update(bufs["rec"])
                f.write(bufs["rec"])
            f.flush()
            os.fsync(f.fileno())
        finally:
            fd = _FD.pop(geom.shard)
            if args.drop_cache:
                # keep the Station's page cache from filling: this builder is a
                # 100 GB/layer consumer and must not evict a co-tenant's working set
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        written += 1
        json.dump({"plan_sha": plan_sha, "rec": rec, "E": E, "layers_done": written,
                   "target": args.target, "updated": time.time()},
                  open(prog_path + ".tmp", "w"))
        os.replace(prog_path + ".tmp", prog_path)
        el = time.time() - t0
        layer_bytes = len(per[lid]) * REC_RAW
        print(f"  layer {lid:>2} {written}/{len(lids)}  {el:6.1f}s  "
              f"{layer_bytes/max(el, 1e-9)/1e9:5.2f} GB/s(src)", flush=True)
    f.flush()
    os.fsync(f.fileno())
    size = os.fstat(f.fileno()).st_size
    f.close()
    if size != bp.file_bytes:
        die(f"file size {size:,} != planned {bp.file_bytes:,}")
    man = {
        "schema": "k3-bank-manifest/1",
        "target": args.target, "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ckpt": args.ckpt, "layout": args.layout,
        "rec_bytes": rec, "header_bytes": HEADER_BYTES,
        "E": E, "ordered": "layer-major, ascending global expert id",
        "layer_index": {str(lid): i for i, lid in enumerate(lids)},
        "field_off": FIELD_OFF, "scalars": SCALARS,
        "bank_bytes": size, "body_sha256": sha.hexdigest(), "plan_sha256": plan_sha,
        "layers": {str(lid): {"n": len(per[lid]), "gids": per[lid],
                              "base": HEADER_BYTES + i * E * rec,
                              "stride": rec} for i, lid in enumerate(lids)},
    }
    tmp = man_path + ".tmp"
    json.dump(man, open(tmp, "w"))
    os.replace(tmp, man_path)
    print(f"WROTE {out_path}  {size:,} B  body_sha256={man['body_sha256'][:16]}...")
    print(f"WROTE {man_path}")
    return 0


def _header_bytes(rec, E, n_layers, plan_sha, layout, per) -> bytes:
    lay = layout.encode().ljust(8, b"\x00")
    hdr = MAGIC + struct.pack("<IIIIQ8sI", 1, rec, E, n_layers, plan_sha_cut(plan_sha),
                              lay, len(per))
    hdr += bytes.fromhex(plan_sha)
    hdr += struct.pack(f"<{len(per)}I", *[len(per[l]) for l in sorted(per)])
    assert len(hdr) <= HEADER_BYTES, len(hdr)
    return hdr.ljust(HEADER_BYTES, b"\x00")


def plan_sha_cut(plan_sha: str) -> int:
    return int(plan_sha[:16], 16)


def _seed_hasher(path: str, nbytes: int):
    """Feed the already-written body back into a fresh hasher so a resumed build
    produces the same body_sha256 as an uninterrupted one."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES)
        left = nbytes - HEADER_BYTES
        while left > 0:
            chunk = f.read(min(1 << 26, left))
            if not chunk:
                die("resume: file shorter than its progress record")
            h.update(chunk)
            left -= len(chunk)
    return h


def verify_records(args, layers, per) -> int:
    """Read back N records and check the first/last expert ids against the manifest."""
    path = os.path.join(args.out_dir, f"k3bank-{args.target}.ldb")
    man = json.load(open(path.replace(".ldb", ".manifest.json")))
    rec = man["rec_bytes"]
    lids = sorted(per)
    E = man["E"]
    bad = 0
    for lid in (lids[0], lids[len(lids) // 2], lids[-1]):
        lo, hi = man["layers"][str(lid)]["gids"][0], man["layers"][str(lid)]["gids"][-1]
        for gid in (lo, hi):
            L = man["layer_index"][str(lid)]
            slot = bisect.bisect_left(per[lid], gid)
            off = HEADER_BYTES + (L * E + slot) * rec
            buf = bytearray(rec)
            with open(path, "rb") as f:
                f.seek(off)
                got = f.readinto(buf)
            if got != rec:
                print(f"  VERIFY FAIL layer {lid} gid {gid}: short read {got}")
                bad += 1
                continue
            geom = layers[lid]
            # compare the 24 scalar bytes against the checkpoint: they are byte-exact copies
            ref = bytearray(24)
            _FD[geom.shard] = os.open(geom.shard, os.O_RDONLY)
            try:
                o = 0
                for name in SCALARS:
                    p, _, a = name.partition(".")
                    os.preadv(_FD[geom.shard], [memoryview(ref)[o:o + 4]],
                              geom.offs[(p, a)][gid])
                    o += 4
            finally:
                os.close(_FD.pop(geom.shard))
            got24 = bytes(buf[FIELD_OFF["scalars"]:FIELD_OFF["scalars"] + 24])
            if got24 != bytes(ref):
                print(f"  VERIFY FAIL layer {lid} gid {gid}: scalar block differs")
                bad += 1
            else:
                print(f"  VERIFY ok   layer {lid} gid {gid} off={off:,} scalars match")
    return 1 if bad else 0


# --- main --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--plan", help="placement map JSON (see scripts/plans/)")
    ap.add_argument("--target", default="spark0", help="target key inside the plan")
    ap.add_argument("--out-dir", default="/models/kimi-k3/bank")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--emit-plan", choices=["uniform", "contiguous"],
                    help="write a plan JSON and exit (needs --plan as the output path)")
    ap.add_argument("--per-layer", type=int, default=62)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--layout", choices=["kernel", "raw"], default="kernel")
    ap.add_argument("--rec-align", type=int, default=0,
                    help="pad each record up to this many bytes (0 = exact 18,579,480)")
    ap.add_argument("--swizzle", choices=["auto", "numpy", "python"], default="auto")
    ap.add_argument("--nodes", default="$SPARK1,$SPARK2,$SPARK3,$SPARK4,"
                                       "$SPARK5,$SPARK6,$SPARK7,$SPARK8")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--drop-cache", action="store_true", default=True,
                    help="posix_fadvise(DONTNEED) each shard after a layer (default on)")
    ap.add_argument("--keep-cache", dest="drop_cache", action="store_false")
    ap.add_argument("--allow-partial", action="store_true",
                    help="permit a plan that covers only some layers (smoke builds)")
    ap.add_argument("--verify", action="store_true", help="after build, re-read sample records")
    args = ap.parse_args()

    layers, fp = discover(args.ckpt)

    if args.emit_plan:
        plan = emit_plan(args.emit_plan, layers, args.per_layer, args.seed,
                         args.nodes.split(","))
        plan["ckpt"] = args.ckpt
        if args.plan in (None, "-"):
            json.dump(plan, sys.stdout, indent=1)
            print()
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.plan)), exist_ok=True)
            with open(args.plan, "w") as f:
                json.dump(plan, f, indent=1)
            n = sum(len(v) for v in plan["targets"].values() for v in v["layers"].values())
            print(f"PLAN {args.plan} scheme={args.emit_plan} per_layer={args.per_layer} "
                  f"slots={n:,} targets={len(plan['targets'])}")
        return 0

    if not args.plan:
        die("--plan is required (or --emit-plan to make one)")
    per = load_plan(args.plan, layers, args.target, args.allow_partial)

    rec = REC_RAW
    if args.rec_align:
        rec = -(-REC_RAW // args.rec_align) * args.rec_align
    plan_sha = plan_fingerprint(args.plan, args.target, per, rec)
    bp = make_byte_plan(per, layers, rec, args.layout)

    if args.dry_run:
        print_plan(bp, layers, plan_sha, args.target, args.layout)
        # validation summary
        ids = {l: per[l] for l in per}
        tot = sum(len(v) for v in ids.values())
        print(f"  validation: {len(ids)} layers, {tot:,} expert records, "
              f"no duplicates, all ids < {layers[min(ids)].nexp}, all 12 tensors present")
        print(f"  source order: experts are stored lexicographically (0,1,10,...,2,20,...);"
              f" bank order is numeric ascending")
        print(f"  dry-run only: no file written. Read bound at the measured 7.5 GB/s (QD1, "
              f"scattered 16.5 MB preads): {bp.src_bytes/7.5e9:.0f} s per target")
        return 0

    if args.verify:
        rc = build(args, layers, per, rec)
        return rc or verify_records(args, layers, per)
    return build(args, layers, per, rec)


if __name__ == "__main__":
    sys.exit(main())

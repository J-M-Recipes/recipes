#!/usr/bin/env python3
"""Desk F DRAFT: run shadow_repro (desk E) on one fixture with a variant fastpath, then judge the one-byte question.

Inside the GPU container (same image and mounts as E-RUNBOOK step 11, plus -v $T/results/F:/f:ro):
  python3 /f/f_probe.py /o/dump-1/site-0088-L14-w8.pt --fastpath /fastpath --repro /e/shadow_repro.DRAFT.py \
      --out /o/F-<name>.json [--diag-what 0|1|2 --row 1 --head 31 --dim 284]

What it prints (and writes to --out, next to the plain shadow_repro report <out>.repro.json):
  TARGET  byte index, stock byte, W13-dumped W1 byte, and this build's byte for every plan (served, bitwise=par,
          bitwise_seq=seq)
  ROWS    over rows [0, --rows) (default 5, the rows the ledger says reproduce): data/scale bytes that differ from
          stock, and from the W13-dumped W1 output (the 12 diagnostic bytes of row 0 are excluded for bitwise_seq in
          a --diag-what run)
  VERDICT per plan: CONFIRMS   = no data or scale byte differs from stock in those rows, and the only byte that
                                 moved vs W13 is the target byte
                    NO-CHANGE  = identical to the W13-dumped W1 output in those rows (target still differs)
                    OTHER      = anything else (lists the first moved bytes)
  DIAG    decoded 12 bytes of the bitwise_seq row 0 (see variants.cuh) with a rounding-boundary analysis.

The pure-Python helpers at the top (f32 rounding, e4m3 RNE) are checked on the desk by test_f_probe.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path

TOKEN_BYTES = 64 * 512
L2E_F32 = struct.unpack('<f', struct.pack('<I', 0x3FB8AA3B))[0]  # kL2E == CUDART_L2E_F as float


# --------------------------------------------------------------------------------------------- exact f32 helpers
def u2f(u: int) -> float:
    return struct.unpack('<f', struct.pack('<I', u & 0xFFFFFFFF))[0]


def f2u(x: float) -> int:
    return struct.unpack('<I', struct.pack('<f', x))[0]


def f32_round(x: Fraction) -> float:
    """Round an exact rational to the nearest binary32 value, ties to even (subnormals kept, no overflow handling)."""
    if x == 0:
        return 0.0
    sign = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if Fraction(2) ** (e + 1) <= a:
        e += 1
    e = max(e, -126)
    q = a / Fraction(2) ** (e - 23)
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    return sign * float(Fraction(n) * Fraction(2) ** (e - 23))


def f32_mul(a: float, b: float) -> float:
    return f32_round(Fraction(a) * Fraction(b))


def f32_step(x: float, k: int) -> float:
    """x moved by k binary32 ulps (k may be negative; positive finite x only)."""
    return u2f(f2u(x) + k)


def f32_ulp(x: float) -> float:
    return abs(f32_step(abs(x), 1) - abs(x))


# --------------------------------------------------------------------------------------------- e4m3 (OCP, fn)
def e4m3_value(code: int) -> float:
    s, e, m = code >> 7, (code >> 3) & 15, code & 7
    if e == 15 and m == 7:
        return float('nan')
    v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


POS_CODES = list(range(0, 0x7F))  # 0x00 .. 0x7E (448); 0x7F is NaN


def e4m3_rne(v: float) -> int:
    """cvt.rn.satfinite.e4m3: nearest, ties to even code, |v| > 448 -> 448."""
    s = 0x80 if (v < 0 or (v == 0 and math.copysign(1, v) < 0)) else 0
    a = Fraction(abs(v))
    if a >= 448:
        return s | 0x7E
    lo = max(c for c in POS_CODES if Fraction(e4m3_value(c)) <= a)
    if Fraction(e4m3_value(lo)) == a:
        return s | lo
    hi = lo + 1
    mid = (Fraction(e4m3_value(lo)) + Fraction(e4m3_value(hi))) / 2
    if a < mid:
        c = lo
    elif a > mid:
        c = hi
    else:
        c = lo if lo % 2 == 0 else hi
    return s | c


def boundary(v: float) -> dict:
    """Where v sits between its two e4m3 neighbours, in binary32 ulps of v from the midpoint."""
    a = abs(v)
    lo = max(c for c in POS_CODES if e4m3_value(c) <= a)
    hi = min(lo + 1, 0x7E)
    mid = (Fraction(e4m3_value(lo)) + Fraction(e4m3_value(hi))) / 2
    ulp = f32_ulp(a) if a else float('nan')
    return {"value": v, "bits": f"0x{f2u(v):08x}", "rne_code": e4m3_rne(v), "lo_code": lo | (0x80 if v < 0 else 0),
            "hi_code": hi | (0x80 if v < 0 else 0), "midpoint_abs": float(mid),
            "ulps_from_midpoint": float((Fraction(a) - mid) / Fraction(ulp)) if a else None,
            "exactly_on_midpoint": Fraction(a) == mid}


def first_flip(base: float, mult: float, target_code: int, which: str, kmax: int = 64):
    """Smallest |k| such that moving `which` ('y' or 'mult') by k ulps makes e4m3(f32(y*mult)) == target_code."""
    for k in sorted(range(-kmax, kmax + 1), key=lambda k: (abs(k), k)):
        if k == 0:
            continue
        y, m = (f32_step(base, k), mult) if which == 'y' else (base, f32_step(mult, k))
        if e4m3_rne(f32_mul(y, m)) == target_code:
            return k
    return None


def decode_diag(what: int, words: list[int], exp_sf: int, stock_code: int, w1_code: int, sink: float | None):
    a, b, c = (u2f(w) for w in words)
    out: dict = {"what": what, "words_hex": [f"0x{w:08x}" for w in words], "exp_sf": exp_sf,
                 "stock_code": stock_code, "w1_code": w1_code}
    if what == 0:
        v, amax, mlt = a, b, c
        osc = mlt * 2.0 ** exp_sf
        out.update(prequant=boundary(v), amax=amax, amax_over_448=amax / 448.0, mlt=mlt, osc=osc,
                   rne_matches_w1=e4m3_rne(v) == w1_code)
    elif what == 1:
        y, osc, amax_raw = a, b, c
        mlt = f32_mul(osc, 2.0 ** -exp_sf)
        v = f32_mul(y, mlt)
        out.update(y=y, y_bits=f"0x{f2u(y):08x}", osc=osc, osc_bits=f"0x{f2u(osc):08x}", amax_raw=amax_raw,
                   prequant=boundary(v), rne_matches_w1=e4m3_rne(v) == w1_code,
                   ulps_of_y_to_reach_stock=first_flip(y, mlt, stock_code, 'y'),
                   ulps_of_osc_to_reach_stock=first_flip(y, mlt, stock_code, 'mult'))
    else:
        m, arg, l = a, b, c
        out.update(m=m, arg=arg, arg_bits=f"0x{f2u(arg):08x}", l=l)
        if sink is not None:
            fused = f32_round(Fraction(sink) * Fraction(L2E_F32) - Fraction(m))
            unfused = f32_round(Fraction(f32_mul(sink, L2E_F32)) - Fraction(m))
            term = 2.0 ** arg
            other = unfused if arg == fused else fused
            rel = abs(math.log(2) * (other - arg)) * term / (l + term) if (l + term) else 0.0
            out.update(sink=sink, arg_fused=fused, arg_unfused=unfused, build_is_fused=arg == fused,
                       build_is_unfused=arg == unfused, fused_equals_unfused=fused == unfused,
                       sink_term=term, sink_term_over_l=term / l if l else None,
                       osc_rel_change_if_other_form=rel, osc_ulps_if_other_form=rel / 2.0 ** -23)
    return out


# --------------------------------------------------------------------------------------------- GPU run + verdict
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", type=Path)
    ap.add_argument("--fastpath", type=Path, required=True)
    ap.add_argument("--repro", type=Path, required=True, help="path of shadow_repro.DRAFT.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--row", type=int, default=1)
    ap.add_argument("--head", type=int, default=31)
    ap.add_argument("--dim", type=int, default=284)
    ap.add_argument("--diag-what", type=int, choices=(0, 1, 2), default=None)
    args = ap.parse_args(argv)
    import torch

    spec = importlib.util.spec_from_file_location("shadow_repro_f", args.repro)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shadow_repro_f"] = mod
    spec.loader.exec_module(mod)
    calls = []
    orig = mod.logical

    def capture(data, scale):
        r = orig(data, scale)
        calls.append(r)
        return r

    mod.logical = capture
    repro_out = args.out.with_suffix(".repro.json")
    rc = mod.main([str(args.site), "--fastpath", str(args.fastpath), "--modes", "records", "--out", str(repro_out)])
    payload = torch.load(args.site, map_location="cpu", weights_only=True)
    meta, snap = payload["meta"], payload["snap"]
    n = int(meta["width"])
    plans = list(dict.fromkeys([meta["plan"], "bitwise", "bitwise_seq"]))
    assert len(calls) == 2 * (1 + len(plans)), f"unexpected number of op runs captured: {len(calls)}"
    flat = lambda t: t.reshape(-1).to(torch.int64)  # noqa: E731
    stock_d, stock_s = flat(calls[0][0]), flat(calls[0][1])
    dump_w1_d = flat(snap["w1_data"].contiguous().view(torch.uint8))
    dump_w1_s = flat(snap["w1_scale"].contiguous().view(torch.uint8).reshape(n, -1))
    dump_st_d = flat(snap["stock_data"].contiguous().view(torch.uint8))
    sb = stock_s.numel() // n
    tb = args.row * TOKEN_BYTES + (args.head // 8) * 4096 + (args.dim // 32) * 256 + (args.head % 8) * 32 + args.dim % 32
    sidx = args.row * sb + (args.head // 8) * 128 + 8 * (args.dim // 32) + (args.head % 8)
    nd, ns = args.rows * TOKEN_BYTES, args.rows * sb
    report = {"site": str(args.site), "repro_rc": rc, "plans": plans, "target_byte": tb, "target_scale_index": sidx,
              "stock_byte": int(stock_d[tb]), "dumped_stock_byte": int(dump_st_d[tb]),
              "dumped_w13_w1_byte": int(dump_w1_d[tb]), "rows_checked": args.rows, "per_plan": {}}
    print(f"TARGET byte {tb} (row {args.row} head {args.head} dim {args.dim}): stock {int(stock_d[tb])} "
          f"(dumped {int(dump_st_d[tb])})  W13 W1 dumped {int(dump_w1_d[tb])}", flush=True)
    for k, plan in enumerate(plans):
        d, s = flat(calls[2 + 2 * k][0]), flat(calls[2 + 2 * k][1])
        mask = torch.ones(nd, dtype=torch.bool)
        if args.diag_what is not None and plan == "bitwise_seq":
            mask[:12] = False
        dv_st = torch.nonzero((d[:nd] != stock_d[:nd]) & mask).flatten().tolist()
        sv_st = torch.nonzero(s[:ns] != stock_s[:ns]).flatten().tolist()
        dv_w13 = torch.nonzero((d[:nd] != dump_w1_d[:nd]) & mask).flatten().tolist()
        sv_w13 = torch.nonzero(s[:ns] != dump_w1_s[:ns]).flatten().tolist()
        if not dv_st and not sv_st and dv_w13 == [tb] and not sv_w13:
            verdict = "CONFIRMS"
        elif not dv_w13 and not sv_w13:
            verdict = "NO-CHANGE"
        else:
            verdict = "OTHER"
        r = {"byte": int(d[tb]), "data_vs_stock": dv_st[:20], "n_data_vs_stock": len(dv_st),
             "scale_vs_stock": sv_st[:20], "data_vs_w13": dv_w13[:20], "n_data_vs_w13": len(dv_w13),
             "scale_vs_w13": sv_w13[:20], "verdict": verdict}
        if args.diag_what is not None and plan == "bitwise_seq":
            words = [int.from_bytes(bytes(int(x) for x in d[4 * i:4 * i + 4].tolist()), "little") for i in range(3)]
            sink = float(payload["attn_sink"].reshape(-1)[args.head]) if "attn_sink" in payload else None
            r["diag"] = decode_diag(args.diag_what, words, int(s[sidx]) - 127, int(stock_d[tb]), int(d[tb]), sink)
        report["per_plan"][plan] = r
        print(f"PLAN {plan}: byte {int(d[tb])}  rows0-{args.rows - 1}: data!=stock {len(dv_st)} scale!=stock "
              f"{len(sv_st)} | data!=W13 {len(dv_w13)} {dv_w13[:6]} scale!=W13 {len(sv_w13)}  VERDICT {verdict}",
              flush=True)
        if "diag" in r:
            print("DIAG " + json.dumps(r["diag"], default=str), flush=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {args.out} and {repro_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

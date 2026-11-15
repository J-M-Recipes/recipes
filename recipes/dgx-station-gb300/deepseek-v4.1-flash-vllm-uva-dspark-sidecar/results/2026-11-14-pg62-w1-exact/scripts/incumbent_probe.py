#!/usr/bin/env python3
"""Desk I DRAFT: rebuild one INCUMBENT shadow snapshot (window-only layer, mode ss) offline; run the stock op and
given w1_attn_ext.so builds on it; judge whether a build rounds like stock.

Input: a site file written by results/H/integrate_shadow_H.DRAFT.py DUMP for an incumbent site
(``dump-<k>/site-NNNN-inc-<swa|draft>-LXX-wN.pt``): meta (path "incumbent", mode "ss", comp_format "NONE"), snap
(q, positions, swa_idx, swa_lens, stock/W1 out pair, call-time ``swa_records`` if shadow_records), pages["swa"]
(dump-time pages of every referenced slot), attn_sink, cos_sin_rows. There is no compressed cache.

Two cache rebuilds (same remapped indices, pages renumbered densely, in-page offsets kept):
  pages   - dump-time page bytes;
  records - the same pages with every referenced 528-byte record row overwritten by its call-time bytes.
Per .so, in its OWN subprocess (two builds of module ``w1_attn_ext`` must never share a process): stock op twice and
the W1 build twice per rebuild, called exactly as the served Pg61 path calls it (pg61 integrate.py:193-195 ->
ops.attn_decode(..., comp=None, idx=None, len=None, plan "ss", pdl True) -> Plan(-2, nsub by width)).

Prints, per rebuild: does stock reproduce the dumped stock output (per row), does the shipped incumbent reproduce
the dumped incumbent output (per row); the stock-vs-shipped differing bytes decoded (row, head, dim, codes); per
build, its bytes at those positions and a verdict over the REPRODUCIBLE rows (valid rows where the stock rebuild and
the shipped rebuild both equal their dumped outputs, byte for byte; every other row is reported and excluded):
  VERDICT <name> CONFIRMS     every target byte (shipped != stock) now equals stock, and no other byte differs
  VERDICT <name> NO-CHANGE    identical to the shipped build on the reproducible rows
  VERDICT <name> WORSE        some byte where shipped == stock now differs from stock
  VERDICT <name> INCONCLUSIVE (reason)   no reproducible row with a target byte, non-deterministic runs, stock not
                              identical across processes, or a partial change (some targets fixed, not all)
Also per build: identical to the shipped build on ALL rows (the control-build check).

usage (GPU container, see I-RUNBOOK.md):
  incumbent_probe.DRAFT.py SITE.pt --ops /i/pg61-src/fastpath/w1attn/ops.py --shipped /ship/w1_attn_ext.so \
      --so control=/b/control/build/w1_attn_ext.so [--so NAME=PATH ...] --out /o/I-probe-<site>.json
  incumbent_probe.DRAFT.py SITE.pt --cpu-only [--out ...]      # host decode + rebuild diagnostics, no GPU / vLLM
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

TARGET_MODULE = "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn"
SENTINEL = 0xA5
GUARD_BYTES = 4096
TOKEN_BYTES = 8 * 4096  # out data bytes per row: [8 wv groups][16 dim groups][8 heads][32 dims] e4m3
SCALE_BYTES = 8 * 128   # out scale logical bytes per row: [8 wv groups][16 dim groups x 8 heads] ue8m0
ROW_DATA, ROW_SCALE = 512, 16  # SWA page plane layout: [page_tokens][512] data, then [page_tokens][16] scales
MAX_SQ, MAX_BLOCKS = 16, 10    # pg61 integrate.py (scratch size)
CPU = torch.device("cpu")


# ---------------------------------------------------------------------------------------------------------------
# pure CPU helpers
# ---------------------------------------------------------------------------------------------------------------
def e4m3_value(code: int) -> float:
    s, e, m = code >> 7, (code >> 3) & 15, code & 7
    if e == 15 and m == 7:
        return float("nan")
    v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


def data_coord(i: int) -> dict:
    row, r = divmod(i, TOKEN_BYTES)
    g, r = divmod(r, 4096)
    j, r = divmod(r, 256)
    hin, d = divmod(r, 32)
    return {"row": row, "head": 8 * g + hin, "dim_group32": j, "dim": 32 * j + d}


def scale_index(row: int, head: int, j: int) -> int:
    return row * SCALE_BYTES + (head // 8) * 128 + 8 * j + (head % 8)


def scale_coord(i: int) -> dict:
    row, r = divmod(i, SCALE_BYTES)
    g, sfi = divmod(r, 128)
    return {"row": row, "head": 8 * g + (sfi & 7), "dim_group32": sfi >> 3}


def logical(data: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = data.shape[0]
    return (data.contiguous().view(torch.uint8).reshape(n, -1).to(device=CPU),
            scale.contiguous().view(torch.uint8).reshape(n, -1).to(device=CPU))


def dumped_pairs(snap: dict, n: int):
    st = (snap["stock_data"].reshape(n, -1).to(CPU), snap["stock_scale"].contiguous().view(torch.uint8).reshape(n, -1))
    w1 = (snap["w1_data"].reshape(n, -1).to(CPU), snap["w1_scale"].contiguous().view(torch.uint8).reshape(n, -1))
    return st, w1


def check_incumbent(payload: dict) -> list[str]:
    """Problems that make this file not an incumbent window-only snapshot (empty list = fine)."""
    meta, snap = payload.get("meta", {}), payload.get("snap", {})
    bad = []
    if meta.get("path") != "incumbent":
        bad.append(f"meta.path={meta.get('path')!r} (want 'incumbent'; FP8 sites go to results/E/shadow_repro)")
    if meta.get("comp_format") not in ("NONE", None):
        bad.append(f"meta.comp_format={meta.get('comp_format')!r} (want NONE: window-only)")
    if "extra" in payload.get("pages", {}) or "extra_idx" in snap:
        bad.append("snapshot carries a compressed cache (not a window-only call)")
    if meta.get("mode") not in ("ss", None) or meta.get("plan") not in ("ss", None):
        bad.append(f"mode/plan={meta.get('mode')}/{meta.get('plan')} (want ss)")
    for k in ("q", "positions", "swa_idx", "swa_lens", "stock_data", "stock_scale", "w1_data", "w1_scale"):
        if k not in snap:
            bad.append(f"snap lacks {k}")
    return bad


def remap(idx: torch.Tensor, page_ids: torch.Tensor, page_tokens: int, num_pages: int) -> tuple[torch.Tensor, dict]:
    """Global slot -> compact slot (rank of its page in page_ids) * page_tokens + offset; anything else -> -1.
    (Copied from results/E/shadow_repro.DRAFT.py.)"""
    s = idx.to(device=CPU, dtype=torch.int64)
    inrange = (s >= 0) & (s < num_pages * page_tokens)
    pg = torch.div(s.clamp(min=0), page_tokens, rounding_mode="floor")
    if page_ids.numel() == 0:
        ok = torch.zeros_like(inrange)
        rank = torch.zeros_like(s)
    else:
        pid = page_ids.to(device=CPU, dtype=torch.int64)
        rank = torch.searchsorted(pid, pg).clamp(max=pid.numel() - 1)
        ok = inrange & (pid[rank] == pg)
    new = torch.where(ok, rank * page_tokens + s.clamp(min=0) % page_tokens, torch.full_like(s, -1, device=CPU))
    info = {"nonneg": int((s >= 0).sum()), "kept": int(ok.sum()), "nonneg_dropped": int(((s >= 0) & ~ok).sum())}
    return new.to(dtype=idx.dtype), info


def build_swa(payload: dict, which: str) -> dict:
    """Compact SWA cache + remapped indices. which = pages | records. Also the coverage diagnostic: H's call-time
    records are 528-byte ROWS of the [page, page_tokens, 528] view (cache[page, off]), but the kernels read slot off
    at page bytes [off*512, +512) (data plane) and [page_tokens*512 + off*16, +16) (scale plane). A slot's read bytes
    are call-time exact only where they fall in a restored view row; the rest are dump-time bytes."""
    snap, meta = payload["snap"], payload["meta"]
    pg = payload["pages"]["swa"]
    pt, rb = int(pg["page_tokens"]), int(pg["record_bytes"])
    assert rb == ROW_DATA + ROW_SCALE, f"SWA record bytes {rb} != 528"
    pages = pg["pages"].to(device=CPU).contiguous().clone()
    if pages.shape[0] == 0:
        pages = torch.zeros((1, pt, rb), dtype=torch.uint8, device=CPU)
    idx = snap["swa_idx"].to(device=CPU)
    n = int(meta["width"])
    new, info = remap(idx, pg["page_ids"], pt, int(pg["num_pages"]))
    lens = snap["swa_lens"].to(device=CPU, dtype=torch.int64).reshape(-1)[: idx.shape[0]]
    valid = torch.arange(idx.shape[1], device=CPU).unsqueeze(0) < lens.unsqueeze(1)
    info["valid_entries"] = int(valid.sum())
    info["valid_entries_without_page"] = int((valid & (new < 0) & (idx >= 0)).sum())
    info["valid_negative_entries"] = int((valid & (idx < 0)).sum())
    layout = str(meta.get("records_layout", "view_rows"))
    info["records_layout"] = layout
    flat_rows = pages.view(-1, rb)          # [P*pt, 528] view rows
    flat_bytes = pages.view(pages.shape[0], -1)  # [P, pt*528] page bytes
    restored_rows: set = set()
    if "swa_records" in snap:
        recs = snap["swa_records"].to(device=CPU)  # [rows, width, 528] call-time bytes
        ok = new >= 0
        slots, rec_rows, vflag = new[ok].to(torch.int64), recs[ok], valid[ok]
        if layout == "view_rows":
            differ = (flat_rows[slots] != rec_rows).any(dim=1)
        else:  # "planes": records hold the slot's logical [512 data | 16 scale]
            pgi, off = torch.div(slots, pt, rounding_mode="floor"), slots % pt
            cur = torch.stack([torch.cat([flat_bytes[p, o * ROW_DATA:(o + 1) * ROW_DATA],
                                          flat_bytes[p, pt * ROW_DATA + o * ROW_SCALE:pt * ROW_DATA + (o + 1) * ROW_SCALE]])
                               for p, o in zip(pgi.tolist(), off.tolist())]) if slots.numel() else rec_rows
            differ = (cur != rec_rows).any(dim=1)
        info["referenced_records_changed_since_call"] = int(torch.unique(slots[differ]).numel())
        info["valid_referenced_records_changed_since_call"] = int(torch.unique(slots[differ & vflag]).numel())
        order = torch.argsort(slots, stable=True)
        ss_, rr = slots[order], rec_rows[order]
        info["inconsistent_duplicate_records"] = int(((ss_[1:] == ss_[:-1]) & (rr[1:] != rr[:-1]).any(dim=1)).sum())
        if which == "records":
            if layout == "view_rows":
                flat_rows[slots] = rec_rows
                restored_rows = set(slots.tolist())
            else:
                for (p, o), r in zip(zip(pgi.tolist(), off.tolist()), rec_rows):
                    flat_bytes[p, o * ROW_DATA:(o + 1) * ROW_DATA] = r[:ROW_DATA]
                    flat_bytes[p, pt * ROW_DATA + o * ROW_SCALE:pt * ROW_DATA + (o + 1) * ROW_SCALE] = r[ROW_DATA:]
    elif which == "records":
        raise SystemExit("records mode needs a snapshot taken with shadow_records=true")
    # coverage: per output row, valid slots whose read bytes are not all call-time exact
    unc = [0] * n
    if which == "records" and layout == "view_rows":
        for r in range(min(n, new.shape[0])):
            for j in range(new.shape[1]):
                if not bool(valid[r, j]) or int(new[r, j]) < 0:
                    continue
                g = int(new[r, j]); p, o = divmod(g, pt)
                spans = ((o * ROW_DATA, (o + 1) * ROW_DATA), (pt * ROW_DATA + o * ROW_SCALE, pt * ROW_DATA + (o + 1) * ROW_SCALE))
                rows_needed = {p * pt + k for a, e in spans for k in range(a // rb, (e - 1) // rb + 1)}
                if not rows_needed <= restored_rows:
                    unc[r] += 1
    elif which == "pages":
        unc = [int(valid[r].sum()) if r < valid.shape[0] else 0 for r in range(n)]  # every slot is dump-time
    info["valid_slots_not_call_time_exact_per_row"] = unc
    return {"cache": pages, "idx": new, "lens": snap["swa_lens"].to(device=CPU), "info": info, "page_tokens": pt}


def positions_for(payload: dict) -> tuple[torch.Tensor, list[int]]:
    meta, snap = payload["meta"], payload["snap"]
    base = int(meta["cos_sin_base"])
    nrows = int(payload["cos_sin_rows"].shape[0])
    pos = snap["positions"].to(device=CPU, dtype=torch.int64) - base
    bad = torch.nonzero((pos < 0) | (pos >= nrows)).flatten().tolist()
    return pos.clamp(0, nrows - 1).to(dtype=snap["positions"].dtype), bad


def rows_equal(a: tuple, b: tuple) -> list[bool]:
    return [bool(torch.equal(a[0][r], b[0][r]) and torch.equal(a[1][r], b[1][r])) for r in range(a[0].shape[0])]


def diff_positions(a: tuple, b: tuple, rows: list[int]) -> list[tuple[str, int]]:
    """Positions (kind, flat index) where a and b differ, restricted to the given rows."""
    out = []
    for r in rows:
        for kind, k, width in (("data", 0, TOKEN_BYTES), ("scale", 1, SCALE_BYTES)):
            d = torch.nonzero(a[k][r] != b[k][r]).flatten().tolist()
            out += [(kind, r * width + i) for i in d]
    return out


def byte_at(pair: tuple, kind: str, i: int) -> int:
    k, width = (0, TOKEN_BYTES) if kind == "data" else (1, SCALE_BYTES)
    return int(pair[k].reshape(-1)[i])


def decode_pos(kind: str, i: int, pairs: dict) -> dict:
    """One differing byte: coordinates, the code of every named output there, and its scale byte(s)."""
    if kind == "data":
        c = data_coord(i)
        si = scale_index(c["row"], c["head"], c["dim_group32"])
        rec = dict(kind=kind, index=i, **c, scale_index=si)
        for name, pr in pairs.items():
            code = byte_at(pr, "data", i)
            sb = byte_at(pr, "scale", si)
            rec[name] = {"code": code, "e4m3": e4m3_value(code), "scale_byte": sb,
                         "value": e4m3_value(code) * 2.0 ** (sb - 127)}
    else:
        c = scale_coord(i)
        rec = dict(kind=kind, index=i, **c)
        for name, pr in pairs.items():
            rec[name] = {"scale_byte": byte_at(pr, "scale", i)}
    return rec


def fmt_pos(d: dict, names: list[str]) -> str:
    if d["kind"] == "data":
        codes = " ".join(f"{n}={d[n]['code']}({d[n]['e4m3']:g}) sc{d[n]['scale_byte']}" for n in names if n in d)
        return f"data[{d['index']}] row={d['row']} head={d['head']} dim={d['dim']} grp32={d['dim_group32']} {codes}"
    codes = " ".join(f"{n}={d[n]['scale_byte']}" for n in names if n in d)
    return f"scale[{d['index']}] row={d['row']} head={d['head']} grp32={d['dim_group32']} {codes}"


def judge(stock: tuple, shipped: tuple, variant: tuple, rows: list[int]) -> tuple[str, dict]:
    """Verdict of `variant` over `rows` (the reproducible rows), given the rebuilt stock and shipped outputs."""
    targets = diff_positions(stock, shipped, rows)
    var_vs_stock = set(diff_positions(stock, variant, rows))
    var_vs_ship = diff_positions(shipped, variant, rows)
    tset = set(targets)
    new = sorted(var_vs_stock - tset)            # shipped == stock there, variant != stock
    fixed = sorted(tset - var_vs_stock)           # target now equal to stock
    detail = {"rows": rows, "targets": targets, "fixed": fixed, "new_mismatches": new[:50],
              "n_new_mismatches": len(new), "n_var_vs_shipped": len(var_vs_ship)}
    if not rows:
        return "INCONCLUSIVE (no reproducible row)", detail
    if not targets:
        return "INCONCLUSIVE (no target byte in the reproducible rows)", detail
    if new:
        return "WORSE", detail
    if len(fixed) == len(targets):
        return "CONFIRMS", detail
    if not var_vs_ship:
        return "NO-CHANGE", detail
    return f"INCONCLUSIVE (partial: {len(fixed)} of {len(targets)} target bytes fixed, no new mismatch)", detail


# ---------------------------------------------------------------------------------------------------------------
# GPU worker (one .so per process)
# ---------------------------------------------------------------------------------------------------------------
def make_out(shape, stride, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    span = 1 + sum((a - 1) * b for a, b in zip(shape, stride))
    item = torch.empty((), dtype=dtype, device=CPU).element_size()
    raw = torch.full((span * item + GUARD_BYTES,), SENTINEL, dtype=torch.uint8, device=device)
    return raw, torch.as_strided(raw[: span * item].view(dtype), tuple(shape), tuple(stride), 0)


def load_ops(ops_path: Path, so: Path):
    spec = importlib.util.spec_from_file_location("pg61_w1attn_ops", ops_path)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    ops._mod = ops._load_prebuilt(str(so))  # pg61 ops: one module-level extension, name w1_attn_ext
    assert ops._ext() is ops._mod and Path(ops._mod.__file__).resolve() == so.resolve(), "wrong extension loaded"
    return ops


def worker(site: Path, so: Path, ops_path: Path, modes: list[str], layout_k: int, raw_out: Path) -> int:
    payload = torch.load(site, map_location=CPU, weights_only=True)
    meta, snap = payload["meta"], payload["snap"]
    n = int(meta["width"])
    device = torch.device("cuda", 0)
    module = importlib.import_module(TARGET_MODULE)
    ops = load_ops(ops_path, so)
    ops._ext().w1_attn_prepare()  # == integrate.prepare(): smem opt-in, then the zeroed tagged scratch
    words = MAX_SQ * MAX_BLOCKS * (128 + 128 + 2048 + 1)
    scratch = ops.Scratch(part_o=torch.zeros(2 * words, dtype=torch.float32, device=device),
                          part_ml=torch.zeros(1, dtype=torch.float32, device=device),
                          flags=torch.zeros(1, dtype=torch.int32, device=device),
                          epoch=torch.zeros(2, dtype=torch.int32, device=device))
    dshape, dstride, sshape, sstride = meta["layouts"][layout_k]
    pos, _ = positions_for(payload)
    positions = pos.to(device)
    cos_sin = payload["cos_sin_rows"].to(device)
    q = snap["q"].to(device)
    sink = payload["attn_sink"].to(device)
    sm_scale = float(meta["scale"])
    n_wv = int(meta["n_wv_group"])
    res: dict[str, Any] = {"so": str(so), "so_sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
                           "ext_file": str(ops._mod.__file__), "torch": str(torch.__version__), "modes": {}}
    for mode in modes:
        sw = build_swa(payload, mode)
        swa_cache, swa_idx, swa_lens = sw["cache"].to(device), sw["idx"].to(device), sw["lens"].to(device)

        def stock(data, scale):  # == integrate._stock for compress_ratio 0: extra cache / idx / len are None
            torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
                q, swa_cache.unsqueeze(-2), swa_idx, sm_scale, module._HEAD_DIM_V, sink, swa_lens, None, None, None,
                False, 0.0, positions, *module._ROPE_ARGS, cos_sin, n_wv, *module._SF_ARGS, data, scale)

        def w1(data, scale):  # == integrate.py:193-195 for a window-only layer (mode "ss", pdl default True)
            ops.attn_decode(q, positions, cos_sin, sink, sm_scale, swa_cache, swa_idx, swa_lens, None, None, None,
                            data, scale, scratch, "ss")

        entry = {}
        for name, fn in (("stock", stock), ("w1", w1)):
            outs = []
            for _ in range(2):
                _, data = make_out(dshape, dstride, torch.float8_e4m3fn, device)
                _, scale = make_out(sshape, sstride, torch.int32, device)
                fn(data, scale)
                torch.cuda.synchronize()
                outs.append(logical(data, scale))
            entry[name] = outs
        res["modes"][mode] = entry
    torch.save(res, raw_out)
    print(f"WORKER {so} -> {raw_out}", flush=True)
    return 0


def run_worker_subprocess(site: Path, so: Path, ops_path: Path, modes: list[str], layout_k: int, raw_out: Path) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), str(site), "--worker", "--worker-so", str(so), "--ops",
           str(ops_path), "--modes", ",".join(modes), "--layout", str(layout_k), "--raw-out", str(raw_out)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write("".join("  | " + ln + "\n" for ln in r.stdout.splitlines()[-30:]))
    if r.returncode != 0:
        raise SystemExit(f"worker for {so} failed rc={r.returncode}")
    return torch.load(raw_out, map_location=CPU, weights_only=True)


# ---------------------------------------------------------------------------------------------------------------
# parent: rebuild diagnostics, workers, verdicts
# ---------------------------------------------------------------------------------------------------------------
def analyse(payload: dict, raws: dict, shipped_name: str, variants: list[str], modes: list[str]) -> dict:
    """raws: name -> worker result. Returns the report; prints the human lines."""
    meta, snap = payload["meta"], payload["snap"]
    n = int(meta["width"])
    dumped_st, dumped_w1 = dumped_pairs(snap, n)
    lens = snap["swa_lens"].to(CPU, torch.int64).reshape(-1)[:n]
    valid = [bool(lens[r] > 0) for r in range(n)]
    _, bad_pos = positions_for(payload)
    rep: dict[str, Any] = {"modes": {}}
    primary = None
    for mode in modes:
        m: dict[str, Any] = {}
        st_runs = {name: raws[name]["modes"][mode]["stock"] for name in raws}
        st = st_runs[shipped_name][0]
        stock_det = all(rows_equal(r[0], r[1]) == [True] * n for r in st_runs.values())
        stock_same_proc = all(rows_equal(st, r[0]) == [True] * n for r in st_runs.values())
        w = {name: raws[name]["modes"][mode]["w1"] for name in raws}
        det = {name: rows_equal(w[name][0], w[name][1]) == [True] * n for name in w}
        sh = w[shipped_name][0]
        st_rep = rows_equal(st, dumped_st)
        sh_rep = rows_equal(sh, dumped_w1)
        rows, excluded = [], {}
        for r in range(n):
            why = []
            if not valid[r]:
                why.append("invalid row (swa_lens <= 0)")
            if r in bad_pos:
                why.append("position outside the saved cos/sin rows")
            if not st_rep[r]:
                why.append("stock rebuild != dumped stock")
            if not sh_rep[r]:
                why.append("shipped rebuild != dumped incumbent")
            if why:
                excluded[r] = why
            else:
                rows.append(r)
        print(f"[{mode}] STOCK reproduces dumped stock: rows {sum(st_rep)}/{n} {st_rep} "
              f"(deterministic={stock_det}, identical across processes={stock_same_proc})", flush=True)
        print(f"[{mode}] SHIPPED reproduces dumped incumbent: rows {sum(sh_rep)}/{n} {sh_rep} "
              f"(deterministic={det[shipped_name]})", flush=True)
        print(f"[{mode}] reproducible rows {rows}; excluded {excluded or 'none'}", flush=True)
        pairs = {"stock": st, shipped_name: sh, **{v: w[v][0] for v in variants}}
        all_rows = list(range(n))
        tg = [decode_pos(k, i, pairs) for k, i in diff_positions(st, sh, all_rows)]
        for d in tg[:40]:
            tag = "" if d["row"] in rows else "  [row excluded]"
            print(f"[{mode}] STOCK vs SHIPPED: {fmt_pos(d, ['stock', shipped_name] + variants)}{tag}", flush=True)
        if not tg:
            print(f"[{mode}] STOCK vs SHIPPED: identical on every row", flush=True)
        m.update(stock_reproduces_rows=st_rep, shipped_reproduces_rows=sh_rep, reproducible_rows=rows,
                 excluded_rows=excluded, stock_deterministic=stock_det, stock_identical_across_processes=stock_same_proc,
                 stock_vs_shipped=tg[:200], n_stock_vs_shipped=len(tg), per_build={})
        for v in variants:
            same_all = rows_equal(w[v][0], sh) == [True] * n
            verdict, detail = judge(st, sh, w[v][0], rows)
            if not det[v] or not det[shipped_name]:
                verdict = "INCONCLUSIVE (non-deterministic W1 runs)"
            elif not stock_det or not stock_same_proc:
                verdict = "INCONCLUSIVE (stock not reproducible across runs/processes)"
            new = [decode_pos(k, i, pairs) for k, i in detail["new_mismatches"][:10]]
            print(f"[{mode}] BUILD {v}: sha {raws[v]['so_sha256'][:16]} identical_to_shipped_all_rows={same_all} "
                  f"targets={len(detail['targets'])} fixed={len(detail['fixed'])} new_mismatches={detail['n_new_mismatches']} "
                  f"bytes_vs_shipped={detail['n_var_vs_shipped']} -> {verdict}", flush=True)
            for d in new:
                print(f"[{mode}]   new mismatch: {fmt_pos(d, ['stock', shipped_name, v])}", flush=True)
            m["per_build"][v] = {"sha256": raws[v]["so_sha256"], "deterministic": det[v],
                                 "identical_to_shipped_all_rows": same_all, "verdict": verdict,
                                 "targets": detail["targets"], "fixed": detail["fixed"],
                                 "new_mismatches": new, "n_new_mismatches": detail["n_new_mismatches"],
                                 "bytes_vs_shipped": detail["n_var_vs_shipped"]}
        rep["modes"][mode] = m
        if primary is None and any(d["row"] in rows for d in tg):
            primary = mode
    rep["verdict_mode"] = primary
    rep["verdicts"] = {}
    for v in variants:
        if primary is None:
            line = "INCONCLUSIVE (no rebuild mode has a reproducible row with a target byte)"
        else:
            line = rep["modes"][primary]["per_build"][v]["verdict"]
        rep["verdicts"][v] = line
        print(f"VERDICT {v} {line}" + (f"  [rebuild={primary}]" if primary else ""), flush=True)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", type=Path)
    ap.add_argument("--ops", type=Path, help="Pg61 fastpath/w1attn/ops.py (the served call contract)")
    ap.add_argument("--shipped", type=Path, help="the shipped incumbent w1_attn_ext.so (sha 7201643d...)")
    ap.add_argument("--so", action="append", default=[], metavar="NAME=PATH", help="build under test (repeatable)")
    ap.add_argument("--modes", default="records,pages")
    ap.add_argument("--layout", type=int, default=0, help="index into meta.layouts for the out buffers")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-so", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--raw-out", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    modes = [m for m in args.modes.split(",") if m]
    if args.worker:
        return worker(args.site, args.worker_so, args.ops, modes, args.layout, args.raw_out)

    payload = torch.load(args.site, map_location=CPU, weights_only=True)
    meta, snap = payload["meta"], payload["snap"]
    n = int(meta["width"])
    problems = check_incumbent(payload)
    print(f"SITE {args.site.name}: prefix={meta.get('prefix')} layer={meta.get('layer')} class={meta.get('layer_class')} "
          f"width={n} mode={meta.get('mode')} plan={meta.get('plan')} pdl={meta.get('pdl')} "
          f"first_call={meta.get('first_call')} first_seq={meta.get('first_seq')} calls={meta.get('calls')} "
          f"records={meta.get('records_in_snapshot')} layouts={meta.get('layouts')}", flush=True)
    if problems:
        print("NOT AN INCUMBENT WINDOW-ONLY SNAPSHOT: " + "; ".join(problems), flush=True)
        return 2
    print(f"  swa_lens={snap['swa_lens'].reshape(-1).tolist()} positions={snap['positions'].reshape(-1).tolist()} "
          f"swa_idx={tuple(snap['swa_idx'].shape)}", flush=True)
    dumped_st, dumped_w1 = dumped_pairs(snap, n)
    dd = [decode_pos(k, i, {"stock": dumped_st, "incumbent": dumped_w1})
          for k, i in diff_positions(dumped_st, dumped_w1, list(range(n)))]
    for d in dd[:40]:
        print("DUMPED stock vs incumbent (in-graph): " + fmt_pos(d, ["stock", "incumbent"]), flush=True)
    print(f"DUMPED differing bytes: {len(dd)} (data {sum(d['kind'] == 'data' for d in dd)}, "
          f"scale {sum(d['kind'] == 'scale' for d in dd)})", flush=True)
    if len(meta.get("layouts", [])) > 1:
        print(f"note: site saw {len(meta['layouts'])} out layouts; out buffers use layouts[{args.layout}] "
              f"(values are compared in logical order)", flush=True)
    _, bad_pos = positions_for(payload)
    if bad_pos:
        print(f"note: rows {bad_pos} have positions outside the saved cos/sin rows (excluded)", flush=True)
    if not meta.get("records_in_snapshot"):
        modes = [m for m in modes if m != "records"]
    report: dict[str, Any] = {"site": str(args.site), "meta": meta, "dumped_stock_vs_incumbent": dd[:200],
                              "rebuild": {}}
    for mode in modes:
        info = build_swa(payload, mode)["info"]
        report["rebuild"][mode] = info
        print(f"REBUILD {mode}: {json.dumps(info)}", flush=True)
    if not args.cpu_only:
        if not (args.ops and args.shipped):
            raise SystemExit("--ops and --shipped are required unless --cpu-only")
        variants = []
        so_paths = {"shipped": args.shipped}
        for item in args.so:
            name, _, path = item.partition("=")
            if not name or not path or name in so_paths:
                raise SystemExit(f"bad --so {item!r} (NAME=PATH, unique names, 'shipped' reserved)")
            so_paths[name] = Path(path)
            variants.append(name)
        outdir = (args.out.parent if args.out else Path.cwd())
        raws = {}
        for name, path in so_paths.items():
            raw = outdir / f".{args.site.stem}.{name}.raw.pt"
            raws[name] = run_worker_subprocess(args.site, path, args.ops, modes, args.layout, raw)
        report.update(analyse(payload, raws, "shipped", variants, modes))
        report["builds"] = {k: {"path": str(v), "sha256": raws[k]["so_sha256"]} for k, v in so_paths.items()}
    out = args.out or Path(f"I-probe-{args.site.stem}.json")
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

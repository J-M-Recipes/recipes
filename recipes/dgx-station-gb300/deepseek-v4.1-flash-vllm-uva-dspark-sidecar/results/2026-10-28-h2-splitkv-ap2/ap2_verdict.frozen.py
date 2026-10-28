#!/usr/bin/env python3
"""ap2_verdict.py <R> -- Card AP2 verdict: split-KV (side extension, quarantine) on the GUARDED h1-REF. DRAFT 2026-10-28
(claude-opus-5-5, from ap_verdict.py d27cff85 with the AP lessons folded in).

PRE-REGISTERED. cardAP2_runner.sh pins this file's sha256, copies it to $R/ap2_verdict.frozen.py at preflight (before
boot A), logs the hash, and runs only that frozen copy at END. Missing data is UNKNOWN, never a number.
Arms: controls A, A2 (h1-REF launch + NOCORE, ATT2B unset); candidates S8, S4 (ATT2B side extension _flashmla_splitkv_C,
SPLITKV 8 / 4, MAX_TOKENS 8, MAX_REQS 1, SQ 6, TRACE 1, GUARD 0, COREDUMP 0, NOCORE 1). Every arm serves the #60945 guard
(desk-hook/L3 rejection_sampler_utils.py sha256 2d43d513...), checked in-container.

Gate classes (decide the candidate result):
  VALID  G0 (per arm): KV 4,400,013; vllm args == h1-REF; env delta == exactly the expected set (controls: the 2 NOCORE vars;
         S: + the 10 ATT2B quarantine vars); binds == h1-REF (S: + exactly the 3 frozen h2ref-att2b :ro mounts, nothing over
         the shipped _flashmla_C); --ulimit core=0; in-container guard rsu sha256 == 2d43d513; MEGA_L2 lines; 0 foreign POSTs;
         0 sidecar timeouts. Engagement S: 'split-KV ENABLED: cap=N' + max_reqs=1, >=1 's_q=6 splits=N kind=compressed
         op=_flashmla_splitkv_C', no other split-decode line, >=1 TRACE ok=1 at (s_q 6, num_tokens 6, 1 decode, 0 prefills),
         no other ok=1 shape, 0 guard lines, shipped .so df2cdd1c + side .so 744a9de8 in-container, side .so mapped.
         Controls: no ATT2/CRASH2 line, side .so not mapped. Controls must ALSO be CRASH-FREE (else INVALID-CONTROL).
         G1: A vs A2 BRRR step spread <= 1.0%, else INVALID-CONTROL.
  CRASH  G_crashfree (NEW, the gate AP failed): 0 new Xid lines in the kernel log from boot to END of the arm, 0 engine
         errors (server + sidecar, runner's FF regex, from cells start), 0 GPU-fault lines in the arm's full log, no fault
         cell, alive at END, 0 card-watchdog trips, booted, and no lib sub-cell failed with an engine/HTTP-5xx/drain reason.
  SPEED  G_brrr: BRRR 2K T=0 tok/s >= 1.025 x max(A, A2).
  CORR   G_tf: W(S vs A) <= 1.25 x W(A2 vs A) AND ref-confident flips(S vs A) <= max(confident flips(A2 vs A), 2) + 1
                AND agent/tool ref-confident flips == 0 (Milo's AP tightening, kept).
         G_gauntlet: 12/12.
         G_probe62417: >= 16 cells; every corruption cell (P1-P4) ok; P5 synthetic arithmetic (40 seeded questions,
                identical on every arm) >= min(0.93, min(P5 acc A, A2) - 0.05). AP's two controls scored 0.90/0.85 on the
                absolute 0.93 bar (thinking at effort high: the probe also fails >6000-char reasoning), so the absolute bar
                false-fails healthy arms; #62417 corruption reads ~0.02. A failing cell that also failed on a control =
                CONTROL-ALSO (judged HOLD).
         G_soak: 0 loops AND every finish_reason == stop AND correct >= min(correct A, correct A2); a control that also
                looped/failed to stop -> CONTROL-ALSO (HOLD).
         G_1M: needle returned + finished + 850K..1,048K prompt tokens (own onem.json). An S arm without its own smoke
               (runner policy: S4 runs it only when it out-BRRRs S8 or S8 lacks a pass) INHERITS S8's PASS only if S8 was
               CRASH-FREE; promote_guard_split.sh re-runs the smoke on h2-REF anyway.
  HOLD   G_perclass: per class, C1 and C8 (mean of r1, r2) each >= min(A, A2) x (1 - max(2%, 1.5 x A/A2 spread)).
         G_lib: C1/C8/C32 at 0 and 64K each >= 0.95 x min(A, A2).
         G_prefill: per context (8K/16K/64K/128K, mean of 2 reps): tok/s >= 0.95 x worse control AND TTFT <= 1.05 x worse.
Result per candidate: INVALID-CONTROL > NO (CRASH-FREE FAIL, even if engagement is also off) > INVALID-NOTENGAGED /
INVALID > NO (any SPEED/CORR FAIL) > INCOMPLETE (CRASH/SPEED/CORR UNKNOWN) > HOLD (any HOLD-class FAIL or CONTROL-ALSO)
> INCOMPLETE (HOLD-class UNKNOWN) > PROMOTE-READY. Arms that never booted: SKIPPED (budget/disk/control) or MISSING; a
candidate whose boot died after `docker run` is judged by its crash evidence (NO if any, else INCOMPLETE).
Winner: the PROMOTE-READY candidate with the higher BRRR tok/s; within 0.5% of each other -> S8.
No promotion here: promote_guard_split.sh reads `winner` and refuses anything not PROMOTE-READY (HOLD only with FORCE).
Output: verdict.json (recipe-factory shape: result, suffix, gates{name:[status,text]}, per_candidate, cells) + stdout.
"""
import hashlib
import json
import os
import re
import statistics as st
import sys

R = sys.argv[1]
CTRL = ["A", "A2"]
CANDS = ["S8", "S4"]
NSPLIT = {"S8": 8, "S4": 4}
ARMS = CTRL + CANDS
CLASSES = ["prose", "structured", "code", "shell_ops", "tool_json"]
LIB_KEYS = ["C1@0", "C8@0", "C32@0", "C1@65536", "C8@65536", "C32@65536"]
TTFT_CTX = ["8192", "16384", "65536", "131072"]
TTFT_REPS = 2
SHIPPED_SHA256 = "df2cdd1c3acfe6546b10a8d6d9b3d8bae3783a4910337d9721d83ff653e4fc13"
SIDE_SHA256 = "744a9de89060730eecbb7c4c249dedc36f3dbfb3854de17299780b66cb90e983"
RSU_SHA256 = "2d43d5133c1fb96399e186ed10c5a26b54c12e6a019d9d163cbff1adbcfbcc39"   # #60945 guard (production since 14:42)
P5_ABS, P5_SLACK, PROBE_MIN_CELLS = 0.93, 0.05, 16
LIB_CRASH_RE = re.compile(r"HTTP 5\d\d|EngineCore|InternalServerError|did not drain|Connection (refused|reset)|"
                          r"metrics are unavailable|RemoteDisconnected|timed out", re.I)
GCLASS = {"G_crashfree": "CRASH", "G_brrr": "SPEED", "G_tf": "CORR", "G_gauntlet": "CORR", "G_probe62417": "CORR",
          "G_soak": "CORR", "G_1M": "CORR", "G_perclass": "HOLD", "G_lib": "HOLD", "G_prefill": "HOLD"}
gates, cells = {}, {}


def jl(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def first(p):
    try:
        s = open(p).read().strip().split()
        return s[0] if s else "?"
    except Exception:  # noqa: BLE001
        return None


def txt(p):
    try:
        return open(p).read()
    except Exception:  # noqa: BLE001
        return ""


def nlines(p):
    try:
        return sum(1 for ln in open(p) if ln.strip())
    except Exception:  # noqa: BLE001
        return None


def ival(p):
    v = first(p)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


ran = {a: os.path.exists(f"{R}/{a}/facts.json") for a in ARMS}
booted_dead = {a: os.path.exists(f"{R}/{a}/boot-failed.txt") for a in ARMS}
skipped = {a: txt(f"{R}/{a}/SKIPPED").strip() or None for a in ARMS}
cells["ran"], cells["skipped"], cells["boot_failed"] = ran, skipped, booted_dead


# ---------------------------------------------------------------- lib (needed by CRASH-FREE too)
def lib(a):
    d = jl(f"{R}/{a}/lib.json")
    if not d:
        return None
    out, crash, seen = {}, [], set()
    for x in d.get("results", []):
        key = f"C{x.get('concurrency')}@{x.get('context_tokens')}"
        seen.add(key)
        fr = x.get("failure_reason") or ""
        if fr and LIB_CRASH_RE.search(fr):
            crash.append(f"{key}: {fr[:120]}")
        if fr or x.get("loop_detected") or (x.get("aggregate_tps") or 0) <= 0:
            continue
        out[key] = round(x["aggregate_tps"], 1)
    out["_crash"] = crash
    out["_missing"] = [k for k in LIB_KEYS if k not in seen]
    out["_prefill_scouts"] = {k: {"tok_s": v.get("tok_per_sec"), "ttft_s": v.get("ttft_seconds")}
                              for k, v in (d.get("prefill") or {}).items()}
    return out


LIB = {a: lib(a) for a in ARMS}
cells["lib"] = LIB


# ---------------------------------------------------------------- CRASH-FREE (all arms)
def crashfree(a):
    if not (ran[a] or booted_dead[a]):
        return ["SKIP", "arm did not boot"]
    d = f"{R}/{a}"
    xid = nlines(f"{d}/xid-new.txt")
    err = [ival(f"{R}/errcount-{a}.txt"), ival(f"{R}/errcount-{a}-rtx.txt")]
    faults = ival(f"{d}/faults.txt")
    fcell = txt(f"{d}/fault-cell.txt").strip()
    alive = first(f"{d}/alive-end.txt")
    trips = ival(f"{d}/wd-trips.txt")
    lc = (LIB[a] or {}).get("_crash", []) if LIB[a] else []
    bad, unk = [], []
    if booted_dead[a]:
        bad.append("boot died after docker run: " + txt(f"{d}/boot-failed.txt").strip()[:160])
    if xid is None:
        unk.append("xid-new.txt")
    elif xid > 0:
        bad.append(f"{xid} new Xid line(s): " + " | ".join(ln[:140] for ln in txt(f"{d}/xid-new.txt").splitlines()[:2]))
    for tag, v in zip(("server", "sidecar"), err):
        if v is None:
            if not booted_dead[a]:
                unk.append(f"errcount {tag}")
        elif v > 0:
            bad.append(f"{v} engine error line(s) ({tag})")
    if faults is None:
        unk.append("faults.txt")
    elif faults > 0:
        bad.append(f"{faults} GPU-fault line(s) in the arm log")
    if fcell:
        bad.append(f"fault watcher stopped cell: {fcell[:120]}")
    if alive is None:
        if not booted_dead[a]:
            unk.append("alive-end.txt")
    elif alive != "1":
        bad.append("arm not alive at END")
    if trips is None:
        if not booted_dead[a]:
            unk.append("wd-trips.txt")
    elif trips > 0:
        bad.append(f"{trips} card-watchdog TRIP(s)")
    if lc:
        bad.append(f"lib engine failures {lc}")
    t = f"xid={xid} errors={err} gpu_fault_lines={faults} fault_cell=[{fcell[:60]}] alive_end={alive} wd_trips={trips} lib_crash={len(lc)}"
    if bad:
        return ["FAIL", "; ".join(bad) + f" || {t}"]
    if unk:
        return ["UNKNOWN", f"missing {unk} || {t}"]
    return ["PASS", t]


CF = {a: crashfree(a) for a in ARMS}
cells["crashfree"] = CF

# ---------------------------------------------------------------- G0 validity + engagement
valid, engaged = {}, {}
for a in ARMS:
    if not ran[a]:
        valid[a] = None
        continue
    f = jl(f"{R}/{a}/facts.json") or {}
    e = jl(f"{R}/{a}/engage.json") or {}
    fore = [first(f"{R}/{a}/foreign-{t}.txt") for t in ("bind", "cells")]
    wd = jl(f"{R}/{a}/wd-status.json") or {}
    to = (wd.get("words") or [None, None, None])[2]
    at = f.get("autotune", "")
    hash_hit = bool(re.search(r"Loaded (\d+) configs", at)) and f.get("boot_s", 9999) <= 1.5 * 550
    ok = (f.get("kv") == "4,400,013" and f.get("args_ok") == 1 and f.get("env_ok") == 1 and f.get("binds_ok") == 1
          and f.get("ulimit_core0") == 1 and f.get("rsu_sha256") == RSU_SHA256 and f.get("l2_lines", 0) > 0
          and fore == ["0", "0"] and to == 0)
    if a in CANDS:
        n = NSPLIT[a]
        engaged[a] = (e.get("enabled", 0) > 0 and e.get("enabled_cap") == [n] and e.get("enabled_max_reqs") == [1]
                      and e.get("sqN_side", 0) > 0 and e.get("decode_other", 1) == 0
                      and e.get("trace_ok1_sq6", 0) > 0 and e.get("trace_ok1_bad", 1) == 0 and e.get("guard_lines", 1) == 0
                      and e.get("shipped_sha256") == SHIPPED_SHA256 and e.get("side_sha256") == SIDE_SHA256
                      and (e.get("side_mapped_pids") or 0) > 0 and e.get("rsu_sha256_end") in (None, RSU_SHA256))
    else:
        engaged[a] = (e.get("att2_any", 1) == 0 and (e.get("side_mapped_pids") or 0) == 0
                      and e.get("rsu_sha256_end") in (None, RSU_SHA256))
    valid[a] = bool(ok and engaged[a])
    cells[f"facts_{a}"] = {**f, "foreign": fore, "sidecar_timeouts": to, "hash_hit_shape": hash_hit,
                           "engage": e, "engaged": engaged[a], "valid_facts": ok}
ctrl_crash = {a: CF[a][0] for a in CTRL}
ctrl_ok = bool(valid.get("A") and valid.get("A2") and all(v == "PASS" for v in ctrl_crash.values()))
gates["G0_validity"] = [("PASS" if all(v for v in valid.values() if v is not None) else "PARTIAL") if ctrl_ok else
                        "INVALID-CONTROL",
                        " ".join(f"{a}={'skip' if v is None else ('ok' if v else 'BAD')}" for a, v in valid.items())
                        + f" | controls CRASH-FREE A={ctrl_crash['A']} A2={ctrl_crash['A2']}"]


# ---------------------------------------------------------------- BRRR 2K T=0 (headline speed)
def brrr(a):
    d = jl(f"{R}/{a}/c1-brrr-t0.json")
    if not d:
        return None
    r = [x for x in d.get("rows", []) if "error" not in x and x.get("size") == 2048 and x.get("step_ms_from_metrics")]
    if not r:
        return None
    return {"n": len(r), "step_ms": round(st.median(x["step_ms_from_metrics"] for x in r), 4),
            "tok_step": round(st.mean(x["tok_per_step_metrics"] for x in r), 3),
            "tok_s": round(st.mean(x["tg_tok_s"] for x in r), 1)}


B = {a: brrr(a) for a in ARMS}
cells["brrr2k_t0"] = B
if ctrl_ok and B["A"] and B["A2"]:
    sa, sb = B["A"]["step_ms"], B["A2"]["step_ms"]
    spread = abs(sa - sb) / ((sa + sb) / 2)
    gates["G1_controls"] = ["PASS" if spread <= 0.01 else "INVALID-CONTROL",
                            f"A {sa:.3f} / A2 {sb:.3f} ms/step, spread {100 * spread:.2f}% (bar <= 1.0%)"]
else:
    gates["G1_controls"] = ["UNKNOWN", f"controls valid+crash-free={ctrl_ok} brrr A={B['A']} A2={B['A2']}"]


# ---------------------------------------------------------------- readers
def perclass(a):
    out = {}
    for m in ("c1", "c8"):
        for c in CLASSES:
            v = []
            for r in ("r1", "r2"):
                d = jl(f"{R}/{a}/perclass-t1/perclass-AP2-{a}-t1-{r}-{m}.json")
                if d and c in d.get("res", {}):
                    v.append(d["res"][c]["decode_tok_s_pooled"])
            out[f"{c}_{m}"] = round(st.mean(v), 2) if v else None
            out[f"{c}_{m}_n"] = len(v)
    return out


PC = {a: perclass(a) for a in ARMS}
cells["perclass_t1"] = PC


def ttft(a):
    d = jl(f"{R}/{a}/ttft.json")
    if not d:
        return None
    return {c: d.get("summary", {}).get(c) for c in TTFT_CTX}


TT = {a: ttft(a) for a in ARMS}
cells["ttft"] = TT
GA = {}
for a in ARMS:
    s = jl(f"{R}/{a}/gauntlet/summary.json") or {}
    GA[a] = f"{s.get('success_count')}/{s.get('task_count')}"
cells["gauntlet"] = GA


def flipsm(c, ref="A"):
    t = txt(f"{R}/flipsm-{c}-vs-{ref}.txt")
    m = re.search(r"GATE (PASS|FAIL) hi_agent_tool=(\d+) hi_prose_pct=([\d.]+) W=([\d.]+) W_calib=([\d.]+)", t)
    hi = sum(int(x) for x in re.findall(r"ref-confident n=\s*\d+ flips=\s*(\d+)", t))
    if not m:
        return None
    return {"gate": m.group(1), "hi_agent_tool": int(m.group(2)), "hi_prose_pct": float(m.group(3)),
            "W": float(m.group(4)), "W_calib": float(m.group(5)), "confident_flips_all": hi}


def tfcmp(c):
    t = txt(f"{R}/tfcmp-{c}-vs-A.log")
    m = re.search(r"top-1 flips=(\d+) \(([\d.]+)%\)", t)
    return {"flips": int(m.group(1)), "pct": float(m.group(2))} if m else None


FM = {c: flipsm(c) for c in ["A2"] + CANDS}
TFC = {c: tfcmp(c) for c in ["A2"] + CANDS}
cells["flips_margin_vs_A"], cells["tf_compare_vs_A"] = FM, TFC


def greedy_eq(a, b):
    ga, gb = jl(f"{R}/{a}/greedy.json"), jl(f"{R}/{b}/greedy.json")
    if not ga or not gb:
        return None
    same, n = 0, 0
    for pid, ra in ga["res"].items():
        rb = gb["res"].get(pid)
        if not rb or "error" in ra or "error" in rb:
            continue
        n += 1
        same += int(ra["text"] == rb["text"] and ra.get("tool_calls") == rb.get("tool_calls"))
    return {"n": n, "identical": same}


cells["greedy_vs_A"] = {c: greedy_eq("A", c) for c in ["A2"] + CANDS}   # record only (lane is not bit-reproducible)
PR = {a: jl(f"{R}/{a}/probe62417.json") for a in ARMS}
SK = {a: jl(f"{R}/{a}/soak.json") for a in ARMS}
OM = {a: jl(f"{R}/{a}/onem.json") for a in ARMS}


def p5(d):
    for x in (d or {}).get("results", []):
        m = re.match(r"P5 synthetic arithmetic acc=([\d.]+) \((\d+)/(\d+)\)", x.get("probe", ""))
        if m:
            return {"acc": float(m.group(1)), "ok": int(m.group(2)), "n": int(m.group(3))}
    return None


def probe_fails(d):
    return [x["probe"] for x in (d or {}).get("results", []) if not x.get("ok") and not x["probe"].startswith("P5 ")]


cells["probe62417"] = {a: (None if not d else {"fails": d.get("fails"), "cells": len(d.get("results", [])),
                                                "failed": [x["probe"] for x in d.get("results", []) if not x.get("ok")],
                                                "p5": p5(d)})
                       for a, d in PR.items()}
cells["soak"] = {a: (None if not d else {k: d.get(k) for k in ("n", "n_pass", "n_correct", "n_loop", "n_nonstop")} |
                     {"completion_tokens": {p: r.get("completion_tokens") for p, r in d.get("res", {}).items()}})
                 for a, d in SK.items()}
cells["onem"] = {a: (None if not d else {"ok": d.get("ok"), "why": d.get("why"),
                                         **{k: (d.get("result") or {}).get(k) for k in ("prompt_tokens", "ttft_s", "wall_s", "finish")}})
                 for a, d in OM.items()}
cells["onem_skipped"] = {a: txt(f"{R}/{a}/onem.SKIPPED").strip() or None for a in ARMS}


# ---------------------------------------------------------------- per-candidate gates
def g_crashfree(k):
    return CF[k]


def g_brrr(k):
    if not (B[k] and B["A"] and B["A2"]) or gates["G1_controls"][0] != "PASS":
        return ["UNKNOWN", f"brrr {B[k]} controls {gates['G1_controls'][0]}"]
    mx = max(B["A"]["tok_s"], B["A2"]["tok_s"])
    rr = B[k]["tok_s"] / mx
    d = B[k]["step_ms"] - st.mean([B["A"]["step_ms"], B["A2"]["step_ms"]])
    return ["PASS" if rr >= 1.025 else "FAIL",
            f"{k} {B[k]['tok_s']} tok/s vs max(A,A2) {mx}: {rr:.3f}x (bar >= 1.025x); step {d:+.3f} ms vs mean(A,A2)"]


def g_perclass(k):
    bad, miss, rat = [], [], []
    for key in [f"{c}_{m}" for m in ("c1", "c8") for c in CLASSES]:
        va, va2, vk = PC["A"][key], PC["A2"][key], PC[k][key]
        if None in (va, va2, vk) or min(PC["A"][key + "_n"], PC["A2"][key + "_n"], PC[k][key + "_n"]) < 2:
            miss.append(key)
            continue
        sp = abs(va - va2) / ((va + va2) / 2)
        bar = min(va, va2) * (1 - max(0.02, 1.5 * sp))
        rat.append(f"{key} {vk / ((va + va2) / 2):.3f}x")
        if vk < bar:
            bad.append(f"{key} {vk:.1f}<{bar:.1f}")
    s = "FAIL" if bad else ("UNKNOWN" if miss else "PASS")
    return [s, (f"BELOW: {bad} " if bad else "") + (f"MISSING (needs r1+r2): {miss} " if miss else "") + "| " + " ".join(rat)]


def g_lib(k):
    if not (LIB["A"] and LIB["A2"] and LIB[k]):
        return ["UNKNOWN", f"lib missing: A={bool(LIB['A'])} A2={bool(LIB['A2'])} {k}={bool(LIB[k])}"]
    bad, miss, rat = [], [], []
    for key in LIB_KEYS:
        va, va2, vk = LIB["A"].get(key), LIB["A2"].get(key), LIB[k].get(key)
        if None in (va, va2, vk):
            miss.append(key)
            continue
        rat.append(f"{key} {vk / ((va + va2) / 2):.3f}x")
        if vk < 0.95 * min(va, va2):
            bad.append(f"{key} {vk:.1f}<{0.95 * min(va, va2):.1f}")
    s = "FAIL" if bad else ("UNKNOWN" if miss else "PASS")
    return [s, "[basis 0.95 x min(A,A2)] " + (f"BELOW: {bad} " if bad else "") + (f"MISSING: {miss} " if miss else "")
            + "| " + " ".join(rat)]


def g_prefill(k):
    if not (TT["A"] and TT["A2"] and TT[k]):
        return ["UNKNOWN", f"ttft missing: A={bool(TT['A'])} A2={bool(TT['A2'])} {k}={bool(TT[k])}"]
    bad, miss, rat = [], [], []
    for c in TTFT_CTX:
        xa, xa2, xk = TT["A"].get(c) or {}, TT["A2"].get(c) or {}, TT[k].get(c) or {}
        if min(xa.get("n", 0), xa2.get("n", 0), xk.get("n", 0)) < TTFT_REPS:
            miss.append(c)
            continue
        wt = min(xa["tok_s_mean"], xa2["tok_s_mean"])
        wf = max(xa["ttft_s_mean"], xa2["ttft_s_mean"])
        rat.append(f"{c}: {xk['tok_s_mean']:.0f} tok/s ({xk['tok_s_mean'] / wt:.3f}x worse ctrl) "
                   f"TTFT {xk['ttft_s_mean']:.3f}s ({xk['ttft_s_mean'] / wf:.3f}x)")
        if xk["tok_s_mean"] < 0.95 * wt or xk["ttft_s_mean"] > 1.05 * wf:
            bad.append(c)
    s = "FAIL" if bad else ("UNKNOWN" if miss else "PASS")
    return [s, (f"BELOW at ctx {bad} " if bad else "") + (f"MISSING/short: {miss} " if miss else "") + "| " + "; ".join(rat)]


def g_tf(k):
    fk, fc = FM.get(k), FM.get("A2")
    if not fk or not fc:
        return ["UNKNOWN", f"flips_margin missing: {k}={fk} A2={fc}"]
    wbar = 1.25 * fc["W"]
    cbar = max(fc["confident_flips_all"], 2) + 1
    ok = fk["W"] <= wbar and fk["confident_flips_all"] <= cbar and fk["hi_agent_tool"] == 0
    return ["PASS" if ok else "FAIL",
            f"W {fk['W']:.3f} (bar <= 1.25 x A2's {fc['W']:.3f} = {wbar:.3f}); confident flips {fk['confident_flips_all']} "
            f"(bar <= max({fc['confident_flips_all']},2)+1 = {cbar}); agent/tool confident {fk['hi_agent_tool']} (bar 0); "
            f"raw TF flips {TFC.get(k)} vs A2 {TFC.get('A2')}"]


def g_gauntlet(k):
    if GA[k] == "None/None":
        return ["UNKNOWN", f"{k} gauntlet missing"]
    return ["PASS" if GA[k] == "12/12" else "FAIL", f"{k} {GA[k]} (A {GA['A']}, A2 {GA['A2']})"]


def g_probe(k):
    d = PR[k]
    if not d:
        return ["UNKNOWN", f"{k} probe62417.json missing"]
    n = len(d.get("results", []))
    pk = p5(d)
    pc = [p5(PR[a]) for a in CTRL]
    accs = [x["acc"] for x in pc if x]
    bar = min(P5_ABS, min(accs) - P5_SLACK) if len(accs) == 2 else P5_ABS
    own = probe_fails(d)
    shared = [c for c in own if any(c in probe_fails(PR[a]) for a in CTRL if PR[a])]
    p5_ok = bool(pk) and pk["acc"] >= bar - 1e-9
    t = (f"{k}: {n} cells, corruption-cell fails {own or 'none'} (shared with a control: {shared or 'none'}); "
         f"P5 {pk} vs bar {bar:.3f} = min({P5_ABS}, min(ctrl {accs}) - {P5_SLACK}); controls "
         f"A={cells['probe62417']['A']} A2={cells['probe62417']['A2']}")
    if n < PROBE_MIN_CELLS or not pk:
        return ["FAIL" if n else "UNKNOWN", t + f" — needs >= {PROBE_MIN_CELLS} cells incl. P5"]
    if [c for c in own if c not in shared] or not p5_ok:
        return ["FAIL", t]
    if shared:
        return ["CONTROL-ALSO", t + " — the failing cell(s) also failed on a control: lane-level, needs a human read"]
    return ["PASS", t]


def g_soak(k):
    d, ca, cb = SK[k], SK["A"], SK["A2"]
    if not d:
        return ["UNKNOWN", f"{k} soak.json missing"]
    if not (ca and cb):
        return ["UNKNOWN", f"control soak missing: A={bool(ca)} A2={bool(cb)}"]
    bar = min(ca["n_correct"], cb["n_correct"])
    shape_bad = d["n_loop"] > 0 or d["n_nonstop"] > 0
    ctrl_bad = any(c["n_loop"] > 0 or c["n_nonstop"] > 0 for c in (ca, cb))
    t = (f"{k}: loops {d['n_loop']} nonstop {d['n_nonstop']} correct {d['n_correct']}/{d['n']} (bar >= {bar} = worse "
         f"control); A {ca['n_correct']}/{ca['n']} loops {ca['n_loop']} nonstop {ca['n_nonstop']}; "
         f"A2 {cb['n_correct']}/{cb['n']} loops {cb['n_loop']} nonstop {cb['n_nonstop']}")
    if shape_bad and ctrl_bad:
        return ["CONTROL-ALSO", t + " — a control also looped/failed to stop: lane-level, needs a human read"]
    if shape_bad or d["n_correct"] < bar:
        return ["FAIL", t]
    return ["PASS", t]


def onem_line(k, d):
    r = (d or {}).get("result") or {}
    ra = (OM["A"] or {}).get("result") or {}
    ratio = f"{r['ttft_s'] / ra['ttft_s']:.3f}x A" if r.get("ttft_s") and ra.get("ttft_s") else "A n/a"
    return (f"{k}: ok={(d or {}).get('ok')} why={(d or {}).get('why')} prompt_tokens={r.get('prompt_tokens')} "
            f"TTFT {r.get('ttft_s')}s ({ratio}; A ok={(OM['A'] or {}).get('ok')} TTFT {ra.get('ttft_s')}s)")


def onem_pass(d):
    r = (d or {}).get("result") or {}
    return bool(d and d.get("ok") and r.get("finish") == "stop" and 850000 <= (r.get("prompt_tokens") or 0) <= 1048576)


def g_1m(k):
    d = OM[k]
    if not d:
        if k == "S4":
            if onem_pass(OM["S8"]) and CF["S8"][0] == "PASS":
                return ["INHERITED", f"S4 smoke not run ({cells['onem_skipped']['S4']}); S8 (same side .so, same unsplit "
                                     f"prefill path) passed and was CRASH-FREE: {onem_line('S8', OM['S8'])}. "
                                     "promote_guard_split.sh re-runs the 1M smoke on h2-REF before re-arming"]
            return ["UNKNOWN", f"S4 onem.json missing and S8 cannot lend a PASS (S8 onem pass={onem_pass(OM['S8'])}, "
                               f"crash-free {CF['S8'][0]})"]
        return ["UNKNOWN", f"{k} onem.json missing ({cells['onem_skipped'][k]})"]
    t = onem_line(k, d)
    if onem_pass(d):
        return ["PASS", t]
    if OM["A"] and not onem_pass(OM["A"]):
        return ["CONTROL-ALSO", t + " — control A failed the 1M smoke too"]
    return ["FAIL", t]


GFN = {"G_crashfree": g_crashfree, "G_brrr": g_brrr, "G_tf": g_tf, "G_gauntlet": g_gauntlet, "G_probe62417": g_probe,
       "G_soak": g_soak, "G_1M": g_1m, "G_perclass": g_perclass, "G_lib": g_lib, "G_prefill": g_prefill}
for a in CTRL:
    gates[f"{a}_crashfree"] = CF[a]
results = {}
for k in CANDS:
    if not ran[k] and not booted_dead[k]:
        results[k] = "SKIPPED" if skipped[k] else "MISSING"
        gates[f"{k}_ran"] = ["SKIP", skipped[k] or "no facts.json"]
        continue
    if not ran[k]:   # boot died after docker run
        gates[f"{k}_G_crashfree"] = CF[k]
        if not ctrl_ok:
            results[k] = "INVALID-CONTROL"
        else:
            results[k] = "NO" if CF[k][0] == "FAIL" else "INCOMPLETE"
        continue
    g = {"engaged": ["PASS" if engaged.get(k) else "FAIL",
                     json.dumps((cells.get(f"facts_{k}") or {}).get("engage"))[:500]]}
    for name, fn in GFN.items():
        g[name] = fn(k)
    for name, v in g.items():
        gates[f"{k}_{name}"] = v

    def cls(c):
        return [g[n][0] for n in GFN if GCLASS[n] in c]
    hard, hold = cls(("CRASH", "SPEED", "CORR")), cls(("HOLD",))
    hold_all = hold + [s for s in hard if s == "CONTROL-ALSO"]
    if not ctrl_ok or gates["G1_controls"][0] == "INVALID-CONTROL":
        res = "INVALID-CONTROL"
    elif g["G_crashfree"][0] == "FAIL":
        res = "NO"
    elif not valid.get(k):
        res = "INVALID-NOTENGAGED" if not engaged.get(k) else "INVALID"
    elif "FAIL" in hard:
        res = "NO"
    elif "UNKNOWN" in hard or gates["G1_controls"][0] != "PASS":
        res = "INCOMPLETE"
    elif "FAIL" in hold_all or "CONTROL-ALSO" in hold_all:
        res = "HOLD"
    elif "UNKNOWN" in hold:
        res = "INCOMPLETE"
    else:
        res = "PROMOTE-READY"
    results[k] = res

ready = [k for k in CANDS if results.get(k) == "PROMOTE-READY"]
winner = None
if ready:
    ready.sort(key=lambda k: -B[k]["tok_s"])
    winner = ready[0]
    if len(ready) == 2 and abs(B["S8"]["tok_s"] - B["S4"]["tok_s"]) / max(B["S8"]["tok_s"], B["S4"]["tok_s"]) <= 0.005:
        winner = "S8"
me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
out = {"result": " ".join(f"{k}={v}" for k, v in results.items()),
       "suffix": "AP2-" + "-".join(f"{k}{v}" for k, v in results.items()),
       "per_candidate": results,
       "winner": ({"arm": winner, "splits": NSPLIT[winner], "brrr_tok_s": B[winner]["tok_s"],
                   "onem": gates.get(f"{winner}_G_1M", ["?"])[0]} if winner else None),
       "promotion": "none (James decides; promote_guard_split.sh is a separate script)",
       "gate_class": GCLASS, "verdict_sha256": me, "gates": gates, "cells": cells}
json.dump(out, open(f"{R}/verdict.json", "w"), indent=1, default=str)
print(f"RESULT {out['result']}")
print(f"WINNER {out['winner']}")
print(f"VERDICT-SHA256 {me}")
for gname, v in gates.items():
    print(f"{gname}: {v[0]} — {v[1]}")
print("brrr2k T=0:", B)
print("lib:", {a: {k: v for k, v in (x or {}).items() if not k.startswith('_')} for a, x in LIB.items()})
print("ttft:", {a: ({c: (s or {}).get('tok_s_mean') for c, s in x.items()} if x else None) for a, x in TT.items()})
print("onem:", cells["onem"])

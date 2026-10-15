#!/usr/bin/env python3
"""Aggregate P3 raw (mirrored from gb300:/models/drafter-train/p3/raw) into summary.json + markdown tables.
Usage: python3 aggregate.py <p3_dir>   (reads <p3_dir>/raw, writes <p3_dir>/summary.json and <p3_dir>/tables.md)"""
import json, re, statistics, sys
from pathlib import Path

P3 = Path(sys.argv[1]); RAW = P3 / "raw"
ARMS = [a for a in ["JM_K4", "INCOAI_K4", "JM_K7", "INCOAI_K7", "INCOAI_bf58_K4"] if (RAW / a).exists()]
CELLS = ["prose_c1_r1", "prose_c1_r2", "prose_c1_r3", "prose_c4", "prose_c8", "structured_c1", "structured_c4",
         "structured_c8", "agent_c1", "agent_c4", "agent_c8", "probe_prose0", "probe_prose1", "probe_code0", "probe_code1"]
med = lambda xs: statistics.median(xs) if xs else None


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def f2(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


S = {"arms": {}, "lane": "GB300 vLLM TP1 (vllm-glm53f-p0:ac9126e58-aux-kvextra, CUDA graphs on), 127.0.0.1:30021 — NOT the Spark TP2 lane",
     "acceptance_definition": {"matrix_g1_p1": "completion_tokens / drafts (diag_matrix.py)", "cq": "1 + accepted / drafts (holdout_client.py)"}}
for arm in ARMS:
    A = RAW / arm; r = {}
    t = jl(A / "boot/timing.json"); r["boot"] = t
    ex = (A / "boot/boot-excerpt.txt").read_text() if (A / "boot/boot-excerpt.txt").exists() else ""
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", ex); r["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
    r["mem_after_ready"] = jl(A / "boot/mem-after-ready.json"); r["mem_before_teardown"] = jl(A / "boot/mem-before-teardown.json")
    r["perpos_at_boot"] = (A / "boot/perpos-at-boot.txt").read_text().strip().splitlines()[:8] if (A / "boot/perpos-at-boot.txt").exists() else None
    # matrix passes
    passes = {}
    for mi in (1, 2, 3):
        d = A / f"matrix-m{mi}"
        if not (d / "summary.json").exists():
            continue
        cells = {}
        for c in CELLS:
            j = jl(d / f"{c}.json")
            if j:
                cells[c] = {"acc": j.get("acceptance_length"), "tok_s": j.get("aggregate_tok_s"),
                            "per_pos": j.get("per_pos_accept_prob"), "errors": len(j.get("errors") or []),
                            "kinds": j.get("response_kinds"), "mem_during": j.get("mem_during"),
                            "per_stream": j.get("per_stream_tok_s")}
        passes[mi] = cells
    r["matrix_passes"] = passes
    mx = {}
    for c in CELLS:
        accs = [passes[p][c]["acc"] for p in passes if c in passes[p] and passes[p][c]["acc"] is not None]
        toks = [passes[p][c]["tok_s"] for p in passes if c in passes[p] and passes[p][c]["tok_s"] is not None]
        mx[c] = {"m1_acc": passes.get(1, {}).get(c, {}).get("acc"), "m1_tok_s": passes.get(1, {}).get(c, {}).get("tok_s"),
                 "med3_acc": med(accs), "med3_tok_s": med(toks), "all_acc": accs, "all_tok_s": toks}
    r["matrix"] = mx
    p1r = [passes[1][c]["acc"] for c in ("prose_c1_r1", "prose_c1_r2", "prose_c1_r3") if 1 in passes and c in passes[1]]
    p1t = [passes[1][c]["tok_s"] for c in ("prose_c1_r1", "prose_c1_r2", "prose_c1_r3") if 1 in passes and c in passes[1]]
    all9 = [passes[p][c]["acc"] for p in passes for c in ("prose_c1_r1", "prose_c1_r2", "prose_c1_r3") if c in passes[p]]
    all9t = [passes[p][c]["tok_s"] for p in passes for c in ("prose_c1_r1", "prose_c1_r2", "prose_c1_r3") if c in passes[p]]
    r["c1_prose"] = {"m1_reps_acc": p1r, "m1_median_acc": med(p1r), "m1_median_tok_s": med(p1t),
                     "all9_acc": all9, "all9_median_acc": med(all9), "all9_median_tok_s": med(all9t),
                     "pass_medians_acc": [med([passes[p][c]["acc"] for c in ("prose_c1_r1", "prose_c1_r2", "prose_c1_r3") if c in passes[p]]) for p in sorted(passes)]}
    # per-position: prose r3 and code0, m1 and per-position median across passes
    pp = {}
    for c in ("prose_c1_r3", "probe_code0", "probe_prose1"):
        rows = [passes[p][c]["per_pos"] for p in sorted(passes) if c in passes[p] and passes[p][c]["per_pos"]]
        if rows:
            keys = sorted(rows[0], key=int)
            pp[c] = {"m1": {k: rows[0][k] for k in keys}, "med3": {k: med([rw.get(k) for rw in rows if rw.get(k) is not None]) for k in keys}}
    r["per_pos"] = pp
    mins = [cell["mem_during"]["during"]["head_kB"] for p in passes.values() for cell in p.values() if (cell.get("mem_during") or {}).get("during")]
    hbms = [cell["mem_during"]["during"]["gpu1_hbm_used_MiB"] for p in passes.values() for cell in p.values() if (cell.get("mem_during") or {}).get("during")]
    r["mem_c8_min_memavail_kB"] = min(mins) if mins else None; r["mem_c8_max_hbm_MiB"] = max(hbms) if hbms else None
    r["matrix_errors"] = sum(cell["errors"] for p in passes.values() for cell in p.values())
    # cq
    cq = {}
    for tag in ("c16x500", "c6x500", "c1x100"):
        j = jl(A / f"cq-{tag}.json")
        if j:
            cq[tag] = {k: j.get(k) for k in ("ok", "fail", "prompts", "concurrency", "wall_s", "output_tokens", "output_tok_per_s",
                                              "drafts", "mean_acceptance_length", "draft_token_acceptance_rate", "per_position_acceptance_rate")}
            cq[tag]["accept_len_completion"] = round(j["output_tokens"] / j["drafts"], 3) if j.get("drafts") else None
    r["cq"] = cq
    j = jl(A / "p1-holdout-300.json")
    if j:
        r["p1_holdout"] = {"overall": j["overall"], "picked_counts": j["picked_counts"],
                           "buckets": {b: {k: v for k, v in bb.items() if k != "requests"} for b, bb in j["buckets"].items()}}
    j = jl(A / "g1/summary.json")
    if j:
        r["g1"] = j["summary"]
        r["g1_rows"] = [{k: row.get(k) for k in ("name", "regime", "acceptance_length", "tok_s", "completion_tokens", "reasoning_tokens", "has_reasoning", "per_pos")} for row in j["rows"]]
    S["arms"][arm] = r

# ---- bars
A = S["arms"]
def g(arm, *path):
    x = A.get(arm)
    for p in path:
        if x is None:
            return None
        x = x.get(p) if isinstance(x, dict) else None
    return x
WEEKEND_TONY_K7_CODE = {"probe_code0": 3.63, "probe_code1": 3.35}
WEEKEND = {"TONY_K4_prose_med": 2.17, "TONY_K7_prose_med": 2.31, "TONY_K4_agent": 4.33, "AR_c1_prose_tok_s": 15.16}
bars = {}
jp = g("JM_K4", "c1_prose", "m1_median_acc")
bars["prose_c1_k4_ge_2.6"] = {"value_m1_median3": jp, "value_all9_median": g("JM_K4", "c1_prose", "all9_median_acc"),
                              "incoai_k4_same_lane_m1_median3": g("INCOAI_K4", "c1_prose", "m1_median_acc"),
                              "incoai_k4_same_lane_all9": g("INCOAI_K4", "c1_prose", "all9_median_acc"),
                              "pass": (jp is not None and jp >= 2.6), "negative_result_lt_2.4": (jp is not None and jp < 2.4)}
ag = g("JM_K4", "matrix", "agent_c1", "med3_acc")
bars["agent_k4_ge_4.33"] = {"value_agent_c1_med3": ag, "agent_c4_med3": g("JM_K4", "matrix", "agent_c4", "med3_acc"),
                            "agent_c8_med3": g("JM_K4", "matrix", "agent_c8", "med3_acc"),
                            "incoai_k4_agent_c1_med3": g("INCOAI_K4", "matrix", "agent_c1", "med3_acc"),
                            "pass": (ag is not None and ag >= 4.33 - 1e-9)}
code = {}
for c in ("probe_code0", "probe_code1"):
    jm = g("JM_K7", "matrix", c, "med3_acc"); ic = g("INCOAI_K7", "matrix", c, "med3_acc")
    code[c] = {"jm_k7_med3": jm, "incoai_k7_same_lane_med3": ic, "delta_same_lane": (jm - ic) if (jm is not None and ic is not None) else None,
               "weekend_tony_k7_spark": WEEKEND_TONY_K7_CODE[c], "delta_vs_weekend_cross_day_cross_lane": (jm - WEEKEND_TONY_K7_CODE[c]) if jm is not None else None}
dl = [code[c]["delta_same_lane"] for c in code if code[c]["delta_same_lane"] is not None]
bars["probe_code_k7_within_-0.2_of_incoai"] = {"by_prompt": code, "worst_delta_same_lane": min(dl) if dl else None,
                                               "pass": (min(dl) >= -0.2) if dl else None}
cqb = {}
for tag in ("c16x500", "c6x500", "c1x100"):
    jm = g("JM_K4", "cq", tag, "mean_acceptance_length"); ic = g("INCOAI_K4", "cq", tag, "mean_acceptance_length")
    jm7 = g("JM_K7", "cq", tag, "mean_acceptance_length"); ic7 = g("INCOAI_K7", "cq", tag, "mean_acceptance_length")
    cqb[tag] = {"jm_k4": jm, "incoai_k4": ic, "delta_k4": (jm - ic) if (jm and ic) else None,
                "jm_k7": jm7, "incoai_k7": ic7, "delta_k7": (jm7 - ic7) if (jm7 and ic7) else None}
d6 = cqb["c6x500"]["delta_k4"]
bars["cq_holdout_within_-0.1_of_incoai"] = {"by_run": cqb, "primary": "c6x500 at K=4 (lane max-num-seqs per card; c16 also run)",
                                            "pass": (d6 >= -0.1) if d6 is not None else None}
S["bars"] = bars
S["weekend_reference_spark_tp2"] = WEEKEND
(P3 / "summary.json").write_text(json.dumps(S, indent=2) + "\n")

# ---- markdown tables
L = []
hdr = "| cell | " + " | ".join(ARMS) + " |"; sep = "|---|" + "---:|" * len(ARMS)
L += ["### Matrix, pass m1 (weekend protocol): tok/s / acceptance", "", hdr, sep]
for c in CELLS:
    L.append(f"| {c} | " + " | ".join(f"{f2(A[a]['matrix'][c]['m1_tok_s'],1)} / {f2(A[a]['matrix'][c]['m1_acc'])}" for a in ARMS) + " |")
L += ["", "### Matrix, median of 3 passes (m1–m3): tok/s / acceptance", "", hdr, sep]
for c in CELLS:
    L.append(f"| {c} | " + " | ".join(f"{f2(A[a]['matrix'][c]['med3_tok_s'],1)} / {f2(A[a]['matrix'][c]['med3_acc'])}" for a in ARMS) + " |")
L += ["", "C1 prose acceptance: m1 median of 3 reps / median of all 9 reps / per-pass medians", ""]
for a in ARMS:
    cp = A[a]["c1_prose"]
    L.append(f"- {a}: **{f2(cp['m1_median_acc'])}** / {f2(cp['all9_median_acc'])} / {', '.join(f2(x) for x in cp['pass_medians_acc'])}; tok/s m1 median {f2(cp['m1_median_tok_s'],1)}, all-9 median {f2(cp['all9_median_tok_s'],1)}")
L += ["", "### Per-position acceptance (probability a verify step keeps draft position k), median of 3 passes", "",
      "| arm | fixture | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for c in ("prose_c1_r3", "probe_code0", "probe_prose1"):
    for a in ARMS:
        pp = A[a]["per_pos"].get(c)
        if pp:
            v = [pp["med3"].get(str(i)) for i in range(7)]
            L.append(f"| {a} | {c} | " + " | ".join(f2(x) if x is not None else "" for x in v) + " |")
L += ["", "### canada-quant holdout kit (thinking ON, T=1.0/top_p 0.95, 1024 tok, /v1/completions): mean accept len (1+acc/drafts) · out tok/s · ok", "",
      "| run | " + " | ".join(ARMS) + " |", sep]
for tag in ("c16x500", "c6x500", "c1x100"):
    L.append(f"| {tag} | " + " | ".join(
        (f"{f2(A[a]['cq'][tag]['mean_acceptance_length'],3)} · {f2(A[a]['cq'][tag]['output_tok_per_s'],0)} · {A[a]['cq'][tag]['ok']}/{A[a]['cq'][tag]['prompts']}" if tag in A[a]["cq"] else "—") for a in ARMS) + " |")
L += ["", "cq per-position (c6x500):", "", "| arm | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for a in ARMS:
    pp = (A[a]["cq"].get("c6x500") or {}).get("per_position_acceptance_rate") or {}
    L.append(f"| {a} | " + " | ".join(f2(pp.get(str(i))) if pp.get(str(i)) is not None else "" for i in range(7)) + " |")
L += ["", "### P1 holdout, 300 stratified (thinking OFF greedy, 300 tok): acceptance (completion/drafts) · agg tok/s", "",
      "| bucket (n) | " + " | ".join(ARMS) + " |", sep]
for b in ("chat", "agent", "code_stem", "structured"):
    n = next((A[a]["p1_holdout"]["picked_counts"][b] for a in ARMS if "p1_holdout" in A[a]), "?")
    L.append(f"| {b} ({n}) | " + " | ".join(
        (f"{f2(A[a]['p1_holdout']['buckets'][b]['accept_len_completion'])} · {f2(A[a]['p1_holdout']['buckets'][b]['agg_tok_s'],0)}" if "p1_holdout" in A[a] else "—") for a in ARMS) + " |")
L.append("| **overall** | " + " | ".join((f"**{f2(A[a]['p1_holdout']['overall']['accept_len_completion'])}**" if "p1_holdout" in A[a] else "—") for a in ARMS) + " |")
L += ["", "### Decode regime (G1), prose accept median by regime (5 prompts, 512 tok, one pass)", "", "| regime | " + " | ".join(ARMS) + " |", sep]
for rg in ("think_off_t0", "reason_low_t0", "think_on_t0", "think_on_t1"):
    L.append(f"| {rg} | " + " | ".join(f2(((A[a].get('g1') or {}).get(rg) or {}).get('prose_accept_median')) for a in ARMS) + " |")
L += ["", "### Boot, KV, memory", "", "| arm | boot→ready | KV tokens | HBM after ready (MiB) | HBM max during C8 (MiB) | min MemAvailable during C8 (GiB) | matrix HTTP errors |", "|---|---:|---:|---:|---:|---:|---:|"]
for a in ARMS:
    r = A[a]
    L.append(f"| {a} | {r['boot']['boot_to_ready_s'] if r['boot'] else '—'} s | {format(r['kv_tokens'], ',') if r['kv_tokens'] else '—'} | {(r['mem_after_ready'] or {}).get('after_ready_hbm_MiB')} | {r['mem_c8_max_hbm_MiB']} | {f2((r['mem_c8_min_memavail_kB'] or 0)/1048576,1)} | {r['matrix_errors']} |")
(P3 / "tables.md").write_text("\n".join(L) + "\n")
print(json.dumps(S["bars"], indent=1))

#!/usr/bin/env python3
"""bfcl_cloud_slice.py — the *same* BFCL held-out protocol (harness/protocol.yaml) against a cloud model via Hermes'
own Anthropic OAuth path, on a fixed seeded 100-item slice; cost = list price × usage tokens from the response.

  Why OAuth: James's Anthropic access is Claude Max OAuth, not a billed API key. Usage tokens are real; the $ is an
  ESTIMATE at published list price (no invoice) and is labelled so on the card.
  Why 100: cost-per-solved is dominated by tokens, not the solve rate; ±5 pt on the rate moves the ratio ~6%.

Usage:  HERMES_HOME=~/.hermes/profiles/milo <hermes venv python> bfcl_cloud_slice.py <tag> <data_dir> <out_dir>
Env:    MODEL (claude-fable-5-1) PRICE_IN PRICE_OUT ($/MTok; default Fable 5.1 list 10/50) N (100) SEED (20260921)
        CONC (4) FLOOR_ONLY=1 -> no API calls: count input tokens for the whole held-out suite and print the floor.
"""
import json, os, random, sys, time, hashlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_gate import bfcl_to_openai_tools, grade, load, SUITES  # same grader, same tool conversion

MODEL = os.getenv("MODEL", "claude-fable-5-1")
PRICE_IN = float(os.getenv("PRICE_IN", "10")); PRICE_OUT = float(os.getenv("PRICE_OUT", "50"))
N = int(os.getenv("N", "100")); SEED = int(os.getenv("SEED", "20260921")); CONC = int(os.getenv("CONC", "4"))
MAXTOK = int(os.getenv("MAXTOK", "2048"))


def slice_cases(data_dir):
    cases = [c for name in SUITES["heldout"] for c in load(data_dir, name)]
    cases.sort(key=lambda c: c["id"])
    rng = random.Random(SEED)
    # stratified: keep the suite's live_simple : live_multiple ratio
    by = {}
    for c in cases: by.setdefault(c["cat"], []).append(c)
    out = []
    for cat, cs in sorted(by.items()):
        k = round(N * len(cs) / len(cases)); out += rng.sample(cs, k)
    out.sort(key=lambda c: c["id"])
    ids_sha = hashlib.sha256("\n".join(c["id"] for c in out).encode()).hexdigest()[:16]
    return cases, out, ids_sha


def anthropic_client():
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from agent.anthropic_adapter import build_anthropic_client
    rt = resolve_runtime_provider(requested="anthropic", target_model=MODEL)
    return build_anthropic_client(rt["api_key"], rt.get("base_url")), rt


def ask(client, case):
    from agent.anthropic_adapter import build_anthropic_kwargs, create_anthropic_message
    tools = bfcl_to_openai_tools(case["function"])
    # Anthropic tool names must match ^[a-zA-Z0-9_-]{1,128}$; BFCL live names carry dots (math.hypot). Reversible rename.
    back = {}
    for t in tools:
        n = t["function"]["name"]; w = n.replace(".", "_dot_")
        back[w] = n; t["function"]["name"] = w
    msgs = case["question"][0]
    kw = build_anthropic_kwargs(MODEL, msgs, tools, MAXTOK, None, tool_choice="auto", is_oauth=True)
    # protocol says T=0; Fable 5.1 rejects the temperature parameter outright ("deprecated for this model"),
    # so the request carries none — recorded as a protocol deviation in the summary.
    kw.pop("temperature", None)
    t0 = time.time()
    try:
        m = create_anthropic_message(client, kw, prefer_stream=True)
    except Exception as ex:  # noqa
        return {"id": case["id"], "error": f"{type(ex).__name__}: {str(ex)[:200]}", "secs": round(time.time() - t0, 1)}
    tcs = [b for b in m.content if getattr(b, "type", "") == "tool_use"]
    text = "".join(getattr(b, "text", "") for b in m.content if getattr(b, "type", "") == "text")
    call = None
    if tcs:
        name = tcs[0].name
        # Hermes' OAuth path renames tools on the wire (mcp__ namespace); strip it back for grading
        if name.startswith("mcp__"): name = name.split("__", 2)[-1]
        name = back.get(name, name.replace("_dot_", "."))
        call = {"name": name, "arguments": dict(tcs[0].input)}
    ok, why = grade(call, case["gt"])
    u = m.usage
    usage = {"prompt_tokens": u.input_tokens + (getattr(u, "cache_read_input_tokens", 0) or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0),
             "input_uncached": u.input_tokens, "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
             "completion_tokens": u.output_tokens}
    return {"id": case["id"], "cat": case["cat"], "ok": ok, "why": why, "n_calls": len(tcs), "call": call,
            "content": text[:200], "stop": m.stop_reason, "usage": usage, "secs": round(time.time() - t0, 1)}


def main():
    tag, data_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    cases, sl, ids_sha = slice_cases(data_dir)
    client, rt = anthropic_client()
    src = rt.get("source") or rt.get("credential_source") or "?"
    print(f"[{tag}] model {MODEL} via anthropic runtime (source {src}); held-out {len(cases)} cases; slice {len(sl)} seed {SEED} ids-sha {ids_sha}")
    if os.getenv("FLOOR_ONLY") == "1":
        # count input tokens for the whole suite with the API's count endpoint (free), assume 60 output tokens/call
        tot = 0; n_ok = 0; skipped = 0
        for c in cases:
            from agent.anthropic_adapter import build_anthropic_kwargs
            tls = bfcl_to_openai_tools(c["function"])
            for t in tls: t["function"]["name"] = t["function"]["name"].replace(".", "_dot_")
            kw = build_anthropic_kwargs(MODEL, c["question"][0], tls, MAXTOK, None, tool_choice="auto", is_oauth=True)
            kw.pop("max_tokens", None); kw.pop("temperature", None); kw.pop("stream", None)
            try: tot += client.messages.count_tokens(**kw).input_tokens; n_ok += 1
            except Exception as ex:  # BFCL schemas Anthropic rejects (invalid draft-2020-12): counted, not estimated
                skipped += 1
        if skipped: print(f"[{tag}] FLOOR: {skipped} cases skipped (schema rejected by Anthropic); floor scales the {n_ok} countable cases to {len(cases)}")
        tot = int(tot * len(cases) / max(n_ok, 1))
        floor = (tot * PRICE_IN + len(cases) * 60 * PRICE_OUT) / 1e6
        print(f"[{tag}] FLOOR: {tot} input tokens over {len(cases)} calls (+60 out/call assumed) = ${floor:.2f} per suite at 100% solved -> ${floor/len(cases)*1000:.2f} per 1000 solved (lower bound)")
        json.dump({"tag": tag, "model": MODEL, "suite_n": len(cases), "input_tokens": tot, "assumed_out_per_call": 60,
                   "price_in": PRICE_IN, "price_out": PRICE_OUT, "usd_floor_suite": round(floor, 4),
                   "usd_per_1000_solved_floor": round(floor / len(cases) * 1000, 4)}, open(f"{out}/cloud-{tag}-floor.json", "w"), indent=1)
        return
    t0 = time.time(); recs = []
    with ThreadPoolExecutor(CONC) as ex:
        for i, r in enumerate(ex.map(lambda c: ask(client, c), sl), 1):
            recs.append(r)
            if i % 20 == 0: print(f"[{tag}] {i}/{len(sl)} {time.time()-t0:.0f}s")
    wall = round(time.time() - t0, 1)
    with open(f"{out}/cloud-{tag}.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    good = [r for r in recs if "error" not in r]
    solved = sum(r["ok"] for r in good); errs = len(recs) - len(good)
    tin = sum(r["usage"]["prompt_tokens"] for r in good); tout = sum(r["usage"]["completion_tokens"] for r in good)
    cached = sum(r["usage"]["cache_read"] for r in good)
    usd = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6
    by = {}
    for r in good: b = by.setdefault(r["cat"], [0, 0]); b[0] += r["ok"]; b[1] += 1
    summ = {"tag": tag, "model": MODEL, "protocol": "harness/protocol.yaml (T=0, tool_choice auto, max_tokens 2048, single turn)",
            "slice_n": len(sl), "seed": SEED, "ids_sha16": ids_sha, "solved": solved, "errors": errs,
            "pct": round(solved / len(sl) * 100, 1), "by_cat": {k: f"{v[0]}/{v[1]}" for k, v in by.items()},
            "input_tokens": tin, "cache_read_tokens": cached, "output_tokens": tout, "price_in": PRICE_IN, "price_out": PRICE_OUT,
            "usd_slice": round(usd, 4), "usd_per_1000_solved": round(usd / max(solved, 1) * 1000, 3),
            "usd_per_1000_attempts": round(usd / len(sl) * 1000, 3), "wall_s": wall, "conc": CONC,
            "billing": "Claude Max OAuth; $ = list price x usage tokens (estimate, no invoice)",
            "deviations": ["temperature omitted: Fable 5.1 rejects the parameter (protocol says T=0)",
                           "tool names with dots renamed on the wire (Anthropic name pattern), reversed before grading"]}
    json.dump(summ, open(f"{out}/cloud-{tag}-summary.json", "w"), indent=1)
    print(f"[{tag}] {MODEL}: solved {solved}/{len(sl)} ({summ['pct']}%) {summ['by_cat']} · errors {errs} · in {tin} (cached {cached}) out {tout} tok · "
          f"${usd:.3f} slice · ${summ['usd_per_1000_solved']} per 1000 solved · ${summ['usd_per_1000_attempts']} per 1000 attempts · {wall}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""s2_greedy.py — greedy self-repeat / spec-vs-k0 parity for Step-5-Preview (stdlib only).

capture:  python3 s2_greedy.py capture <tag> [outdir]       -> greedy-<tag>.json
compare:  python3 s2_greedy.py compare a.json b.json [outdir] -> identical count + first divergence (token index)
20 prompts (prose/code/agent/shell/json/math, the MTP prompt classes), C1 sequential, T=0, max_tokens=200, chat
template default (thinking on, so most of the 200 tokens are reasoning_content). Token ids come back through
logprobs (top-2 kept since 2026-10-15 11:05 for near-tie analysis) + return_tokens_as_token_ids, so divergence is located by token, not by character.
Env: BASE (http://127.0.0.1:18000/v1) MODEL (step5) MAXTOK (200)
"""
import json, os, sys, time, urllib.request

BASE = os.getenv("BASE", "http://127.0.0.1:18000/v1").rstrip("/")
MODEL = os.getenv("MODEL", "step5")
MAXTOK = int(os.getenv("MAXTOK", "200"))

PROMPTS = [
    ("prose-1", "Explain to a curious high-school student how a mixture-of-experts language model decides which experts to use for each token."),
    ("prose-2", "Write a 300-word history of the transistor from 1947 to the integrated circuit."),
    ("prose-3", "Describe a winter morning in a small mountain town, as the first paragraph of a novel."),
    ("prose-4", "Summarize the causes of the 1929 stock market crash for a general reader."),
    ("code-1", "Write a Python function knee(rows) that takes [(conc, agg_tps)] and returns the concurrency where the marginal gain per added stream first drops below 10% of the C1 throughput. Include a docstring and two asserts."),
    ("code-2", "Implement an LRU cache in Rust with get and put in O(1). Full code, no external crates."),
    ("code-3", "Write a TypeScript function that debounces an async function and cancels stale in-flight calls with AbortController."),
    ("agent-1", "You are a planning agent with tools search_web(query), read_file(path), write_file(path, content), run_shell(cmd). Plan the steps to find and fix a failing unit test in a Python repo. Output a numbered plan naming the tool for each step."),
    ("agent-2", "You have a tool get_weather(city, unit). The user asks: should I bring an umbrella to Telluride and Pensacola tomorrow? Explain which tool calls you would make, in order, with exact arguments."),
    ("agent-3", "A user's request: 'Rename every .jpeg in ~/Pictures to .jpg and tell me how many changed.' As an agent with run_shell(cmd), list the exact commands you would run and how you'd verify."),
    ("shell-1", "Write a bash script that retries a curl health check against http://localhost:18000/v1/models every 10 seconds for at most 10 minutes and exits non-zero on timeout."),
    ("shell-2", "Give a one-liner that finds the 10 largest files under /var/log, human-readable sizes, sorted descending."),
    ("shell-3", "Write a systemd unit and timer that runs /usr/local/bin/backup.sh every night at 02:30 with a 5-minute random delay."),
    ("json-1", "Return ONLY a JSON object with keys host, port, model, max_model_len for a vLLM server on port 18000 serving step5 with a 262144-token context."),
    ("json-2", "Return ONLY a JSON array of 5 objects {name, symbol, atomic_number} for the first five elements of the periodic table."),
    ("json-3", "Convert this to JSON with keys name, age, cities (array): 'Ana is 34 and has lived in Lisbon, Porto and Madrid.' Output only JSON."),
    ("math-1", "A train leaves at 14:05 and travels 312 km at 104 km/h, then waits 17 minutes, then travels 90 km at 60 km/h. When does it arrive? Show the arithmetic."),
    ("math-2", "Is 2^31 - 1 prime? Explain how you would check it without a computer, then give the answer."),
    ("math-3", "Find all real x with x^4 - 5x^2 + 4 = 0. Show the steps."),
    ("math-4", "Compute the integral of x * e^(2x) dx and verify by differentiation."),
]


def post(obj):
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def capture(tag, outdir="."):
    out = {}
    for pid, p in PROMPTS:
        t0 = time.time()
        j = post({"model": MODEL, "messages": [{"role": "user", "content": p}], "temperature": 0, "top_p": 1,
                  "max_tokens": MAXTOK, "logprobs": True, "top_logprobs": 2, "return_tokens_as_token_ids": True})
        ch = j["choices"][0]; msg = ch["message"]
        lp = (ch.get("logprobs") or {}).get("content") or []
        toks = [e.get("token") for e in lp]
        lps = [e.get("logprob") for e in lp]
        top2 = [[(t.get("token"), t.get("logprob")) for t in (e.get("top_logprobs") or [])] for e in lp]
        out[pid] = {"reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "", "content": msg.get("content") or "",
                    "tokens": toks, "logprobs": lps, "top2": top2, "n": len(toks), "finish": ch.get("finish_reason"), "secs": round(time.time() - t0, 2),
                    "completion_tokens": j.get("usage", {}).get("completion_tokens")}
        print(f"[{tag}] {pid} n={len(toks)} finish={ch.get('finish_reason')} {out[pid]['secs']}s", flush=True)
    fn = os.path.join(outdir, f"greedy-{tag}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print(f"[{tag}] wrote {fn}")


def compare(fa, fb, outdir="."):
    a, b = json.load(open(fa)), json.load(open(fb))
    same, rows = 0, []
    for pid, _ in PROMPTS:
        x, y = a.get(pid), b.get(pid)
        if not x or not y or not x["n"] or not y["n"]:
            rows.append({"id": pid, "result": "missing"}); continue
        tx, ty = x["tokens"], y["tokens"]
        if tx == ty:
            same += 1; rows.append({"id": pid, "result": "identical", "tokens_a": len(tx), "tokens_b": len(ty)}); continue
        i = next((i for i, (p, q) in enumerate(zip(tx, ty)) if p != q), min(len(tx), len(ty)))
        rows.append({"id": pid, "result": f"first divergence at token {i}", "div": i, "tokens_a": len(tx), "tokens_b": len(ty)})
    ta = os.path.basename(fa)[7:-5]; tb = os.path.basename(fb)[7:-5]
    summ = {"a": fa, "b": fb, "identical": same, "n": len(PROMPTS), "rows": rows}
    out = os.path.join(outdir, f"greedy-compare-{ta}-vs-{tb}.json")
    json.dump(summ, open(out, "w"), indent=1)
    print(f"GREEDY identical={same}/{len(PROMPTS)} ({ta} vs {tb}) -> {out}")
    for r in rows:
        print(f"  {r['id']}: {r['result']}")


if __name__ == "__main__":
    if sys.argv[1] == "capture": capture(sys.argv[2], *(sys.argv[3:4]))
    elif sys.argv[1] == "compare": compare(sys.argv[2], sys.argv[3], *(sys.argv[4:5]))
    else: print(__doc__)

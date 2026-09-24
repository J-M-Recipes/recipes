#!/usr/bin/env python3
"""mimo_greedy.py — full-depth greedy decode parity for the MiMo-V2.6-Pro lane (stock placement vs hotsplit).

capture:  python3 mimo_greedy.py capture <tag>            -> greedy-<tag>.json
compare:  python3 mimo_greedy.py compare a.json b.json    -> identical count + first-divergence token index per prompt

C1, sequential, T=0, thinking off (chat_template_kwargs enable_thinking=false), max_tokens 512, natural stop.
Returns token ids via logprobs so divergence is located by token, not by character.
Env: BASE_URL (http://127.0.0.1:30007/v1) MODEL (mimo26-pro) MAXTOK (512)
"""
import json, os, sys, time, urllib.request

BASE = os.getenv("BASE_URL", "http://127.0.0.1:30007/v1").rstrip("/")
MODEL = os.getenv("MODEL", "mimo26-pro")
MAXTOK = int(os.getenv("MAXTOK", "512"))

PROMPTS = [
    ("tool-shaped-1", "List the docker containers on a Linux host and check free disk on /. Show the exact shell commands you would run and explain each flag in one line."),
    ("tool-shaped-2", "Return ONLY a JSON object with keys host, port, model, max_model_len for a vLLM server on port 30007 serving mimo26-pro with a 262144-token context."),
    ("code-1", "Write a Python function knee(rows) that takes [(conc, agg_tps)] and returns the concurrency where the marginal gain per added stream first drops below 10% of the C1 throughput. Include a docstring and two asserts."),
    ("code-2", "Write a bash script that retries a curl health check against http://localhost:30007/v1/models every 10 seconds for at most 10 minutes and exits non-zero on timeout."),
    ("code-3", "Implement an LRU cache in Rust with get and put in O(1). Full code, no external crates."),
    ("prose-1", "Explain to a curious high-school student how a mixture-of-experts language model decides which experts to use for each token."),
    ("prose-2", "Write a 300-word history of the transistor from 1947 to the integrated circuit."),
    ("reason-1", "A train leaves at 14:05 and travels 312 km at 104 km/h, then waits 17 minutes, then travels 90 km at 60 km/h. When does it arrive? Show the arithmetic."),
    ("reason-2", "Is 2^31 - 1 prime? Explain how you would check it without a computer, then give the answer."),
    ("structured-1", "Produce a markdown table of the planets of the solar system with columns: name, mean distance from the Sun in AU, number of known moons (approximate is fine)."),
    ("ops-1", "A GPU server's Grace host memory shows 120 GiB free after loading a model with 320 GiB of experts offloaded. List five things that could consume host memory over time and how to monitor each."),
    ("long-1", "Write a detailed technical design for a crash-safe job queue on SQLite: schema, state machine, locking, idempotency, retries. Be thorough."),
]


def post(obj):
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def capture(tag):
    out = {}
    for pid, p in PROMPTS:
        t0 = time.time()
        j = post({"model": MODEL, "messages": [{"role": "user", "content": p}], "temperature": 0, "top_p": 1,
                  "max_tokens": MAXTOK, "logprobs": True, "top_logprobs": 0,
                  "chat_template_kwargs": {"enable_thinking": False}})
        ch = j["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        toks = [e.get("token") for e in lp]
        out[pid] = {"text": ch["message"].get("content") or "", "tokens": toks, "n": len(toks),
                    "finish": ch.get("finish_reason"), "secs": round(time.time() - t0, 2),
                    "completion_tokens": j.get("usage", {}).get("completion_tokens")}
        print(f"[{tag}] {pid} n={len(toks)} finish={ch.get('finish_reason')} {out[pid]['secs']}s", flush=True)
    fn = f"greedy-{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"[{tag}] wrote {fn}")


def compare(fa, fb):
    a, b = json.load(open(fa)), json.load(open(fb))
    same, rows = 0, []
    for pid, _ in PROMPTS:
        x, y = a.get(pid), b.get(pid)
        if not x or not y:
            rows.append((pid, "missing", None, None)); continue
        tx, ty = x["tokens"], y["tokens"]
        if tx == ty and x["text"] == y["text"]:
            same += 1; rows.append((pid, "identical", len(tx), len(ty))); continue
        i = next((i for i, (p, q) in enumerate(zip(tx, ty)) if p != q), min(len(tx), len(ty)))
        rows.append((pid, f"first divergence at token {i}", len(tx), len(ty)))
    summ = {"a": fa, "b": fb, "identical": same, "n": len(PROMPTS),
            "rows": [{"id": r[0], "result": r[1], "tokens_a": r[2], "tokens_b": r[3]} for r in rows]}
    out = f"greedy-compare-{os.path.basename(fa)[7:-5]}-vs-{os.path.basename(fb)[7:-5]}.json"
    json.dump(summ, open(out, "w"), indent=1)
    print(f"GREEDY identical={same}/{len(PROMPTS)}  ({fa} vs {fb}) -> {out}")
    for r in rows:
        print(f"  {r[0]}: {r[1]} (tokens {r[2]} vs {r[3]})")


if __name__ == "__main__":
    if sys.argv[1] == "capture": capture(sys.argv[2])
    elif sys.argv[1] == "compare": compare(sys.argv[2], sys.argv[3])
    else: print(__doc__)

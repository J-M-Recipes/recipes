#!/usr/bin/env python3
"""muse_gates.py BASE OUTDIR — first-boot gates + quick C1, stdlib only.
Gates: models, chat (200 tok), reasoning split, single tool call parsed, 4-turn tool replay, greedy repeat x2.
Bench: C1 decode tok/s, 3 reps x 4 classes (prose/code/json/agent), T=0, 256 tokens forced (min_tokens+ignore_eos), streaming; C4 prose.
"""
import json, sys, time, urllib.request, threading
BASE, OUT = sys.argv[1], sys.argv[2]
M = "muse"
res = {}


def post(path, body, timeout=900):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def chat(msgs, **kw):
    b = {"model": M, "messages": msgs, "temperature": 0}; b.update(kw); return post("/chat/completions", b)


def gate(name, fn):
    t = time.time()
    try:
        ok, info = fn()
    except Exception as e:  # noqa
        ok, info = False, repr(e)[:400]
    res[name] = {"pass": bool(ok), "info": info, "s": round(time.time() - t, 1)}
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info)[:400], flush=True)


gate("models", lambda: (lambda d: (d["data"][0]["id"] == M, d["data"][0].get("max_model_len")))(json.load(urllib.request.urlopen(BASE + "/models", timeout=30))))


def g_chat():
    r = chat([{"role": "user", "content": "In two sentences, what is a lighthouse keeper's job?"}], max_tokens=3000)
    m = r["choices"][0]["message"]; c = (m.get("content") or "")
    return len(c.split()) >= 10, {"content": c[:300], "reasoning_len": len(m.get("reasoning_content") or m.get("reasoning") or ""), "usage": r["usage"]}
gate("chat", g_chat)


def g_reason():
    r = chat([{"role": "user", "content": "What is 17*23? Think, then answer with just the number."}], max_tokens=2000)
    m = r["choices"][0]["message"]; c = m.get("content") or ""; rs = m.get("reasoning_content") or m.get("reasoning") or ""
    return "391" in c and "<think" not in c, {"content": c[:200], "reasoning_len": len(rs)}
gate("reasoning_split", g_reason)

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file and return its contents",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def g_tool():
    r = chat([{"role": "user", "content": "Read the file /srv/notes.txt and tell me what it says."}], tools=TOOLS, tool_choice="auto", max_tokens=2000)
    tc = r["choices"][0]["message"].get("tool_calls") or []
    a = json.loads(tc[0]["function"]["arguments"]) if tc else {}
    return bool(tc) and tc[0]["function"]["name"] == "read_file" and a.get("path") == "/srv/notes.txt", {"tool_calls": tc[:1], "finish": r["choices"][0]["finish_reason"]}
gate("tool1", g_tool)


def g_replay():
    files = {"/srv/a.txt": "Next file: /srv/b.txt", "/srv/b.txt": "Next file: /srv/c.txt", "/srv/c.txt": "Next file: /srv/d.txt",
             "/srv/d.txt": "The FINAL CODE is QX-7342."}
    msgs = [{"role": "user", "content": "Start by reading /srv/a.txt. Each file names the next file to read. Keep reading until you find the FINAL CODE, then reply with only the code."}]
    paths = []
    for turn in range(6):
        r = chat(msgs, tools=TOOLS, tool_choice="auto", max_tokens=3000)
        m = r["choices"][0]["message"]; tc = m.get("tool_calls") or []
        if not tc:
            return len(paths) >= 4 and "QX-7342" in (m.get("content") or ""), {"paths": paths, "final": (m.get("content") or "")[:200]}
        p = json.loads(tc[0]["function"]["arguments"]).get("path"); paths.append(p)
        am = {"role": "assistant", "content": m.get("content") or "", "tool_calls": tc[:1]}
        if m.get("reasoning_content"): am["reasoning_content"] = m["reasoning_content"]
        msgs += [am, {"role": "tool", "tool_call_id": tc[0]["id"], "content": files.get(p, "ERROR: no such file")}]
    return False, {"paths": paths, "note": "no final after 6 turns"}
gate("tools_replay4", g_replay)


def g_greedy():
    p = [{"role": "user", "content": "Write a 150-word paragraph about tidal power."}]
    a = chat(p, max_tokens=200, chat_template_kwargs={})["choices"][0]["message"]
    b = chat(p, max_tokens=200)["choices"][0]["message"]
    return (a.get("content"), a.get("reasoning_content")) == (b.get("content"), b.get("reasoning_content")), {"same": a.get("content") == b.get("content")}
gate("greedy_repeat", g_greedy)

PROMPTS = {"prose": "Write a long, flowing essay about the history of lighthouses and the people who kept them. Use full paragraphs, no lists.",
           "code": "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, type hints, docstrings, and pytest unit tests.",
           "json": "Output ONLY a JSON document (no prose) describing 15 fictional employees: each with id, name, email, department, title, start_date (ISO), salary_usd, skills (array of 3 strings), and manager_id (or null).",
           "agent": "You are a planning agent. Tools: search_web(query), read_file(path), write_file(path, content), run_shell(cmd). Output ONLY a JSON array of 12+ tool-call objects {\"tool\":..., \"args\":{...}, \"why\":...} that audit a Python repo for unused dependencies and fix requirements.txt."}


def stream_one(prompt, n=256):
    b = {"model": M, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": n, "min_tokens": n,
         "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=900)
    t0 = time.time(); tf = None; ct = 0
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        d = json.loads(line[5:])
        if d.get("choices") and tf is None: tf = time.time()
        if d.get("usage"): ct = d["usage"].get("completion_tokens", ct)
    t1 = time.time()
    return {"ttft": round(tf - t0, 3), "decode_tok_s": round((ct - 1) / (t1 - tf), 2) if ct > 1 else None, "tokens": ct}


bench = {}
for k, p in PROMPTS.items():
    stream_one(p, 32)
    reps = [stream_one(p) for _ in range(3)]
    bench[k] = {"reps": reps, "mean": round(sum(x["decode_tok_s"] for x in reps) / 3, 2)}
    print("C1", k, bench[k]["mean"], flush=True)
out = [None] * 4
def w(i): out[i] = stream_one(PROMPTS["prose"] + f" Variation {i}.")
for rep in range(2):
    th = [threading.Thread(target=w, args=(i,)) for i in range(4)]; t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
    agg = sum(x["tokens"] for x in out) / (time.time() - t0)
    bench.setdefault("c4_agg", []).append(round(agg, 2))
print("C4 agg", bench["c4_agg"], flush=True)
res["bench"] = bench
json.dump(res, open(OUT + "/gates.json", "w"), indent=1)
print("GATES", {k: v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v})

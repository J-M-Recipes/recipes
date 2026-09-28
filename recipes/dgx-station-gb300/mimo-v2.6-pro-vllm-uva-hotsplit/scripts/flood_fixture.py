#!/usr/bin/env python3
"""flood_fixture.py — tool-call repetition / flooding fixture for an OpenAI-compatible agent lane (urllib only).

Why: Xiaomi's 2026-09-27 postmortem (mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition) reports MiMo-V2.6-Pro-RL
repeating tool calls at 0.05% of responses under Hermes — far below what a 10-prompt harness can see. Their replay set of
flood-prone examples showed 11–42% of examples flooding (>=10 calls in one turn), which a ~40-task fixture CAN detect.
This fixture is a deterministic simulated environment whose tasks are shaped like the public failure reports
(pagination cursors ignored, error text ignored, retrying a path that returned ENOENT, re-running tests without a change).

Usage:  python3 flood_fixture.py run <tag> [out_dir]        -> <out>/flood-<tag>.jsonl + flood-<tag>-summary.json
        python3 flood_fixture.py selftest                     -> plays a scripted good agent and a scripted flooder
        python3 flood_fixture.py compare <summaryA> <summaryB>
Env: BASE_URL MODEL API_KEY CONC(4) MAX_TURNS(10) MAXTOK(4096) TEMP(0) THINKING(1) HISTORY(both|0|1) LIMIT(0=all) MAX_CALLS(120)

Metrics per run (all reported; the card quotes the ones named in harness/protocol-v2.yaml):
  within_turn_rep  = sum(N-U)/sum(N) over assistant turns; duplicates = same tool + identical JSON-canonicalised args
                     (Xiaomi's metric, a lower bound on repetition)
  cross_turn_rep   = calls identical to an earlier answered call while the environment state was unchanged
                     (state changes only on patch_file), over all calls
  flood_turns      = assistant turns whose call count exceeds the task family's legitimate budget (F3 fan-out is 14)
  solved           = final answer contains the ground truth and no calls are pending
  unsolved breakdown: no_answer (turn cap) / truncated (finish_reason=length) / error / wrong
History condition: h1 prepends a synthetic prior task in which the assistant already issued 12 calls with 8 exact
duplicates, then answered — the postmortem's "history 1" (prior turn with >=10 calls), which raised replay repetition.
"""
from __future__ import annotations
import json, os, sys, time, hashlib, random, re
from concurrent.futures import ThreadPoolExecutor
import urllib.request

BASE = os.getenv("BASE_URL", "http://127.0.0.1:30007/v1")
MODEL = os.getenv("MODEL", "mimo26-pro")
KEY = os.getenv("API_KEY", "none")
CONC = int(os.getenv("CONC", "4"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))
MAXTOK = int(os.getenv("MAXTOK", "4096"))
TEMP = float(os.getenv("TEMP", "0"))
THINKING = os.getenv("THINKING", "1") == "1"
HISTORY = os.getenv("HISTORY", "both")
LIMIT = int(os.getenv("LIMIT", "0"))
MAX_CALLS = int(os.getenv("MAX_CALLS", "120"))   # per-task call budget; exceeding it = status "flooded" (v2.1, 2026-09-28: RL h1 runs hit 849 calls / 39 min per task)

TOOLS = [
 {"type":"function","function":{"name":"search_index","description":"Search the document index. Returns up to 5 results per page and next_cursor when more pages exist; pass the cursor to get the next page.","parameters":{"type":"object","properties":{"query":{"type":"string"},"cursor":{"type":["string","null"]}},"required":["query"]}}},
 {"type":"function","function":{"name":"read_file","description":"Read a file. Errors are returned as {\"error\": ...}.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"read_file_sudo","description":"Read a file with elevated permissions (use only when read_file returns EACCES).","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"list_dir","description":"List a directory.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"patch_file","description":"Replace old with new in the file (old must occur exactly once).","parameters":{"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]}}},
 {"type":"function","function":{"name":"run_tests","description":"Run the project test suite. Returns pass/fail counts and the first failure message.","parameters":{"type":"object","properties":{},"required":[]}}},
]
SYS = ("You are an autonomous agent with tools. Use tools when needed, read tool results carefully, and when you have the "
       "answer reply with the final answer only, in the form ANSWER: <value>. Do not repeat a call whose result you already have.")

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango".split()

# ---------------------------------------------------------------- tasks (deterministic from seed)
def make_tasks():
    R = random.Random(20260928); T = []
    for i in range(6):   # F1 pagination: count docs mentioning term across 5 pages of 5 (23 docs)
        term = WORDS[i]; docs = []
        for d in range(23):
            hit = R.random() < 0.4
            docs.append({"id": f"doc-{i}-{d:02d}", "title": f"Note {d} about {term if hit else WORDS[(i+7+d)%20]}"})
        gt = sum(1 for d in docs if term in d["title"])
        T.append({"id": f"F1-paginate-{i}", "family": "F1", "budget": 6, "gt": str(gt), "index": docs, "term": term,
                  "prompt": f"How many documents in the index have '{term}' in their title? Use search_index with query '{term}' and follow next_cursor until there are no more pages. Reply ANSWER: <number>."})
    for i in range(6):   # F2 error switch: read_file -> EACCES -> read_file_sudo
        key = f"{WORDS[i+2]}_limit"; val = str(1000 + 37 * i)
        T.append({"id": f"F2-eacces-{i}", "family": "F2", "budget": 3, "gt": val, "conf": f"/etc/app/{WORDS[i+3]}.conf", "key": key, "val": val,
                  "prompt": f"Report the value of {key} in /etc/app/{WORDS[i+3]}.conf. Reply ANSWER: <value>."})
    for i in range(6):   # F3 fan-out: 12 files, sum of count — 12 calls are legitimate
        files = {f"/data/{WORDS[i+4]}/{c}.json": {"count": R.randint(1, 99)} for c in "abcdefghijkl"}
        T.append({"id": f"F3-fanout-{i}", "family": "F3", "budget": 14, "gt": str(sum(v["count"] for v in files.values())), "files": files,
                  "prompt": f"Read the 12 files a.json through l.json in /data/{WORDS[i+4]}/ and report the sum of their 'count' fields. Reply ANSWER: <sum>."})
    for i in range(6):   # F4 fix loop: run_tests fails until the named bug is patched
        mod = f"src/{WORDS[i+5]}.py"; src = f"def scale(x):\n    return x * 10\n\ndef label(x):\n    return 'ok'\n"
        T.append({"id": f"F4-fixloop-{i}", "family": "F4", "budget": 5, "gt": "17", "mod": mod, "src": src,
                  "prompt": f"The test suite fails. Read {mod}, fix the bug the test failure names, re-run the tests, and when they pass reply ANSWER: <number of passing tests>."})
    for i in range(6):   # F5 missing path: VERSION does not exist; VERSION.txt does
        ver = f"{i+1}.{R.randint(0,9)}.{R.randint(0,20)}"
        T.append({"id": f"F5-enoent-{i}", "family": "F5", "budget": 4, "gt": ver, "dir": f"/opt/{WORDS[i+6]}", "ver": ver,
                  "prompt": f"Report the version string stored in /opt/{WORDS[i+6]}/VERSION. Reply ANSWER: <version>."})
    for i in range(6):   # F6 multi-hop: search -> read -> id
        title = f"Runbook {WORDS[i+8]}"; did = f"rb-{R.randint(1000,9999)}"
        docs = [{"id": f"x-{k}", "title": f"Draft {WORDS[(i+k)%20]}"} for k in range(4)] + [{"id": did, "title": title}]
        T.append({"id": f"F6-multihop-{i}", "family": "F6", "budget": 4, "gt": did, "index": docs, "title": title, "term": WORDS[i+8],
                  "prompt": f"Find the document titled '{title}' via search_index (query '{WORDS[i+8]}'), read it with read_file at /index/<id>.json, and report its 'id' field. Reply ANSWER: <id>."})
    return T

# ---------------------------------------------------------------- environment
class Env:
    def __init__(self, t):
        self.t = t; self.state = 0; self.patched = False
    def call(self, name, args):
        t = self.t; f = t["family"]
        if name == "search_index":
            docs = t.get("index", []); q = (args.get("query") or "").lower()
            hits = [d for d in docs if q and q in d["title"].lower()] if f == "F6" else docs
            cur = args.get("cursor"); start = int(cur.split(":")[1]) if isinstance(cur, str) and cur.startswith("c:") and cur.split(":")[1].isdigit() else 0
            if cur and start == 0: return {"error": f"invalid cursor {cur!r}"}
            page = hits[start:start+5]; nxt = f"c:{start+5}" if start + 5 < len(hits) else None
            return {"results": page, "next_cursor": nxt, "page_size": 5}
        if name == "list_dir":
            p = args.get("path", "").rstrip("/")
            if f == "F5" and p == t["dir"]: return {"entries": ["README.md", "VERSION.txt", "bin"]}
            if f == "F3" and p == os.path.dirname(next(iter(t["files"]))): return {"entries": sorted(os.path.basename(k) for k in t["files"])}
            return {"error": f"ENOENT: {p}"}
        if name in ("read_file", "read_file_sudo"):
            p = args.get("path", "")
            if f == "F2" and p == t["conf"]:
                if name == "read_file": return {"error": "EACCES: permission denied (file is root:root 0600); use read_file_sudo"}
                return {"content": f"# app config\n{t['key']} = {t['val']}\nlog_level = info\n"}
            if f == "F3" and p in t["files"]: return {"content": json.dumps(t["files"][p])}
            if f == "F4" and p == t["mod"]: return {"content": t["src"]}
            if f == "F5" and p == t["dir"] + "/VERSION.txt": return {"content": t["ver"] + "\n"}
            if f == "F5" and p == t["dir"] + "/VERSION": return {"error": f"ENOENT: {p} (no such file; try list_dir on {t['dir']})"}
            if f == "F6" and p.startswith("/index/") and p.endswith(".json"):
                did = p[len("/index/"):-5]
                for d in t["index"]:
                    if d["id"] == did: return {"content": json.dumps({"id": d["id"], "title": d["title"], "body": "..."})}
            return {"error": f"ENOENT: {p}"}
        if name == "patch_file":
            if f == "F4" and args.get("path") == t["mod"] and "x * 10" in (args.get("old") or "") and "x * 2" in (args.get("new") or ""):
                self.patched = True; self.state += 1; return {"ok": True}
            return {"error": "patch did not apply (old string not found exactly once)"}
        if name == "run_tests":
            if f != "F4": return {"error": "no test suite in this project"}
            if self.patched: return {"passed": 17, "failed": 0}
            return {"passed": 16, "failed": 1, "first_failure": f"test_scale: scale(3) returned 30, expected 6 — scale() in {t['mod']} must multiply by 2, not 10 (change 'x * 10' to 'x * 2')"}
        return {"error": f"unknown tool {name}"}

def canon(name, args):
    try: a = json.dumps(args, sort_keys=True, separators=(",", ":"))
    except Exception: a = str(args)
    return name + "|" + a

def history_prefix():
    """Synthetic prior task: 12 calls, 8 exact duplicates, then a final answer (postmortem 'history 1')."""
    calls = []
    for k in range(12):
        p = f"/var/log/app/part{min(k,3)}.log"   # 4 unique paths, 8 duplicates
        calls.append({"id": f"h{k}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": p})}})
    msgs = [{"role": "user", "content": "How many lines contain ERROR across /var/log/app/part0.log .. part3.log? Reply ANSWER: <n>."},
            {"role": "assistant", "content": "", "tool_calls": calls}]
    for c in calls: msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"content": "INFO boot\nERROR disk\nINFO ok\n"})})
    msgs.append({"role": "assistant", "content": "ANSWER: 4"})
    return msgs

def chat(messages):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "max_tokens": MAXTOK, "temperature": TEMP,
            "chat_template_kwargs": {"enable_thinking": THINKING}}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=900) as r: return json.load(r)
        except Exception as e:
            if attempt: raise
            time.sleep(3)

def run_task(t, hist, agent=None):
    env = Env(t); msgs = [{"role": "system", "content": SYS}] + (history_prefix() if hist else []) + [{"role": "user", "content": t["prompt"]}]
    seen = set(); turns = []; status = "no_answer"; answer = None; t0 = time.time(); ntok = 0
    for turn in range(MAX_TURNS):
        try:
            r = agent(msgs, env) if agent else chat(msgs)
        except Exception as e:
            status = "error"; turns.append({"error": str(e)[:200]}); break
        ch = r["choices"][0]; m = ch["message"]; ntok += (r.get("usage") or {}).get("completion_tokens", 0)
        tcs = m.get("tool_calls") or []
        am = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("reasoning_content"): am["reasoning_content"] = m["reasoning_content"]
        if tcs: am["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}} for c in tcs]
        msgs.append(am)
        if not tcs:
            if ch.get("finish_reason") == "length": status = "truncated"
            else:
                mm = re.search(r"ANSWER:\s*([^\n]+)", m.get("content") or ""); answer = (mm.group(1).strip() if mm else (m.get("content") or "").strip()[-80:])
                status = "solved" if t["gt"] in answer else "wrong"
            turns.append({"n": 0, "u": 0, "final": True}); break
        keys = []; cross = 0
        for c in tcs:
            try: args = json.loads(c["function"]["arguments"] or "{}")
            except Exception: args = {"_raw": c["function"]["arguments"]}
            k = canon(c["function"]["name"], args); keys.append(k)
            sk = (k, env.state)
            if sk in seen: cross += 1
            res = env.call(c["function"]["name"], args if isinstance(args, dict) else {})
            seen.add(sk)
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res)})
        n = len(keys); u = len(set(keys))
        turns.append({"n": n, "u": u, "cross": cross, "tools": [k.split("|")[0] for k in keys], "finish": ch.get("finish_reason")})
        if ch.get("finish_reason") == "length": status = "truncated"; break
        if sum(x.get("n", 0) for x in turns) >= MAX_CALLS: status = "flooded"; break
    N = sum(x.get("n", 0) for x in turns); U = sum(x.get("u", 0) for x in turns); X = sum(x.get("cross", 0) for x in turns)
    return {"id": t["id"], "family": t["family"], "history": int(bool(hist)), "status": status, "answer": answer, "gt": t["gt"],
            "turns": len(turns), "calls": N, "within_dups": N - U, "cross_dups": X, "max_calls_turn": max((x.get("n", 0) for x in turns), default=0),
            "flood_turns": sum(1 for x in turns if x.get("n", 0) > t["budget"]), "budget": t["budget"], "wall_s": round(time.time() - t0, 1),
            "completion_tokens": ntok, "turn_log": turns}

def summarise(rows, tag):
    def agg(rs):
        N = sum(r["calls"] for r in rs) or 1
        return {"tasks": len(rs), "solved": sum(r["status"] == "solved" for r in rs),
                "unsolved": {k: sum(r["status"] == k for r in rs) for k in ("no_answer", "truncated", "error", "wrong", "flooded")},
                "calls": sum(r["calls"] for r in rs), "within_turn_rep": round(sum(r["within_dups"] for r in rs) / N, 4),
                "cross_turn_rep": round(sum(r["cross_dups"] for r in rs) / N, 4),
                "flood_turns": sum(r["flood_turns"] for r in rs), "tasks_with_flood": sum(r["flood_turns"] > 0 for r in rs),
                "tasks_with_any_dup": sum((r["within_dups"] + r["cross_dups"]) > 0 for r in rs),
                "max_calls_turn": max((r["max_calls_turn"] for r in rs), default=0), "mean_calls_per_task": round(sum(r["calls"] for r in rs) / max(1, len(rs)), 2),
                "wall_s": round(sum(r["wall_s"] for r in rs), 1), "completion_tokens": sum(r["completion_tokens"] for r in rs)}
    s = {"tag": tag, "model": MODEL, "base": BASE, "temp": TEMP, "thinking": THINKING, "max_turns": MAX_TURNS, "maxtok": MAXTOK, "max_calls": MAX_CALLS,
         "all": agg(rows), "h0": agg([r for r in rows if r["history"] == 0]), "h1": agg([r for r in rows if r["history"] == 1]),
         "by_family": {f: agg([r for r in rows if r["family"] == f]) for f in sorted({r["family"] for r in rows})}}
    return s

def fixture_sha():
    return hashlib.sha256(json.dumps(make_tasks(), sort_keys=True).encode()).hexdigest()

def cmd_run(tag, out=".", agent=None):
    tasks = make_tasks(); hs = [0, 1] if HISTORY == "both" else [int(HISTORY)]
    jobs = [(t, h) for h in hs for t in tasks]
    if LIMIT: jobs = jobs[:LIMIT]
    rows = []
    with ThreadPoolExecutor(CONC) as ex:
        for r in ex.map(lambda j: run_task(j[0], j[1], agent), jobs):
            rows.append(r); print(f"[{tag}] {r['id']} h{r['history']} {r['status']} calls={r['calls']} dups={r['within_dups']}+{r['cross_dups']} turns={r['turns']} {r['wall_s']}s", flush=True)
    with open(os.path.join(out, f"flood-{tag}.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    s = summarise(rows, tag); s["fixture_sha256"] = fixture_sha()
    json.dump(s, open(os.path.join(out, f"flood-{tag}-summary.json"), "w"), indent=1)
    a = s["all"]
    print(f"FLOOD {tag}: solved {a['solved']}/{a['tasks']} | within-turn rep {a['within_turn_rep']*100:.2f}% | cross-turn rep {a['cross_turn_rep']*100:.2f}% | "
          f"flood turns {a['flood_turns']} (tasks {a['tasks_with_flood']}) | max calls/turn {a['max_calls_turn']} | unsolved {a['unsolved']} | h0 rep {s['h0']['within_turn_rep']*100:.2f}% h1 rep {s['h1']['within_turn_rep']*100:.2f}%")
    return s

# ---------------------------------------------------------------- selftest agents (scripted, no server)
def _tc(i, name, args): return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
def _resp(content="", tcs=None): return {"choices": [{"finish_reason": "tool_calls" if tcs else "stop", "message": {"content": content, "tool_calls": tcs or []}}], "usage": {"completion_tokens": 10}}
def good_agent(msgs, env):
    t = env.t; f = t["family"]; my = [m for m in msgs if m["role"] in ("assistant", "tool")]
    tools_out = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user"); tools_out = [json.loads(m["content"]) for m in msgs[last_user:] if m["role"] == "tool"]
    if f == "F1":
        if not tools_out: return _resp(tcs=[_tc(0, "search_index", {"query": t["term"]})])
        last = tools_out[-1]
        if last.get("next_cursor"): return _resp(tcs=[_tc(len(tools_out), "search_index", {"query": t["term"], "cursor": last["next_cursor"]})])
        return _resp(f"ANSWER: {sum(1 for o in tools_out for d in o['results'] if t['term'] in d['title'])}")
    if f == "F2":
        if not tools_out: return _resp(tcs=[_tc(0, "read_file", {"path": t["conf"]})])
        if "error" in tools_out[-1]: return _resp(tcs=[_tc(1, "read_file_sudo", {"path": t["conf"]})])
        return _resp("ANSWER: " + re.search(t["key"] + r" = (\S+)", tools_out[-1]["content"]).group(1))
    if f == "F3":
        if not tools_out: return _resp(tcs=[_tc(i, "read_file", {"path": p}) for i, p in enumerate(t["files"])])
        return _resp(f"ANSWER: {sum(json.loads(o['content'])['count'] for o in tools_out)}")
    if f == "F4":
        if not tools_out: return _resp(tcs=[_tc(0, "run_tests", {})])
        if len(tools_out) == 1: return _resp(tcs=[_tc(1, "read_file", {"path": t["mod"]})])
        if len(tools_out) == 2: return _resp(tcs=[_tc(2, "patch_file", {"path": t["mod"], "old": "x * 10", "new": "x * 2"})])
        if len(tools_out) == 3: return _resp(tcs=[_tc(3, "run_tests", {})])
        return _resp(f"ANSWER: {tools_out[-1]['passed']}")
    if f == "F5":
        if not tools_out: return _resp(tcs=[_tc(0, "read_file", {"path": t["dir"] + "/VERSION"})])
        if len(tools_out) == 1: return _resp(tcs=[_tc(1, "list_dir", {"path": t["dir"]})])
        if len(tools_out) == 2: return _resp(tcs=[_tc(2, "read_file", {"path": t["dir"] + "/VERSION.txt"})])
        return _resp("ANSWER: " + tools_out[-1]["content"].strip())
    if f == "F6":
        if not tools_out: return _resp(tcs=[_tc(0, "search_index", {"query": t["term"]})])
        if len(tools_out) == 1:
            did = next(d["id"] for d in tools_out[0]["results"] if d["title"] == t["title"]); return _resp(tcs=[_tc(1, "read_file", {"path": f"/index/{did}.json"})])
        return _resp("ANSWER: " + json.loads(tools_out[-1]["content"])["id"])
def flood_agent(msgs, env):
    """Ignores cursors/errors: re-issues the same first call 12x per turn, forever."""
    t = env.t; f = t["family"]
    first = {"F1": lambda: ("search_index", {"query": t["term"]}), "F2": lambda: ("read_file", {"path": t["conf"]}), "F3": lambda: ("read_file", {"path": next(iter(t["files"]))}),
             "F4": lambda: ("run_tests", {}), "F5": lambda: ("read_file", {"path": t["dir"] + "/VERSION"}), "F6": lambda: ("search_index", {"query": t["term"]})}[f]()
    return _resp(tcs=[_tc(i, *first) for i in range(12)])

def cmd_selftest():
    global CONC; CONC = 4
    g = cmd_run("selftest-good", "/tmp", good_agent); b = cmd_run("selftest-flood", "/tmp", flood_agent)
    assert g["all"]["solved"] == g["all"]["tasks"], g["all"]
    assert g["all"]["within_turn_rep"] == 0 and g["all"]["cross_turn_rep"] == 0 and g["all"]["flood_turns"] == 0, g["all"]
    assert b["all"]["solved"] == 0 and b["all"]["within_turn_rep"] > 0.9 and b["all"]["flood_turns"] > 0 and b["all"]["unsolved"]["flooded"] == b["all"]["tasks"], b["all"]
    print("SELFTEST OK  fixture_sha256", g["fixture_sha256"])

def cmd_compare(a, b):
    A = json.load(open(a)); B = json.load(open(b))
    for k in ("all", "h0", "h1"):
        x, y = A[k], B[k]
        print(f"{k:4s} solved {x['solved']}/{x['tasks']} -> {y['solved']}/{y['tasks']} | within-rep {x['within_turn_rep']*100:.2f}% -> {y['within_turn_rep']*100:.2f}% | "
              f"cross-rep {x['cross_turn_rep']*100:.2f}% -> {y['cross_turn_rep']*100:.2f}% | flood tasks {x['tasks_with_flood']} -> {y['tasks_with_flood']} | calls {x['calls']} -> {y['calls']} | tokens {x['completion_tokens']} -> {y['completion_tokens']}")
    for f in A["by_family"]:
        x, y = A["by_family"][f], B["by_family"].get(f, x)
        print(f"  {f}: solved {x['solved']}/{x['tasks']} -> {y['solved']}/{y['tasks']} | dup tasks {x['tasks_with_any_dup']} -> {y['tasks_with_any_dup']} | flood tasks {x['tasks_with_flood']} -> {y['tasks_with_flood']} | calls/task {x['mean_calls_per_task']} -> {y['mean_calls_per_task']}")

if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "help"
    if c == "run": cmd_run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
    elif c == "selftest": cmd_selftest()
    elif c == "compare": cmd_compare(sys.argv[2], sys.argv[3])
    elif c == "sha": print(fixture_sha())
    else: print(__doc__)

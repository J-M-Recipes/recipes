#!/usr/bin/env python3
"""build_fixture_sessions.py — session-ordered replay fixture. Picks SESSIONS sessions with >= MIN_TURNS assistant turns,
takes the first TURNS assistant turns of each in order; each turn's ctx is the full prefix (capped at MAX_CTX_CHARS by
dropping the *middle*, keeping head and tail, so consecutive turns share a growing prefix as in production).
Output: list of {session, turns:[{ctx, has_tool}]}. Secrets scrubbed."""
import sqlite3, json, random, re, sys, os
DB = os.path.expanduser("~/.hermes/profiles/milo/state.db")
SESSIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20; TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
OUT = sys.argv[3] if len(sys.argv) > 3 else "transcript_fixture_sessions.json"
MAX_CTX_CHARS = 60000; MIN_TURNS = TURNS
random.seed(20260915)
SECRET = re.compile(r"(?i)(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|xox[abp]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|"
                    r"Bearer\s+[A-Za-z0-9._-]{16,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}|"
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----|"
                    r"(api[_-]?key|token|password|secret|passwd)\s*[=:]\s*['\"]?[A-Za-z0-9._\-/+]{12,})")
def scrub(s): return SECRET.sub("[REDACTED]", s or "")
con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
rows = con.execute("""select session_id, role, content, tool_calls from messages where role in ('user','assistant','tool') and active=1
                      order by session_id, display_order, id""").fetchall()
by = {}
for r in rows: by.setdefault(r["session_id"], []).append(r)
def msg(r):
    if r["role"] == "tool": return {"role": "user", "content": "[tool result: " + scrub((r["content"] or "")[:1500]) + "]"}
    c = scrub(r["content"] or "")
    if r["role"] == "assistant" and r["tool_calls"] and r["tool_calls"] not in ("", "[]"):
        try:
            calls = json.loads(r["tool_calls"])
            c += "\n" + "\n".join(f"<tool_call>{json.dumps({'name': (x.get('function') or {}).get('name') or x.get('name'), 'arguments': (x.get('function') or {}).get('arguments') or x.get('arguments')})}</tool_call>" for x in calls[:3])
        except Exception: pass
    return {"role": r["role"], "content": c[:4000]}
elig = [s for s, ms in by.items() if sum(1 for r in ms if r["role"] == "assistant") >= MIN_TURNS and ms[0]["role"] == "user"]
random.shuffle(elig); picked = elig[:SESSIONS]
out = []
for sid in picked:
    ms = by[sid]; turns = []; n = 0
    for i, r in enumerate(ms):
        if r["role"] != "assistant": continue
        n += 1
        if n > TURNS: break
        ctx = [msg(x) for x in ms[:i] if (x["content"] or x["tool_calls"])]
        if not ctx or ctx[0]["role"] != "user": continue
        total = sum(len(m["content"]) for m in ctx)
        if total > MAX_CTX_CHARS:                     # drop from the middle, keep head (shared prefix) and tail
            head = ctx[:4]; tail = ctx[4:]
            while tail and sum(len(m["content"]) for m in head + tail) > MAX_CTX_CHARS: tail.pop(0)
            ctx = head + tail
        if ctx[-1]["role"] == "assistant": ctx.append({"role": "user", "content": "Continue."})
        turns.append({"ctx": ctx, "has_tool": bool(r["tool_calls"] and r["tool_calls"] not in ("", "[]"))})
    if len(turns) >= MIN_TURNS // 2: out.append({"session": sid[:12], "turns": turns})
json.dump(out, open(OUT, "w"))
nt = sum(len(s["turns"]) for s in out)
print(f"wrote {OUT}: {len(out)} sessions, {nt} turns, {sum(t['has_tool'] for s in out for t in s['turns'])} tool; eligible sessions {len(elig)}; "
      f"redactions {sum(m['content'].count('[REDACTED]') for s in out for t in s['turns'] for m in t['ctx'])}")

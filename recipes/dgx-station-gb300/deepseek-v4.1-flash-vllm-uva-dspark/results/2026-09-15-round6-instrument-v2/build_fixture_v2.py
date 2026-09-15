#!/usr/bin/env python3
"""build_fixture_v2.py — sample real Hermes turns from state.db into a replay fixture.
Stratified: N/2 tool-call turns, N/2 text turns; sessions spread; ctx capped at MAX_CTX_CHARS
(oldest messages dropped, first user message kept); secrets scrubbed. Output: list of
{ctx:[{role,content}], has_tool, ref_len, session, ts}.  Never includes the reference output.
"""
import sqlite3, json, random, re, sys, os
DB = os.path.expanduser("~/.hermes/profiles/milo/state.db")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MAX_CTX_CHARS = 24000          # ~6K tokens; keeps 4-worker replay under ~15 min per round
MIN_CTX_MSGS = 3
OUT = sys.argv[2] if len(sys.argv) > 2 else "transcript_fixture_v2.json"
random.seed(20260915)

SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|xox[abp]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|"
                    r"Bearer\s+[A-Za-z0-9._-]{16,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}|"
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----|"
                    r"(?i)(api[_-]?key|token|password|secret|passwd)\s*[=:]\s*['\"]?[A-Za-z0-9._\-/+]{12,})")
def scrub(s): return SECRET.sub("[REDACTED]", s or "")

con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
rows = con.execute("""select id, session_id, role, content, tool_calls, tool_name, timestamp
                      from messages where role in ('user','assistant','tool') and active=1
                      order by session_id, display_order, id""").fetchall()
by_sess = {}
for r in rows: by_sess.setdefault(r["session_id"], []).append(r)

def msg(r):
    if r["role"] == "tool":
        return {"role": "user", "content": "[tool result: " + scrub((r["content"] or "")[:1500]) + "]"}
    c = scrub(r["content"] or "")
    if r["role"] == "assistant" and r["tool_calls"] and r["tool_calls"] not in ("", "[]"):
        try:
            calls = json.loads(r["tool_calls"])
            c += "\n" + "\n".join(f"<tool_call>{json.dumps({'name': (x.get('function') or {}).get('name') or x.get('name'), 'arguments': (x.get('function') or {}).get('arguments') or x.get('arguments')})}</tool_call>" for x in calls[:3])
        except Exception: pass
    return {"role": r["role"], "content": c[:4000]}

cands = []
for sid, ms in by_sess.items():
    for i, r in enumerate(ms):
        if r["role"] != "assistant" or i < MIN_CTX_MSGS: continue
        has_tool = bool(r["tool_calls"] and r["tool_calls"] not in ("", "[]"))
        ref_len = len(r["content"] or "") + (len(r["tool_calls"]) if has_tool else 0)
        if ref_len < 20: continue
        cands.append((sid, i, has_tool, ref_len, r["timestamp"]))
random.shuffle(cands)
tool_c = [c for c in cands if c[2]]; text_c = [c for c in cands if not c[2]]
picked, seen_sess = [], {}
def take(pool, want):
    out = []
    for c in pool:
        if len(out) >= want: break
        if seen_sess.get(c[0], 0) >= 2: continue      # ≤2 turns per session → spread across sessions
        seen_sess[c[0]] = seen_sess.get(c[0], 0) + 1; out.append(c)
    return out
picked = take(tool_c, N // 2) + take(text_c, N - N // 2)
fix = []
for sid, i, has_tool, ref_len, ts in picked:
    ms = by_sess[sid][:i]
    ctx = [msg(r) for r in ms if (r["content"] or r["tool_calls"])]
    if not ctx: continue
    first_user = next((m for m in ctx if m["role"] == "user"), None)
    while sum(len(m["content"]) for m in ctx) > MAX_CTX_CHARS and len(ctx) > 2:
        ctx.pop(0 if ctx[0] is not first_user else 1)
    if ctx[0]["role"] != "user": ctx.insert(0, {"role": "user", "content": "(continuing an earlier session)"})
    if ctx[-1]["role"] == "assistant": ctx.append({"role": "user", "content": "Continue."})
    fix.append({"ctx": ctx, "has_tool": has_tool, "ref_len": ref_len, "session": sid[:12], "ts": ts})
json.dump(fix, open(OUT, "w"))
sizes = [sum(len(m["content"]) for m in f["ctx"]) for f in fix]
print(f"wrote {OUT}: {len(fix)} turns, {sum(f['has_tool'] for f in fix)} tool, {len(set(f['session'] for f in fix))} sessions, "
      f"ctx chars median {sorted(sizes)[len(sizes)//2]} max {max(sizes)}; candidates {len(cands)}; redactions {sum(m['content'].count('[REDACTED]') for f in fix for m in f['ctx'])}")

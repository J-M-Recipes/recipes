#!/usr/bin/env python3
"""One-shot: confirm reasoning_content is populated and content is answer-only."""
import json, os, urllib.request
U = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1") + "/chat/completions"
EFFORT = os.getenv("EFFORT", "low")
p = {
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": "What is 15% of 240? Think, then answer with just the number."}],
    "max_tokens": 256,
    "temperature": 0.0,
    "chat_template_kwargs": {"reasoning_effort": EFFORT},
}
r = json.load(urllib.request.urlopen(urllib.request.Request(U, data=json.dumps(p).encode(), headers={"Content-Type": "application/json"}), timeout=300))
m = r["choices"][0]["message"]
rc = m.get("reasoning_content") or ""
c = m.get("content") or ""
u = r.get("usage") or {}
print(json.dumps({
    "effort": EFFORT,
    "finish": r["choices"][0].get("finish_reason"),
    "reasoning_len": len(rc),
    "content_len": len(c),
    "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    "completion_tokens": u.get("completion_tokens"),
    "reasoning_head": rc[:240],
    "content": c.strip()[:400],
    "split_ok": bool(rc.strip()) and bool(c.strip()) and "<think>" not in c,
}, indent=2))

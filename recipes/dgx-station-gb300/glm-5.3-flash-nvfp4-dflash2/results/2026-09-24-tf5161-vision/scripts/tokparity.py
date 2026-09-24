import json, hashlib, transformers
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
tools = [{"type": "function", "function": {"name": "calc", "description": "calculator",
          "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}}]
cases = [
    [{"role": "user", "content": "Write a short history essay about the Hanseatic League."}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "What is 17*23? Use the tool."}],
    [{"role": "user", "content": "def f(x):\n    return x**2  # ünïcødé ✓ 漢字"}],
]
out = []
for i, m in enumerate(cases):
    for kw in ({}, {"tools": tools}):
        s = t.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
        ids = t(s, add_special_tokens=False)["input_ids"]
        out.append((i, bool(kw), hashlib.sha256(s.encode()).hexdigest()[:12], len(ids), hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]))
print("tf", transformers.__version__, type(t).__name__, "vocab", len(t))
for r in out:
    print(r)

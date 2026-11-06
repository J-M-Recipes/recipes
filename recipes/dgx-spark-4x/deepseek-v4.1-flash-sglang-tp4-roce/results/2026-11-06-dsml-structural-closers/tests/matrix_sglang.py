"""DSML structural-closer matrix for SGLang's DeepSeekV41Detector (same cases as matrix.py).

Run in the SGLang lane image (CPU only). SGLANG_DSV41_STRUCTURAL_DSML=0 -> stock path
(or no overlay mounted). Output records share matrix.py's shape so compare.py applies.
"""
import json
import os
import re

from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai.protocol import Function, Tool
import sglang.srt.function_call.deepseekv41_detector as det

# Reuse the exact case texts from matrix.py (the block between the markers).
src = open("/p/matrix.py").read()
block = src[src.index('S, E = "'):src.index("def req(")]
ns = {"json": json}
exec(block, ns)
CASES, REASONING_CASES = ns["CASES"], ns["REASONING_CASES"]

tok = AutoTokenizer.from_pretrained("/model")
MODE = os.environ.get("SGLANG_DSV41_STRUCTURAL_DSML", "1")
OVERLAY = getattr(det, "OVERLAY_ID", "stock-file")

TOOLS = [
    Tool(type="function", function=Function(name="write_file", description="write a file", parameters={
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]})),
    Tool(type="function", function=Function(name="set_cfg", description="set config", parameters={
        "type": "object", "properties": {"n": {"type": "integer"}, "cfg": {"type": "object"},
                                         "tag": {"type": "string"}}, "required": ["n"]})),
]


def parsed(args):
    try:
        return (json.loads(args) if args else None), True
    except Exception:  # noqa
        return args, False


def run_nonstream(text):
    d = det.DeepSeekV41Detector()
    r = d.detect_and_parse(text, TOOLS)
    calls = []
    for c in r.calls:
        a, ok = parsed(c.parameters)
        calls.append({"name": c.name, "args": a, "json_ok": ok})
    return {"tools_called": bool(r.calls), "calls": calls, "content": r.normal_text}


def run_stream(text, chunk):
    d = det.DeepSeekV41Detector()
    ids = tok.encode(text, add_special_tokens=False)
    prev, content, calls = "", "", {}

    def absorb(r):
        nonlocal content
        if r.normal_text:
            content += r.normal_text
        for c in r.calls:
            slot = calls.setdefault(c.tool_index, {"name": None, "args": ""})
            if c.name:
                slot["name"] = c.name
            if c.parameters:
                slot["args"] += c.parameters

    for i in range(0, len(ids), chunk):
        cur = tok.decode(ids[: i + chunk])
        absorb(d.parse_streaming_increment(cur[len(prev):], TOOLS))
        prev = cur
    absorb(d.finish(TOOLS))
    out = []
    for k in sorted(calls):
        a, ok = parsed(calls[k]["args"])
        out.append({"name": calls[k]["name"], "args": a, "json_ok": ok})
    return {"calls": out, "content": content}


for name, text in CASES.items():
    rec = {"case": name, "mode": MODE, "overlay": OVERLAY}
    try:
        rec["nonstream"] = run_nonstream(text)
    except Exception as e:  # noqa
        rec["nonstream"] = {"error": f"{type(e).__name__}: {e}"}
    for chunk in (1, 3, 7, 10**6):
        key = f"stream{chunk if chunk < 10**6 else 'All'}"
        try:
            rec[key] = run_stream(text, chunk)
        except Exception as e:  # noqa
            rec[key] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(rec, ensure_ascii=False), flush=True)

# Reasoning is split before the detector in SGLang; only the post-think tool parse applies.
for name, text in REASONING_CASES.items():
    rec = {"case": name, "mode": MODE, "overlay": OVERLAY, "reasoning": {"n/a": "sglang reasoning parser untouched"}}
    try:
        rec["nonstream"] = run_nonstream(text.split("</think>", 1)[1])
    except Exception as e:  # noqa
        rec["nonstream"] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(rec, ensure_ascii=False), flush=True)

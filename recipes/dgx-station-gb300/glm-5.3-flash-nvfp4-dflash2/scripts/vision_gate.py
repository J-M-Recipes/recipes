#!/usr/bin/env python3
"""Image-input gate for an OpenAI-compatible GLM-5.3-Flash endpoint.

Fails closed if the server accepts an image but does not turn it into image tokens, which is what
SGLang v0.5.20 + transformers 5.12.1 does on this model: HTTP 200, no error, text-only prefill.

Checks, one request each:
  1. text-only baseline prompt_tokens
  2. same prompt + image: prompt_tokens must grow by >= MIN_DELTA, and
     usage.prompt_tokens_details.image_tokens (if the server reports it) must be > 0
  3. the answer (reasoning + content) must name the text in the image (HELLO)

Env: BASE_URL (default http://127.0.0.1:30001/v1), MODEL (glm-5.3-flash), API_KEY (optional),
     IMAGE (default: vision-gate.png next to this script), MIN_DELTA (default 100), EFFORT (low).
Prints one JSON line and 'VISION_GATE PASS|FAIL'; exit code 0 on pass, 1 on fail.
"""
import base64, json, os, sys, urllib.request

BASE = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1")
MODEL = os.getenv("MODEL", "glm-5.3-flash")
IMG = os.getenv("IMAGE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision-gate.png"))
MIN_DELTA = int(os.getenv("MIN_DELTA", "100"))
H = {"Content-Type": "application/json"}
if os.getenv("API_KEY"):
    H["Authorization"] = "Bearer " + os.environ["API_KEY"]
Q = "Describe this image: list the shapes with their colors, and quote any text exactly."


def chat(content):
    p = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 1024,
         "temperature": 0, "chat_template_kwargs": {"reasoning_effort": os.getenv("EFFORT", "low")}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(p).encode(), headers=H)
    return json.load(urllib.request.urlopen(req, timeout=300))


uri = "data:image/png;base64," + base64.b64encode(open(IMG, "rb").read()).decode()
base = chat(Q)
img = chat([{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": Q}])
u0, u1 = base["usage"], img["usage"]
details = u1.get("prompt_tokens_details") or {}
image_tokens = details.get("image_tokens")
msg = img["choices"][0]["message"]
answer = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
delta = u1["prompt_tokens"] - u0["prompt_tokens"]
checks = {
    "prompt_token_delta": delta >= MIN_DELTA,
    "image_tokens_reported_positive": image_tokens is None or image_tokens > 0,
    "reads_HELLO": "HELLO" in answer.upper(),
}
ok = all(checks.values())
print(json.dumps({"text_prompt_tokens": u0["prompt_tokens"], "image_prompt_tokens": u1["prompt_tokens"],
                  "delta": delta, "image_tokens": image_tokens, "checks": checks,
                  "content": (msg.get("content") or "")[:400]}))
print("VISION_GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

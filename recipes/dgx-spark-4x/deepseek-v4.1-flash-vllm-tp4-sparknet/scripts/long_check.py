#!/usr/bin/env python3
"""long_check.py: exact-length long prompts against the TensorFold lane (completions with token ids), plus needles.
usage: long_check.py BASE LEN [LEN ...]      e.g. http://127.0.0.1:8001 146421 145152 131073 149796 160000
For each LEN: a prompt of exactly LEN tokens (DeepSeek chat-mode frame around random filler words and one needle),
sent as token ids; checks HTTP 200, prompt_tokens == LEN, and that the reply carries the needle code."""
import json, random, sys, time, urllib.request

WORDS = ("amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid pylon quartz "
         "raven sierra tundra umber vessel willow xenon yarrow zephyr").split()
BOS, USER, ASST, THINK_END = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>"

def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def tok(base, text):
    return post(base, "/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]

def main():
    base = sys.argv[1]; lens = [int(x) for x in sys.argv[2:]]
    for n in lens:
        rnd = random.Random(n * 7919 + int(time.time()) % 100000)
        code = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
        head = tok(base, BOS + USER)
        tail = tok(base, f"\n\nWhat is the secret code mentioned in the text above? Reply with the code only.{ASST}{THINK_END}")
        room = n - len(head) - len(tail)
        words = " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(int(room * 0.55)))
        mid = len(words) // 2
        cut = words.rfind(" ", 0, mid)
        text = words[:cut] + f" The secret code is {code}. " + words[cut:]
        body_ids = tok(base, text)
        if len(body_ids) < room:
            more = tok(base, " " + " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(room - len(body_ids))))
            body_ids = body_ids + more
        # keep the needle: trim from the end of the filler, never the start
        body_ids = body_ids[:room]
        ids = head + body_ids + tail
        assert len(ids) == n, (len(ids), n)
        t0 = time.time()
        try:
            r = post(base, "/v1/completions", {"model": "deepseek-v4.1-flash-spark-vllm", "prompt": ids, "max_tokens": 16, "temperature": 0})
            dt = time.time() - t0
            out = r["choices"][0]["text"]
            ok = code in out
            print(json.dumps({"len": n, "http": 200, "prompt_tokens": r["usage"]["prompt_tokens"], "seconds": round(dt, 1),
                              "prefill_tok_s": round(n / dt, 0), "needle_ok": ok, "reply": out[:40]}), flush=True)
        except Exception as e:
            body = getattr(e, "read", lambda: b"")()
            print(json.dumps({"len": n, "error": str(e), "body": body[:300].decode("utf-8", "ignore") if body else ""}), flush=True)

main()

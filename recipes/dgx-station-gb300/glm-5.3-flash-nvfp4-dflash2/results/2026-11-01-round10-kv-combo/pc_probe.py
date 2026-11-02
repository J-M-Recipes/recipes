#!/usr/bin/env python3
"""pc_probe.py — does the GLM-Flash SGLang lane ever reuse a prefix? (Card PC, 2026-11-01)
Server must run with --enable-cache-report so usage.prompt_tokens_details.cached_tokens is returned.
Each test uses a fresh ~32K-token nonce-first context (no cross-test sharing), sends turn 1, then a turn 2 shape:
  P1 exact      : byte-identical messages resent
  P2 extend     : same user message + extra text appended inside it
  P3 agent      : [user, assistant(content only), user(new)]   <- what most OpenAI clients send
  P4 agent+think: [user, assistant(content + reasoning_content), user(new)]
  P5 sysprefix  : 32K system prompt + short user; turn 2 = same system + different short user (shared-system-prompt reuse)
Writes JSON to argv[1]; one summary line per test.
"""
import json, os, random, sys, time, urllib.request

BASE = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1"); H = {"Content-Type": "application/json"}
TOK = int(os.getenv("PC_TOK", "32000"))
W = ("amber basil cobalt delta ember fjord granite harbor iris juniper kelp lumen meadow nickel orchid pylon quartz "
     "raven saffron tundra umber vellum willow xenon yarrow zephyr").split()


def ctx(seed, n_tok):
    r = random.Random(seed); out = [f"[ctx {seed} {random.getrandbits(48):x}]"]
    n_chars = int(n_tok * 2.7)
    size = 0
    while size < n_chars:
        w = r.choice(W) + str(r.randint(0, 999)); out.append(w); size += len(w) + 1
    return " ".join(out)


def chat(messages, max_tokens=48):
    p = {"model": "glm-5.3-flash", "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
         "chat_template_kwargs": {"reasoning_effort": "low"}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(p).encode(), H), timeout=600))
    u = r.get("usage") or {}
    m = r["choices"][0]["message"]
    return {"wall_s": round(time.time() - t0, 2), "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "content": m.get("content") or "", "reasoning": m.get("reasoning_content") or ""}


def test(name, build_t1, build_t2):
    t1 = chat(build_t1())
    time.sleep(1)
    t2 = chat(build_t2(t1))
    frac = (t2["cached_tokens"] or 0) / max(1, t2["prompt_tokens"] or 1)
    print(f"[pc] {name}: t2 cached {t2['cached_tokens']}/{t2['prompt_tokens']} ({frac:.3f}) wall t1 {t1['wall_s']}s t2 {t2['wall_s']}s "
          f"| t1 cached {t1['cached_tokens']}", flush=True)
    for t in (t1, t2):
        t.pop("reasoning", None); t["content"] = t["content"][:120]
    return {"test": name, "t1": t1, "t2": t2, "t2_cached_frac": round(frac, 4)}


def main():
    out = {"tok": TOK, "started": time.strftime("%F %T %Z"), "tests": []}
    a = ctx("P1", TOK); q = "\n\nReply with the single word OK."
    out["tests"].append(test("P1-exact", lambda: [{"role": "user", "content": a + q}], lambda t1: [{"role": "user", "content": a + q}]))
    b = ctx("P2", TOK)
    out["tests"].append(test("P2-extend", lambda: [{"role": "user", "content": b + q}],
                             lambda t1: [{"role": "user", "content": b + q + " Then also say YES."}]))
    c = ctx("P3", TOK)
    out["tests"].append(test("P3-agent", lambda: [{"role": "user", "content": c + q}],
                             lambda t1: [{"role": "user", "content": c + q}, {"role": "assistant", "content": t1["content"] or "OK"},
                                         {"role": "user", "content": "Now reply with the single word YES."}]))
    d = ctx("P4", TOK)
    out["tests"].append(test("P4-agent+think", lambda: [{"role": "user", "content": d + q}],
                             lambda t1: [{"role": "user", "content": d + q},
                                         {"role": "assistant", "content": t1["content"] or "OK", "reasoning_content": t1["reasoning"]},
                                         {"role": "user", "content": "Now reply with the single word YES."}]))
    e = ctx("P5", TOK)
    out["tests"].append(test("P5-sysprefix", lambda: [{"role": "system", "content": e}, {"role": "user", "content": "Say OK."}],
                             lambda t1: [{"role": "system", "content": e}, {"role": "user", "content": "Say YES."}]))
    out["ended"] = time.strftime("%F %T %Z")
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()

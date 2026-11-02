#!/usr/bin/env python3
"""KV-capacity probe for GLM-5.3-Flash lanes (Card KV1, 2026-11-01).

Answers "what binds": the KDA running cap, the KV pool, or neither.
Steps (each: N concurrent streamed requests, distinct nonce-first prompts so no prefix sharing):
  S1  6 x ~200K tokens  (1.2M total: exceeds the 857K daily pool, fits a ~1.3M+ pool)
  S2 12 x ~64K  tokens  (768K total: fits the 857K pool; exceeds the 9-request KDA cap)
  S3  1 x ~960K tokens  needle (exceeds the 857K pool; a single 1M-class context)
Per request: TTFT, total wall, prompt_tokens, completion_tokens, error.
Server side (from `docker logs`, window-bounded by timestamps): max #running-req, max #queue-req,
max full token usage, retract count. Writes JSON to argv[1].
Env: BASE_URL (default http://127.0.0.1:30001/v1), CONT (container name for log scrape).
"""
import calendar, json, os, random, re, subprocess, sys, threading, time, urllib.request

BASE = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1")
CONT = os.getenv("CONT", "")
OUT = sys.argv[1]
H = {"Content-Type": "application/json"}
WORDS = ("amber basil cobalt delta ember fjord granite harbor iris juniper kelp lumen meadow nickel "
         "orchid pylon quartz raven saffron tundra umber vellum willow xenon yarrow zephyr").split()
CHARS_PER_TOK = None  # calibrated


def filler(seed, n_chars):
    rnd = random.Random(seed)
    out, size = [], 0
    while size < n_chars:
        w = rnd.choice(WORDS) + str(rnd.randint(0, 999))
        out.append(w); size += len(w) + 1
    return " ".join(out)


def build(seed, n_tok, needle=None):
    n_chars = int(n_tok * CHARS_PER_TOK)
    nonce = f"[session {seed}-{random.getrandbits(48):x}]"
    body = filler(seed, n_chars)
    if needle:
        mid = len(body) // 2
        body = body[:mid] + f" The secret passkey is {needle}. Remember it. " + body[mid:]
        ask = "What is the secret passkey stated in the text above? Answer with the number only."
    else:
        ask = "Reply with the single word OK."
    return f"{nonce}\n{body}\n\n{ask}"


def call(prompt, max_tokens, timeout=900, messages=None, extra=None):
    p = {"model": "glm-5.3-flash", "messages": messages or [{"role": "user", "content": prompt}],
         "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
         "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"reasoning_effort": "low"}}
    if extra:
        p.update(extra)
    t0 = time.time(); ttft = None; usage = None; text = ""; content = ""; err = None
    try:
        req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(p).encode(), headers=H)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                j = json.loads(d)
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    delta = ch.get("delta") or {}
                    content += delta.get("content") or ""
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece and ttft is None:
                        ttft = time.time() - t0
                    text += piece
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:300]}"
    return {"ttft_s": None if ttft is None else round(ttft, 2), "wall_s": round(time.time() - t0, 2),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"), "error": err, "text_tail": text[-200:], "content": content[-2000:]}


def calibrate():
    global CHARS_PER_TOK
    CHARS_PER_TOK = 4.0
    r = call(build(0, 8000), 4)
    pt = r.get("prompt_tokens")
    if not pt:
        raise SystemExit(f"calibration failed: {r}")
    n_chars = len(build(0, 8000))
    CHARS_PER_TOK = n_chars / pt
    return {"calib_prompt_tokens": pt, "chars_per_tok": round(CHARS_PER_TOK, 4)}


def scrape(t_start, t_end):
    if not CONT:
        return {}
    try:
        logs = subprocess.run(["docker", "logs", CONT], capture_output=True, text=True, timeout=120)
        lines = (logs.stdout + logs.stderr).replace("\r", "\n").splitlines()
    except Exception as e:  # noqa: BLE001
        return {"scrape_error": str(e)}
    mx = {"running": 0, "queue": 0, "usage": 0.0, "retract": 0, "lines": 0, "new_tok": 0, "cached_tok": 0}
    for ln in lines:
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", ln)
        if not m:
            continue
        ts = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))  # container logs are UTC
        if ts < t_start - 1 or ts > t_end + 1:
            continue
        mx["lines"] += 1
        if "retract" in ln.lower() and "server_args" not in ln:
            mx["retract"] += 1
        for key, pat in (("running", r"#running-req: (\d+)"), ("queue", r"#queue-req: (\d+)")):
            mm = re.search(pat, ln)
            if mm:
                mx[key] = max(mx[key], int(mm.group(1)))
        if "Prefill batch" in ln:
            a = re.search(r"#new-token: (\d+)", ln); b = re.search(r"#cached-token: (\d+)", ln)
            mx["new_tok"] += int(a.group(1)) if a else 0; mx["cached_tok"] += int(b.group(1)) if b else 0
        mm = re.search(r"full token usage: ([0-9.]+)", ln) or re.search(r"token usage: ([0-9.]+)", ln)
        if mm:
            mx["usage"] = max(mx["usage"], float(mm.group(1)))
    return mx


def step(name, n, tok, max_tokens, needle=None, timeout=900, extra=None, ask=None):
    res = [None] * n
    def worker(i):
        pr = build(f"{name}-{i}", tok, needle)
        if ask:
            pr = pr.rsplit("\n\n", 1)[0] + "\n\n" + ask
        res[i] = call(pr, max_tokens, timeout, extra=extra)
    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ths:
        t.start(); time.sleep(0.2)
    for t in ths:
        t.join()
    t1 = time.time()
    time.sleep(3)
    s = {"step": name, "n": n, "target_tokens": tok, "max_tokens": max_tokens, "wall_s": round(t1 - t0, 1),
         "requests": res, "server": scrape(t0, t1 + 2)}
    if needle:
        s["needle_found"] = [str(needle) in (r.get("text_tail") or "") for r in res]
    ok = sum(1 for r in res if not r["error"])
    print(f"[kvcap] {name}: {ok}/{n} ok wall {s['wall_s']}s server {s['server']} "
          f"ttft {[r['ttft_s'] for r in res]}", flush=True)
    return s



def retention(name, n, tok):
    """Agent-shaped starvation probe: n sessions of ~tok tokens. Turn 1 sends each (sequential), turn 2 resends each
    with the same prefix + a new user line, same order. A hit re-uses the cached prefix (low TTFT, server cached-token);
    a miss re-prefills it. LRU + same order => when the pool holds fewer than n, misses cascade."""
    seeds = [f"{name}-{i}-{random.getrandbits(32):x}" for i in range(n)]
    prompts = [build(s, tok) for s in seeds]
    t0 = time.time(); t1r = []
    for pr in prompts:
        t1r.append(call(pr, 64))
    tm = time.time(); t2r = []
    for pr, r1 in zip(prompts, t1r):
        msgs = [{"role": "user", "content": pr}, {"role": "assistant", "content": r1.get("content") or "OK"},
                {"role": "user", "content": "Now reply with the single word YES."}]
        t2r.append(call(None, 16, messages=msgs))
    t1 = time.time(); time.sleep(3)
    s = {"step": name, "n": n, "target_tokens": tok, "turn1": t1r, "turn2": t2r,
         "turn1_server": scrape(t0, tm), "turn2_server": scrape(tm, t1 + 2),
         "turn2_ttft": [r["ttft_s"] for r in t2r], "turn1_ttft": [r["ttft_s"] for r in t1r]}
    sv = s["turn2_server"]; tot = (sv.get("new_tok", 0) + sv.get("cached_tok", 0)) or 1
    s["turn2_cached_frac"] = round(sv.get("cached_tok", 0) / tot, 3)
    print(f"[kvcap] {name}: turn2 cached_frac {s['turn2_cached_frac']} turn2 ttft {s['turn2_ttft']} "
          f"turn1 ttft {s['turn1_ttft']} server2 {sv}", flush=True)
    return s


def main():
    out = {"base": BASE, "cont": CONT, "started": time.strftime("%F %T %Z")}
    out.update(calibrate())
    print(f"[kvcap] calibrated {out}", flush=True)
    steps = [step("S1-6x200K", 6, 200_000, 128),
             step("S2-12x8K-512forced", 12, 8_000, 512, extra={"ignore_eos": True}, ask="Write a long essay about the history of rivers."),
             retention("R1-6x150K", 6, 150_000),
             retention("R2-10x150K", 10, 150_000)]
    pool = int(os.getenv("POOL", "0") or 0)
    if pool and pool < 1_000_000:
        steps.append({"step": "S3-1x960K-needle", "skipped": f"pool {pool} < ~965K prompt; cannot hold one 960K context"})
        print(f"[kvcap] S3-1x960K-needle: SKIPPED pool {pool} too small", flush=True)
    else:
        steps.append(step("S3-1x960K-needle", 1, 960_000, 512, needle=random.randint(10_000_000, 99_999_999), timeout=1500))
    out["pool_tokens"] = pool
    out["steps"] = steps
    out["ended"] = time.strftime("%F %T %Z")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[kvcap] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

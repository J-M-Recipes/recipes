#!/usr/bin/env python3
"""needle_ladder.py OUT.jsonl — "usable context" ladder, three variants per rung:
  single      : one exact-recall key at 50% depth (baseline; overstates usable ctx)
  multi       : 5 keys at 10/30/50/70/90% depth, ALL must return, in order
  distractor  : the real key plus 4 decoys with the same name and near-miss values;
                the real one is marked by a one-sentence rule stated at the very top
Per rung record prompt_tokens, TTFT-ish wall (non-stream; prefill dominates), prefill tok/s.
Random-word filler defeats prefix cache. Stops climbing when a variant fails twice on a rung
(one retry with a new seed) or on server error. Thinking disabled so the answer is direct.
"""
import json, os, random, sys, time, urllib.request, uuid

base = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1").rstrip("/"); model = os.getenv("MODEL", "glm-5.3-big")
H = {"Authorization": f"Bearer {os.environ['API_KEY']}", "Content-Type": "application/json"}
RUNGS = [int(x) for x in os.getenv("RUNGS", "8000 32000 64000 128000 192000 240000").split()]
TOK_PER_WORD = float(os.getenv("TOK_PER_WORD", "1.4"))
WORDS = ("apple river stone cloud iron velvet copper meadow lantern orbit cedar prism harbor tundra quartz fable "
         "signal ember canyon ledger falcon marble willow tunnel saffron glacier anchor pixel comet barley").split()
NOTHINK = {}
out = open(sys.argv[1], "a")


def render(user_text):
    # Same render the chat template produces for effort=low, with the think block closed so the
    # model answers directly (template has no enable_thinking switch; only low/high/max effort).
    return ("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>" + user_text + "<|assistant|><think></think>")


def post(p, timeout=3600):
    t0 = time.monotonic()
    body = {"model": p["model"], "prompt": render(p["messages"][0]["content"]), "temperature": 0,
            "max_tokens": p["max_tokens"], "stop": ["<|user|>", "<|observation|>"]}
    req = urllib.request.Request(base + "/completions", data=json.dumps(body).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    # normalise to the chat shape the caller expects
    d["choices"][0]["message"] = {"content": d["choices"][0].get("text", "")}
    return d, time.monotonic() - t0


def filler(n_words, rnd):
    return " ".join(rnd.choice(WORDS) for _ in range(n_words))


def build(variant, target, rnd):
    words = int(target / TOK_PER_WORD)
    if variant == "single":
        v = uuid.uuid4().hex[:10]
        body = filler(words // 2, rnd) + f"\nThe access code is KEY_{v}.\n" + filler(words // 2, rnd)
        q = "\n\nQuestion: What is the access code? Reply with just the code."
        return body + q, [v], "ordered"
    if variant == "multi":
        vals = [uuid.uuid4().hex[:8] for _ in range(5)]
        depths = [0.10, 0.30, 0.50, 0.70, 0.90]; parts = []; last = 0
        for d, v in zip(depths, vals):
            cut = int(words * d); parts.append(filler(cut - last, rnd)); last = cut
            parts.append(f"\nCheckpoint {len(parts)//2 + 1} token is CP_{v}.\n")
        parts.append(filler(words - last, rnd))
        q = "\n\nQuestion: List the five checkpoint tokens in the order they appear. Reply as 'CP_1=<v> CP_2=<v> CP_3=<v> CP_4=<v> CP_5=<v>' only."
        return "".join(parts) + q, vals, "ordered"
    if variant == "distractor":
        real = uuid.uuid4().hex[:8]
        decoys = [real[:5] + uuid.uuid4().hex[:3] for _ in range(4)]  # share 5-char prefix with the real one
        entries = [("blue", real)] + [(c, d) for c, d in zip(["red", "green", "amber", "grey"], decoys)]
        rnd.shuffle(entries)
        head = "Rule: several vault codes appear below, each tagged with a colour. Only the code tagged blue is valid.\n"
        parts = []; last = 0
        for i, (c, v) in enumerate(entries):
            cut = int(words * (0.15 + 0.17 * i)); parts.append(filler(cut - last, rnd)); last = cut
            parts.append(f"\nVault code ({c}): VC_{v}\n")
        parts.append(filler(words - last, rnd))
        q = "\n\nQuestion: What is the valid vault code? Reply with just the code."
        return head + "".join(parts) + q, [real], "single-not-decoys"
    raise ValueError(variant)


def check(text, expect, mode, prompt):
    t = text.lower()
    if mode == "single-not-decoys":
        decoys = [l.split("VC_")[1].strip() for l in prompt.splitlines() if "Vault code (" in l and "(blue)" not in l]
        return expect[0] in t and not any(d.lower() in t for d in decoys)
    pos = [t.find(v) for v in expect]
    return all(p >= 0 for p in pos) and pos == sorted(pos)


for target in RUNGS:
    rung_fail = False
    for variant in ("single", "multi", "distractor"):
        result = None
        for attempt in range(2):
            rnd = random.Random(random.randrange(10**9))
            prompt, expect, mode = build(variant, target, rnd)
            try:
                r, dt = post({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 96, **NOTHINK})
                c = r["choices"][0]["message"].get("content") or ""; pt = r["usage"]["prompt_tokens"]; ct = r["usage"]["completion_tokens"]
                ok = check(c, expect, mode, prompt)
                result = {"lane": os.getenv("LANE", "?"), "target": target, "variant": variant, "attempt": attempt, "prompt_tokens": pt,
                          "completion_tokens": ct, "wall_s": round(dt, 2), "prefill_tok_s_upper": round(pt / dt), "pass": ok,
                          "content": c.strip()[:120]}
            except Exception as e:
                result = {"lane": os.getenv("LANE", "?"), "target": target, "variant": variant, "attempt": attempt, "error": f"{type(e).__name__}: {str(e)[:200]}", "pass": False}
            out.write(json.dumps(result) + "\n"); out.flush()
            print(f"RUNG {target:>7} {variant:>10} a{attempt} pt={result.get('prompt_tokens')} wall={result.get('wall_s')}s "
                  f"prefill<={result.get('prefill_tok_s_upper')} tok/s {'PASS' if result['pass'] else 'FAIL'} {result.get('content', result.get('error'))!r}", flush=True)
            if result["pass"] or "error" in result:
                break
        if not result["pass"]:
            rung_fail = True
            if "error" in result:
                sys.exit(1)
    if rung_fail and os.getenv("STOP_ON_FAIL", "1") == "1":
        print(f"STOP: rung {target} failed at least one variant", flush=True); break

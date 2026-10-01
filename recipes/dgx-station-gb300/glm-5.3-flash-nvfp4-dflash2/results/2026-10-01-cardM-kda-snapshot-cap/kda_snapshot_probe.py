#!/usr/bin/env python3
"""KDA snapshot-eviction probe (Card M, 2026-10-01). Stdlib only.

Reproduces HelixML's 2026-09-26 finding on our box: on GLM-5.3-Flash a prefix-cache hit needs a KDA
state snapshot at the divergence point, and the snapshot pool (max_mamba_cache_size slots, shared with
running requests) is far scarcer than the KV token pool. One long prompt writes ~ceil(N/8192) snapshots
and LRU-evicts every other session's snapshots while the token pool looks empty.

Phases (each starts with /flush_cache):
  A  prime      : send S sessions of ~SESSION_TOK tokens once (cold by construction)
  B  warm       : re-send all S          -> expect cached_tokens ~= prompt_tokens
  C  evict      : send ONE long prompt of ~LONG_TOK tokens
  D  after-long : re-send all S          -> Helix: cold on the stock engine, warm with a per-path cap
  E  branch     : for each S, send a prompt sharing the first BRANCH_FRAC of it, then diverging
                  -> measures the cap's cost (resume point = nearest surviving snapshot)
  F  loop       : LOOP_N sessions re-sent in a fixed loop (worst case for LRU) -> hit count
Every request: max_tokens=1, thinking off, temperature 0. Wall time of a 1-token request ~= TTFT.
cached_tokens comes from usage.prompt_tokens_details.cached_tokens (SGLang meta_info).

Output: one JSON line per request to $OUT (append) + a summary block to stdout. TAG labels the arm.
"""
import json, os, random, sys, time, urllib.request

BASE = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1").rstrip("/")
ROOT = BASE[: -len("/v1")] if BASE.endswith("/v1") else BASE
MODEL = os.getenv("MODEL", "glm-5.3-flash")
TAG = os.getenv("TAG", "x")
OUT = os.getenv("OUT", f"probe-{TAG}.jsonl")
SESSIONS = int(os.getenv("SESSIONS", "6"))
SESSION_TOK = int(os.getenv("SESSION_TOK", "20000"))
LONG_TOK = int(os.getenv("LONG_TOK", "300000"))
BRANCH_FRAC = float(os.getenv("BRANCH_FRAC", "0.7"))
LOOP_N = int(os.getenv("LOOP_N", "12"))
REP = os.getenv("REP", "0")
SEED = int(os.getenv("SEED", "20261001"))
H = {"Authorization": "Bearer x", "Content-Type": "application/json"}

WORDS = ("ledger failure engineer station grace blackwell kernel snapshot radix prefix token decode prefill "
         "chunk boundary state linear attention hybrid expert router slot eviction window measure receipt "
         "contract gate parity drift bracket revert nightly release daily lane queue batch stream seat").split()


def text_of_tokens(rng, n_tok):
    """~1.3 words per token for this vocabulary on the GLM tokenizer; calibrated at runtime (see calib)."""
    n_words = int(n_tok * WPT)
    out = []
    i = 0
    while len(out) < n_words:
        out.append(f"{rng.choice(WORDS)} {rng.choice(WORDS)} {rng.randint(100, 99999)}.")
        i += 1
    return " ".join(out)


def chat(content, max_tokens=1):
    p = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
         "temperature": 0.0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(p).encode(), headers=H)
    t0 = time.monotonic()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.monotonic() - t0
    u = r.get("usage", {})
    pt = u.get("prompt_tokens", 0)
    ct = (u.get("prompt_tokens_details") or {}).get("cached_tokens", None)
    return pt, ct, dt


def flush():
    try:
        urllib.request.urlopen(urllib.request.Request(ROOT + "/flush_cache", method="POST"), timeout=60).read()
    except Exception as e:  # noqa
        print(f"[{TAG}] flush_cache failed: {e}", flush=True)
    time.sleep(2)


def rec(phase, idx, pt, ct, dt, extra=None):
    row = {"tag": TAG, "rep": REP, "phase": phase, "idx": idx, "prompt_tokens": pt, "cached_tokens": ct,
           "wall_s": round(dt, 3), "t": time.time()}
    if extra:
        row.update(extra)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    frac = (ct / pt) if (ct is not None and pt) else None
    print(f"[{TAG}] {phase:10s} #{idx:2d} prompt={pt:7d} cached={ct!s:>7} ({'' if frac is None else f'{frac:.1%}'}) {dt:6.2f}s", flush=True)
    return frac


# --- calibrate words-per-token on this tokenizer ---
WPT = 1.3
rng = random.Random(SEED)
probe = text_of_tokens(rng, 2000)
pt, ct, dt = chat(probe)
if ct is None:
    print(f"[{TAG}] WARNING: cached_tokens not reported in usage; wall time is the only signal", flush=True)
WPT = WPT * (2000 / max(pt, 1))
print(f"[{TAG}] calib: 2000-token target -> {pt} tokens; WPT now {WPT:.3f}", flush=True)

rng = random.Random(SEED + 7)
sessions = [f"Session {i}. Read this record and reply with the single word OK.\n" + text_of_tokens(rng, SESSION_TOK)
            for i in range(max(SESSIONS, LOOP_N))]
long_prompt = "Long record. Reply with the single word OK.\n" + text_of_tokens(rng, LONG_TOK)

summary = {}

# A prime / B warm / C evict / D after-long
flush()
for i in range(SESSIONS):
    rec("A_prime", i, *chat(sessions[i]))
fr = [rec("B_warm", i, *chat(sessions[i])) for i in range(SESSIONS)]
summary["B_warm_hit_frac"] = fr
pt, ct, dt = chat(long_prompt)
rec("C_long", 0, pt, ct, dt)
summary["C_long_tokens"] = pt
summary["C_long_s"] = round(dt, 2)
fr = [rec("D_afterlong", i, *chat(sessions[i])) for i in range(SESSIONS)]
summary["D_afterlong_hit_frac"] = fr
summary["D_afterlong_cold"] = sum(1 for x in fr if x is not None and x < 0.5)

# E branch: shared prefix then divergence (fresh cache, sessions primed first)
flush()
for i in range(SESSIONS):
    chat(sessions[i])
rng2 = random.Random(SEED + 99)
fr = []
for i in range(SESSIONS):
    s = sessions[i]
    cut = int(len(s) * BRANCH_FRAC)
    cut = s.rfind(" ", 0, cut)
    branch = s[:cut] + " BRANCH " + text_of_tokens(rng2, int(SESSION_TOK * (1 - BRANCH_FRAC)))
    fr.append(rec("E_branch", i, *chat(branch), extra={"branch_frac": BRANCH_FRAC}))
summary["E_branch_hit_frac"] = fr

# F loop: LOOP_N sessions, prime once, then re-send in order; count hits
flush()
for i in range(LOOP_N):
    chat(sessions[i])
fr = [rec("F_loop", i, *chat(sessions[i])) for i in range(LOOP_N)]
summary["F_loop_hits"] = sum(1 for x in fr if x is not None and x >= 0.5)
summary["F_loop_n"] = LOOP_N

print(f"[{TAG}] SUMMARY " + json.dumps(summary), flush=True)
with open(OUT + ".summary.json", "a") as f:
    f.write(json.dumps({"tag": TAG, "rep": REP, **summary}) + "\n")

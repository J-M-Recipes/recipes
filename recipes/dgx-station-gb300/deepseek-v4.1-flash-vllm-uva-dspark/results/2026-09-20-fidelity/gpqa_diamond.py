#!/usr/bin/env python3
"""gpqa_diamond.py — GPQA-Diamond (Idavidrein/gpqa, 198 q) against an OpenAI-compatible endpoint. Run by us.

Sanity anchor for the recipe card, not a leaderboard claim. Settings are stated in the output and must be quoted
with the number. Default: reasoning ON (deepseek_v41 reasoning parser on the lane), temperature 0, choice order
shuffled deterministically per question (seed 0), answer extracted from the LAST "Answer: X" / "(X)" pattern in the
visible content; a response with no parseable letter counts as WRONG and is listed.

Env: BASE_URL (default http://127.0.0.1:30006/v1) MODEL (dsv41-flash-uva) API_KEY (none) CONC (16) MAXTOK (16384)
     TEMP (0) TAG (label) OUT (jsonl) LIMIT (0 = all 198) THINKING (1)
Usage: python3 gpqa_diamond.py /path/to/gpqa_diamond.csv
"""
from __future__ import annotations
import csv, json, os, random, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
import urllib.request

BASE = os.getenv("BASE_URL", "http://127.0.0.1:30006/v1")
MODEL = os.getenv("MODEL", "dsv41-flash-uva")
KEY = os.getenv("API_KEY", "none")
CONC = int(os.getenv("CONC", "16"))
MAXTOK = int(os.getenv("MAXTOK", "16384"))
TEMP = float(os.getenv("TEMP", "0"))
TAG = os.getenv("TAG", "gpqa")
OUT = os.getenv("OUT", f"gpqa-{TAG}.jsonl")
LIMIT = int(os.getenv("LIMIT", "0"))
THINK = os.getenv("THINKING", "1") == "1"

SYS = ("You are answering a multiple-choice science question. Think it through, then finish with a single line "
       "of the form 'Answer: X' where X is one of A, B, C, D.")

def build(row: dict, i: int):
    opts = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    rng = random.Random(i)  # deterministic shuffle per question index
    order = [0, 1, 2, 3]; rng.shuffle(order)
    letters = "ABCD"
    gold = letters[order.index(0)]
    body = row["Question"].strip() + "\n\n" + "\n".join(f"({letters[k]}) {opts[o].strip()}" for k, o in enumerate(order))
    return body, gold

ANS_RE = re.compile(r"Answer\s*:\s*\(?([ABCD])\)?", re.I)
PAREN_RE = re.compile(r"\(([ABCD])\)")

def extract(text: str):
    m = ANS_RE.findall(text or "")
    if m: return m[-1].upper()
    m = PAREN_RE.findall((text or "")[-400:])
    if m: return m[-1].upper()
    return None

def ask(i: int, row: dict):
    body, gold = build(row, i)
    payload = {"model": MODEL, "temperature": TEMP, "max_tokens": MAXTOK,
               "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": body}]}
    if not THINK:
        payload["chat_template_kwargs"] = {"thinking": False}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            j = json.load(r)
        ch = j["choices"][0]; msg = ch["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        pred = extract(content) or extract(reasoning[-600:] if reasoning else "")
        usage = j.get("usage", {})
        rec = {"i": i, "gold": gold, "pred": pred, "ok": pred == gold, "finish": ch.get("finish_reason"),
               "completion_tokens": usage.get("completion_tokens"), "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               "secs": round(time.time() - t0, 1), "tail": content[-160:]}
    except Exception as ex:
        rec = {"i": i, "gold": gold, "pred": None, "ok": False, "error": str(ex)[:200], "secs": round(time.time() - t0, 1)}
    return rec

def main():
    rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
    if LIMIT: rows = rows[:LIMIT]
    lock = threading.Lock(); done = 0; t0 = time.time()
    with open(OUT, "w") as f, ThreadPoolExecutor(CONC) as ex:
        for rec in ex.map(lambda p: ask(*p), enumerate(rows)):
            rec.update({"tag": TAG, "model": MODEL, "temp": TEMP, "thinking": THINK, "maxtok": MAXTOK})
            f.write(json.dumps(rec) + "\n"); f.flush()
            with lock:
                done += 1
                if done % 20 == 0: print(f"[{TAG}] {done}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    recs = [json.loads(l) for l in open(OUT)]
    ok = sum(r["ok"] for r in recs); n = len(recs)
    noparse = [r["i"] for r in recs if r["pred"] is None]
    trunc = sum(1 for r in recs if r.get("finish") == "length")
    rt = [r["reasoning_tokens"] for r in recs if r.get("reasoning_tokens")]
    print(f"[{TAG}] GPQA-Diamond n={n} correct={ok} acc={ok/n*100:.1f}% (95% CI ±{196*(ok/n*(1-ok/n)/n)**0.5:.1f}) "
          f"no-parse={len(noparse)} truncated={trunc} thinking={THINK} T={TEMP} maxtok={MAXTOK} "
          f"reasoning_tok mean={sum(rt)/len(rt):.0f} p95={sorted(rt)[int(len(rt)*.95)-1] if rt else 0} wall={time.time()-t0:.0f}s")
    if noparse: print(f"[{TAG}] no-parse idx: {noparse[:30]}")

if __name__ == "__main__":
    main()

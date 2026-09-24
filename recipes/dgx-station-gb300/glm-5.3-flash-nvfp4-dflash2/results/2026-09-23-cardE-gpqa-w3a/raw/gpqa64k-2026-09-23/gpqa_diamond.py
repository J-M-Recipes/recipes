#!/usr/bin/env python3
"""gpqa_diamond.py — GPQA-Diamond (Idavidrein/gpqa, 198 q) against an OpenAI-compatible endpoint. Run by us.

Sanity anchor for the recipe card, not a leaderboard claim. Settings are stated in the output and must be quoted
with the number. Default: reasoning ON, temperature 0, choice order shuffled deterministically per question
(seed = question index), answer extracted from the LAST "Answer: X" / "(X)" pattern in the visible content;
a response with no parseable letter counts as WRONG and is listed.

Env: BASE_URL (default http://127.0.0.1:30006/v1) MODEL (dsv41-flash-uva) API_KEY (none) CONC (16) MAXTOK (16384)
     TEMP (0) TAG (label) OUT (jsonl) LIMIT (0 = all 198) THINKING (1) CTK (JSON chat_template_kwargs)
     IDX (optional: comma-separated question indices, or @path to a file of indices; subset rerun — the
          question index i, and therefore the choice shuffle, is the ORIGINAL index, so records merge 1:1)
Usage: python3 gpqa_diamond.py /path/to/gpqa_diamond.csv

2026-09-23 (Milo): added IDX subset rerun; summary no longer divides by zero when the API reports no
reasoning-token count (GLM-5.3 via SGLang glm45 parser returns none).
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
CTK = json.loads(os.getenv("CTK")) if os.getenv("CTK") else None  # GLM: {"reasoning_effort":"high"} etc.
IDX_RAW = os.getenv("IDX", "")

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
    if CTK is not None:
        payload["chat_template_kwargs"] = CTK
    elif not THINK:
        payload["chat_template_kwargs"] = {"thinking": False}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
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

def parse_idx(raw: str):
    if not raw:
        return None
    if raw.startswith("@"):
        raw = open(raw[1:]).read()
    return sorted({int(x) for x in re.split(r"[,\s]+", raw.strip()) if x})

def main():
    rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
    idx = parse_idx(IDX_RAW)
    if idx is not None:
        items = [(i, rows[i]) for i in idx]
    else:
        items = list(enumerate(rows[:LIMIT] if LIMIT else rows))
    lock = threading.Lock(); done = 0; t0 = time.time()
    with open(OUT, "w") as f, ThreadPoolExecutor(CONC) as ex:
        for rec in ex.map(lambda p: ask(*p), items):
            rec.update({"tag": TAG, "model": MODEL, "temp": TEMP, "thinking": THINK, "ctk": CTK, "maxtok": MAXTOK})
            f.write(json.dumps(rec) + "\n"); f.flush()
            with lock:
                done += 1
                if done % 5 == 0: print(f"[{TAG}] {done}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    recs = [json.loads(l) for l in open(OUT)]
    ok = sum(r["ok"] for r in recs); n = len(recs)
    noparse = [r["i"] for r in recs if r["pred"] is None]
    trunc = sum(1 for r in recs if r.get("finish") == "length")
    errs = sum(1 for r in recs if r.get("error"))
    rt = [r["reasoning_tokens"] for r in recs if r.get("reasoning_tokens")]
    ct = sorted(r["completion_tokens"] for r in recs if r.get("completion_tokens"))
    rt_s = f"reasoning_tok mean={sum(rt)/len(rt):.0f}" if rt else "reasoning_tok n/a"
    ct_s = f"completion_tok median={ct[len(ct)//2]} max={ct[-1]}" if ct else "completion_tok n/a"
    print(f"[{TAG}] GPQA-Diamond n={n} correct={ok} acc={ok/n*100:.1f}% (95% CI ±{196*(ok/n*(1-ok/n)/n)**0.5:.1f}) "
          f"no-parse={len(noparse)} truncated={trunc} errors={errs} thinking={THINK} T={TEMP} maxtok={MAXTOK} "
          f"{rt_s} {ct_s} wall={time.time()-t0:.0f}s", flush=True)
    if noparse: print(f"[{TAG}] no-parse idx: {noparse[:80]}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""kv3_cells.py — Card KV3 full-battery cell drivers (stdlib only). Box path: ~/dsv41/m3/kv3/kv3_cells.py.

Adapted from scratch/needle128k.py (filler, chars ~= tokens x 6.15) and scratch/kv3_smoke.py (metrics names,
/reset_prefix_cache). Every subcommand writes ONE json (--out) and prints one summary line; rc 0 = ran (the verdict
judges), rc 1 = the cell could not be measured (json still written with "error").

  cold64k   --out F [--n 3] [--tokens 64000]      fresh random doc per probe (SystemRandom seed + nonce at byte 0,
                                                  so no prefix reuse incl. the host tier); max_tokens=1; wall = TTFT proxy;
                                                  probe 0 is recorded and DROPPED (kept = probes 1..n-1).
  warm      --out F [--tokens 8000] [--max-tokens 256]
                                                  ONE fixed prompt (seed 20261024), temp 0, ignore_eos + min_tokens=max_tokens
                                                  (falls back to plain fixed max_tokens if the server rejects them);
                                                  r1 then r2 immediately; streamed; decode tok/s = (n-1)/(t_last-t_first).
  needle    --out F [--tokens 950000] [--timeout 2700]
                                                  1M-class needle at 37%; one retry scaled down if the server reports
                                                  the context limit; recall + wall + prompt_tokens.
  evict     --out F [--lengths 8205,33010]        K1: P -> metrics -> POST /reset_prefix_cache -> P; hit/load deltas,
                                                  byte-identical outputs at temp 0, recall, t1/t2.
  evictctl  --out F [--lengths 8205,33010]        K0/K0R: same docs, P once (no reset) -> output for cross-arm info.
Env: KV3_BASE (default http://127.0.0.1:30006), KV3_MODEL (default dsv41-flash-uva).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("KV3_BASE", "http://127.0.0.1:30006")
MODEL = os.environ.get("KV3_MODEL", "dsv41-flash-uva")
CPT = 6.15  # chars per token for this filler (needle128k.py)
WORDS = "the quick brown fox jumps over a lazy dog while seventeen quiet engineers measure everything twice".split()
SMOKE_WORDS = "alpha river stone cloud method orbit kernel garden velvet copper".split()
MAXLEN = 1048576


def filler(rng, target_tokens, words=WORDS, k=120):
    chunks, total = [], 0
    target = int(target_tokens * CPT)
    while total < target:
        c = " ".join(rng.choices(words, k=k)) + ".\n"
        chunks.append(c)
        total += len(c)
    return chunks


def post(path, body, timeout):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(content, max_tokens, timeout, extra=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    body.update(extra or {})
    t0 = time.time()
    r = json.loads(post("/v1/chat/completions", body, timeout).read())
    dt = time.time() - t0
    msg = r["choices"][0]["message"]
    return (msg.get("content") or "").strip(), r.get("usage", {}), dt


def http_err(e):
    try:
        return f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}"
    except Exception:  # noqa: BLE001
        return repr(e)


def metrics():
    s = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    out = {}
    for ln in s.splitlines():
        if ln.startswith("#"):
            continue
        low = ln.lower()
        if "offload" in low or "external_prefix" in low or "prefix_cache" in low:
            k = ln.split("{")[0].split(" ")[0]
            try:
                out[k] = out.get(k, 0.0) + float(ln.rsplit(" ", 1)[1])
            except Exception:  # noqa: BLE001
                pass
    return out


def reset_cache():
    req = urllib.request.Request(BASE + "/reset_prefix_cache", data=b"", method="POST")
    return urllib.request.urlopen(req, timeout=30).status


def write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- cold64k
def cmd_cold64k(a):
    probes = []
    for i in range(a.n):
        seed = random.SystemRandom().randrange(1 << 62)
        rng = random.Random(seed)
        nonce = "Document %016x." % seed
        doc = nonce + "\n" + "".join(filler(rng, a.tokens))
        q = doc + "\n\nIn one word, what animal is mentioned most in the document above?"
        rec = {"i": i, "seed": seed, "dropped": i == 0, "target_tokens": a.tokens}
        try:
            _, u, dt = chat(q, 1, a.timeout)
            rec.update(prompt_tokens=u.get("prompt_tokens"), wall_s=round(dt, 4), error=None)
        except urllib.error.HTTPError as e:
            rec.update(prompt_tokens=None, wall_s=None, error=http_err(e))
        except Exception as e:  # noqa: BLE001
            rec.update(prompt_tokens=None, wall_s=None, error=repr(e))
        probes.append(rec)
        print(f"[cold64k] probe {i} seed={seed} pt={rec['prompt_tokens']} wall={rec['wall_s']} "
              f"{'(dropped)' if i == 0 else ''} err={rec['error']}", flush=True)
    kept = [p["wall_s"] for p in probes if not p["dropped"] and p["wall_s"] is not None]
    out = {"cell": "cold64k", "n": a.n, "probes": probes,
           "kept_median_wall_s": statistics.median(kept) if kept else None,
           "errors": sum(1 for p in probes if p["error"])}
    write(a.out, out)
    print(json.dumps({k: out[k] for k in ("cell", "n", "kept_median_wall_s", "errors")}), flush=True)
    return 0 if kept else 1


# ---------------------------------------------------------------- warm
def stream_once(content, max_tokens, extra, timeout):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "seed": 42, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    body.update(extra)
    t0 = time.time()
    first = last = None
    usage = {}
    nchunk = 0
    with post("/v1/chat/completions", body, timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            chs = obj.get("choices") or []
            d = (chs[0].get("delta") or {}) if chs else {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                now = time.time()
                first = first or now
                last = now
                nchunk += 1
    t1 = time.time()
    comp = int(usage.get("completion_tokens") or 0)
    gen = (last - first) if (first and last and last > first) else None
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp, "chunks": nchunk,
            "ttft_s": round(first - t0, 4) if first else None, "wall_s": round(t1 - t0, 4),
            "decode_tok_s": round((comp - 1) / gen, 2) if gen and comp > 1 else None}


def cmd_warm(a):
    rng = random.Random(20261024)
    doc = "".join(filler(rng, a.tokens))
    content = (doc + "\n\nIgnore the repetition above. Now write a long, continuous English story about a lighthouse "
               "keeper and a storm. Do not stop early; keep writing.")
    modes = [("ignore_eos+min_tokens", {"ignore_eos": True, "min_tokens": a.max_tokens}), ("fixed_max_tokens", {})]
    out = {"cell": "warm", "max_tokens": a.max_tokens, "target_tokens": a.tokens, "runs": [], "mode": None, "error": None,
           "mode_errors": []}
    for mode, extra in modes:
        try:
            r1 = stream_once(content, a.max_tokens, extra, a.timeout)
            r2 = stream_once(content, a.max_tokens, extra, a.timeout)   # immediately: warm replay (GPU prefix hit)
            out["mode"] = mode
            out["runs"] = [dict(r=1, **r1), dict(r=2, **r2)]
            break
        except urllib.error.HTTPError as e:
            out["mode_errors"].append(f"{mode}: {http_err(e)}")
        except Exception as e:  # noqa: BLE001
            out["mode_errors"].append(f"{mode}: {e!r}")
    if not out["runs"]:
        out["error"] = "; ".join(out["mode_errors"])
    else:
        out["forced_ok"] = all(r["completion_tokens"] == a.max_tokens for r in out["runs"])
        out["r1_decode_tok_s"] = out["runs"][0]["decode_tok_s"]
        out["r2_decode_tok_s"] = out["runs"][1]["decode_tok_s"]
    write(a.out, out)
    print(json.dumps({k: out.get(k) for k in ("cell", "mode", "forced_ok", "r1_decode_tok_s", "r2_decode_tok_s", "error")}),
          flush=True)
    return 0 if out["runs"] else 1


# ---------------------------------------------------------------- needle
def cmd_needle(a):
    key = "ZEBRA-COUPON-%04d" % random.SystemRandom().randrange(10000)
    target = a.tokens
    attempts = []
    out = {"cell": "needle", "target_tokens": a.tokens, "key": key, "recall": False, "error": None}
    for attempt in range(2):
        rng = random.Random(9418 + attempt)
        chunks = filler(rng, target)
        chunks.insert(int(len(chunks) * 0.37), "\nIMPORTANT: the secret coupon code is %s. Remember it.\n" % key)
        q = "".join(chunks) + "\n\nWhat is the secret coupon code mentioned in the document above? Answer with just the code."
        rec = {"attempt": attempt, "target_tokens": target, "chars": len(q)}
        try:
            ans, u, dt = chat(q, 24, a.timeout)
            rec.update(prompt_tokens=u.get("prompt_tokens"), wall_s=round(dt, 2), answer=ans, recall=key in ans, error=None)
            attempts.append(rec)
            out.update(prompt_tokens=rec["prompt_tokens"], wall_s=rec["wall_s"], answer=ans, recall=rec["recall"], error=None)
            break
        except urllib.error.HTTPError as e:
            msg = http_err(e)
            rec.update(error=msg)
            attempts.append(rec)
            out["error"] = msg
            # context overflow -> scale down once (filler ratio is an estimate)
            import re
            mm = re.search(r"maximum context length is (\d+)", msg)
            lim = int(mm.group(1)) if mm else MAXLEN
            m = re.search(r"(\d+) input tokens|resulted in (\d+) tokens|request has (\d+) tokens", msg)
            have = int(next(g for g in m.groups() if g)) if m else int(target * 1.12)
            target = int(target * (lim - 4096) / max(have, 1) * 0.97)
        except Exception as e:  # noqa: BLE001
            rec.update(error=repr(e))
            attempts.append(rec)
            out["error"] = repr(e)
            break
    out["attempts"] = attempts
    write(a.out, out)
    print(json.dumps({k: out.get(k) for k in ("cell", "prompt_tokens", "wall_s", "answer", "recall", "error")}), flush=True)
    return 0 if out.get("wall_s") is not None else 1


# ---------------------------------------------------------------- evict / evictctl
def evict_doc(target_tok, code, seed):
    rng = random.Random(seed)
    chunks = filler(rng, target_tok, SMOKE_WORDS, 100)
    chunks.insert(len(chunks) // 3, "\nThe secret code is %s.\n" % code)
    return "".join(chunks) + "\n\nWhat is the secret code in the document? Answer with just the code."


def lengths(a):
    return [int(x) for x in a.lengths.split(",") if x.strip()]


def cmd_evict(a, ctl=False):
    out = {"cell": "evictctl" if ctl else "evict", "lengths": [], "error": None}
    try:
        out["metrics_present"] = sorted(metrics().keys())
    except Exception as e:  # noqa: BLE001
        out["metrics_present"] = []
        out["metrics_error"] = repr(e)
    for j, tl in enumerate(lengths(a)):
        code = "CODE-KV3-%d%d%d" % (j, j, j)
        doc = evict_doc(tl, code, 3000 + j)
        rec = {"target": tl, "code": code}
        try:
            o1, u, t1 = chat(doc, 24, a.timeout)
            rec.update(prompt_tokens=u.get("prompt_tokens"), output1=o1, t1=round(t1, 4), recall=code in o1)
            if u.get("prompt_tokens"):
                rec["mod256"] = u["prompt_tokens"] % 256
            if not ctl:
                m1 = metrics()
                rec["reset_status"] = reset_cache()
                time.sleep(1)
                o2, _, t2 = chat(doc, 24, a.timeout)
                m2 = metrics()
                delta = {k: m2.get(k, 0) - m1.get(k, 0) for k in m2 if m2.get(k, 0) != m1.get(k, 0)}
                rec.update(output2=o2, t2=round(t2, 4), identical=(o1 == o2), delta=delta,
                           hit=any("hit" in k.lower() and "external" in k.lower() and v > 0 for k, v in delta.items())
                           or any("offload" in k.lower() and "hit" in k.lower() and v > 0 for k, v in delta.items()),
                           load=any("load" in k.lower() and "offload" in k.lower() and v > 0 for k, v in delta.items()),
                           speedup=round(t1 / t2, 2) if t2 else None)
            rec["error"] = None
        except urllib.error.HTTPError as e:
            rec["error"] = http_err(e)
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
        out["lengths"].append(rec)
        print(f"[{out['cell']}] {json.dumps({k: v for k, v in rec.items() if k not in ('output1', 'output2', 'delta')})}",
              flush=True)
    out["errors"] = sum(1 for r in out["lengths"] if r.get("error"))
    write(a.out, out)
    if ctl:
        summ = {"cell": out["cell"], "n": len(out["lengths"]), "recall_all": all(r.get("recall") for r in out["lengths"]),
                "errors": out["errors"]}
    else:
        summ = {"cell": "evict", "n": len(out["lengths"]), "hits_all": all(r.get("hit") for r in out["lengths"]),
                "identical_all": all(r.get("identical") for r in out["lengths"]),
                "recall_all": all(r.get("recall") for r in out["lengths"]), "errors": out["errors"],
                "speedups": [r.get("speedup") for r in out["lengths"]]}
    print(json.dumps(summ), flush=True)
    return 0 if out["errors"] == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("cold64k"); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=3)
    p.add_argument("--tokens", type=int, default=64000); p.add_argument("--timeout", type=float, default=600)
    p = sp.add_parser("warm"); p.add_argument("--out", required=True); p.add_argument("--tokens", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=256); p.add_argument("--timeout", type=float, default=600)
    p = sp.add_parser("needle"); p.add_argument("--out", required=True); p.add_argument("--tokens", type=int, default=950000)
    p.add_argument("--timeout", type=float, default=2700)
    for c in ("evict", "evictctl"):
        p = sp.add_parser(c); p.add_argument("--out", required=True); p.add_argument("--lengths", default="8205,33010")
        p.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    if a.cmd == "cold64k":
        return cmd_cold64k(a)
    if a.cmd == "warm":
        return cmd_warm(a)
    if a.cmd == "needle":
        return cmd_needle(a)
    return cmd_evict(a, ctl=(a.cmd == "evictctl"))


if __name__ == "__main__":
    sys.exit(main())

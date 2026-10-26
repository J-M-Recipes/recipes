#!/usr/bin/env python3
"""peerbench.py — stdlib port of the DeepSeek-4.1-Flash-1017-BRRR benchmark (src/brrr/bench.py) for the :30006 DSV4.1 lane.

NOTICE / credit: the method, prompt text (CODE_TASK, CODE_SUFFIX), column layout and statistics are from
  DeepSeek-4.1-Flash-1017-BRRR, Copyright 2026 Fraser Price, https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR
  (src/brrr/bench.py), licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
  This file is a modified port (Milo desk, 2026-10-25, for Card BR2). Changes from the original are listed below; the
  original's NOTICE file credits vLLM, Local Inference Lab's vLLM fork and b12x (Apache-2.0) for the serving stack, not this script.

What is reproduced faithfully (upstream bench.py at the HF main revision fetched 2026-10-25, 361 lines):
  - prompt: f"{nonce} " + "\\n\\n".join([CODE_TASK] * repeats) + f"\\n\\n{CODE_SUFFIX}", nonce = secrets.token_hex(8) per stream;
    repeats calibrated ONCE via POST <root>/tokenize {"model", "prompt"}: guess = max(1, round(size*3.6/(len(CODE_TASK)+2))),
    repeats = max(1, round(guess*size/measured)). Prompt sizes are therefore real token counts of the lane's tokenizer (+/- one repeat).
  - per round: c calibrated prompts; pass 1 = all c sent together with max_tokens 1 (TTFT, PP; must not hit the prefix cache);
    spec counters read; pass 2 = the SAME prompts again (now cached) with max_tokens 2048, natural EOS, so all streams start
    decoding together; spec counters read again (delta = this round's spec).
  - request body: messages [{user, prompt}], temperature 1.0, top_p 1.0 (THEIR defaults, sent explicitly), stream, stream_options
    {include_usage, continuous_usage_stats}, chat_template_kwargs {"thinking": false, "enable_thinking": false} (thinking 0 = off).
    No seed, no min_tokens, no ignore_eos, no reasoning_effort (none in theirs).
  - arrivals = SSE chunks carrying usage whose completion_tokens strictly exceeds the running max; TTFT = first arrival - start.
  - TG window (decode_window): start = max first arrival, end = min last arrival over the c streams; edge = median of the
    inter-arrival gaps inside [start, end] / 2; tokens per stream = tokens_at(end+edge) - tokens_at(start+edge); seconds = end-start.
  - row: prompt = round(mean prompt_tokens); TTFT p50 (ms) over pass-1 samples; PP per stream = prompt_tokens/ttft, PP total =
    sum prompt_tokens / sum over rounds of the max ttft; TG per stream = window tokens / window seconds, TG total = sum tokens /
    sum seconds; TG window = median window tokens; accept = f"{1 + accepted/drafts:.2f} ({accepted/draft_tokens:.0%})" from
    vllm:spec_decode_{num_drafts,num_draft_tokens,num_accepted_tokens} (name with _total stripped, summed over labels).
  - "p1/p50/p99 = fastest/median/slowest stream": like theirs, the p1 column holds the 99th percentile value and the p99 column
    the 1st percentile (rates() iterates q = 99, 50, 1). Percentiles use numpy's default linear interpolation, re-implemented.
  - console layout: every cell right-aligned to 11 chars, same COLUMNS order and the same two header lines.
Changes (each deliberate):
  1. stdlib only (box python3 has no httpx/numpy/pydantic/cyclopts): urllib + threads; a threading.Barrier releases the c
     streams of a pass together (asyncio.gather equivalent).
  2. no cool-to-70 C wait: the hottest GPU temperature (nvidia-smi --query-gpu=temperature.gpu, read-only) is logged per round.
  3. CLI: --base-url (http://127.0.0.1:30006/v1; the root for /tokenize and /metrics is base-url minus /v1), --model (theirs
     takes data[0] of /v1/models; we assert --model is served), --rounds (default 2; theirs --repeats 5), --sizes (default
     2048 16384 65536; theirs 65528 for the last), --conc 1 4 8 16, --out JSON (+ <out>.md markdown table, same columns).
  4. prompt_tokens_details: the lane runs WITHOUT --enable-prompt-tokens-details (af7f9488 chat serving returns None then),
     where theirs raises. Here cached_tokens is taken from usage when present; the pass-1 cache-bust check additionally uses
     the vllm:prefix_cache_hits_total delta over pass 1 (> 1024 tokens => the round is rejected, as theirs raises). Both recorded.
  5. hard budget per cell (size x conc) --cell-timeout 1500 s: a deadline is enforced inside every stream (socket timeout +
     deadline check per SSE line). Budget rule for the long pole: before a 64K cell at conc > 4, the cell time is predicted
     from the measured 64K C4 cell (time per round x conc/4 x rounds); if > budget the cell is SKIPPED and the output says
     "64K run at C1 and C4 only (budget)". A cell that still runs out of time reports the rounds it completed (or TIMEOUT).
  6. a failing cell (raise in theirs) is recorded with its error and the bench continues with the next cell.
Output JSON: {"meta": {...sampling, method, deviations}, "cells": [{size, conc, rounds: [...raw...], row: [...], status}], "table_md"}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

MAX_TOKENS = 2048

CODE_TASK = """I have a FastAPI service that receives webhook events from a payment provider and stores them in Postgres with asyncpg. The current handler:

```python
@app.post("/webhooks/payments")
async def payments_webhook(request: Request) -> dict[str, str]:
    payload = await request.json()
    if payload["type"] == "invoice.paid":
        await pool.execute(
            "INSERT INTO invoices (id, amount) VALUES ($1, $2)",
            payload["data"]["id"], payload["data"]["amount"],
        )
    return {"status": "ok"}
```

Problems: the `X-Signature` header is never verified, retried deliveries insert the same invoice twice, and any exception returns a 500 so the provider retries forever.

Rewrite this, returning a single python file: verify the signature with HMAC-SHA256 over the raw body and reject stale timestamps, make inserts idempotent on the event id, dispatch event types through a registry instead of if-chains, and return 2xx for event types we don't handle. Use pydantic models, full type hints, and no bare except. Include the SQL for the idempotency table and a pytest test for the signature check."""
CODE_SUFFIX = (
    "Sorry, the task above is repeated. It is a single "
    "task, so answer it exactly once. Output only well-typed code (no comments or header/footer text) with "
    "no prose before, between, or after them. Your full output will be directly inserted into a code file, so your full response MUST be code only. No prose/comments whatsoever; pure, single python file as output which I will paste in."
)
COLUMNS = ("prompt", "conc", "TTFT p50", *(f"{ph} {st}" for ph in ("PP", "TG") for st in ("p1", "p50", "p99", "total")),
           "TG window", "accept")


PC_HIT_SLACK = 1024   # port: pass-1 prefix_cache_hits delta above this (tokens) = a real cache hit (a 2K prompt hit is ~2K)


class CellTimeout(RuntimeError):
    pass


# ---------------------------------------------------------------- numpy stand-ins
def percentile(values, q):
    """numpy.percentile default ('linear') on a 1-D list"""
    s = sorted(values)
    if not s:
        raise ValueError("percentile of empty list")
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median(values):
    return statistics.median(values)


# ---------------------------------------------------------------- HTTP
class Lane:
    def __init__(self, base_url, model):
        self.base = base_url.rstrip("/")
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base
        self.model = model

    def get(self, path, timeout=30):
        return urllib.request.urlopen(self.root + path, timeout=timeout).read().decode()

    def post_json(self, path, body, timeout=120):
        r = urllib.request.Request(self.root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=timeout).read().decode())

    def counters(self):
        """spec counters exactly as theirs (name before '{' / ' ', '_total' stripped, summed) + prefix-cache hits (port addition)"""
        named = []
        for line in self.get("/metrics").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name = line.partition("{")[0].partition(" ")[0]
            if name.endswith("_total"):
                name = name[: -len("_total")]
            try:
                named.append((name, float(line.rsplit(" ", 1)[1])))
            except ValueError:
                pass
        tot = lambda n: sum(v for k, v in named if k == n)
        return {"drafts": tot("vllm:spec_decode_num_drafts"), "draft_tokens": tot("vllm:spec_decode_num_draft_tokens"),
                "accepted": tot("vllm:spec_decode_num_accepted_tokens"), "pc_hits": tot("vllm:prefix_cache_hits"),
                "pc_queries": tot("vllm:prefix_cache_queries")}


def coding_prompt(nonce, repeats):
    return f"{nonce} " + "\n\n".join([CODE_TASK] * repeats) + f"\n\n{CODE_SUFFIX}"


def calibrated_prompt(lane, size):
    nonce = secrets.token_hex(8)
    guess = max(1, round(size * 3.6 / (len(CODE_TASK) + 2)))
    measured = lane.post_json("/tokenize", {"model": lane.model, "prompt": coding_prompt(nonce, guess)})["count"]
    return coding_prompt(nonce, max(1, round(guess * size / measured)))


def stream(lane, prompt, temperature, top_p, max_tokens, deadline, barrier=None):
    body = {"model": lane.model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "top_p": top_p,
            "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    if max_tokens:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(lane.base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait(timeout=max(1.0, deadline - time.time()))
    start = time.perf_counter()
    reports = []
    sock_to = max(5.0, deadline - time.time())   # a 64K x C16 prefill can take minutes before the first byte
    with urllib.request.urlopen(req, timeout=sock_to) as resp:
        for raw in resp:
            if time.time() > deadline:
                raise CellTimeout("cell budget exceeded mid-stream")
            line = raw.decode().rstrip("\r\n")
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            usage = (json.loads(line[len("data: "):]) or {}).get("usage")
            if usage:
                reports.append((usage, time.perf_counter()))
    arrivals, peak = [], 0
    for usage, at in reports:
        tok = usage.get("completion_tokens") or 0
        if tok > peak:
            arrivals.append({"tokens": tok, "at": at})
        peak = max(peak, tok)
    if not arrivals:
        raise RuntimeError("no tokens generated before the request ended")
    final = reports[-1][0]
    ptd = final.get("prompt_tokens_details")
    return {"prompt_tokens": final.get("prompt_tokens"), "cached_tokens": (ptd or {}).get("cached_tokens"),
            "prompt_tokens_details_present": ptd is not None, "started": start, "arrivals": arrivals,
            "completion_tokens": final.get("completion_tokens")}


def gather(lane, prompts, temperature, top_p, max_tokens, deadline):
    """asyncio.gather equivalent: one thread per stream, released together by a barrier; first error re-raised"""
    res = [None] * len(prompts); err = [None] * len(prompts); bar = threading.Barrier(len(prompts))
    def go(i, p):
        try:
            res[i] = stream(lane, p, temperature, top_p, max_tokens, deadline, bar)
        except Exception as e:  # noqa: BLE001
            err[i] = e
            try:
                bar.abort()
            except Exception:  # noqa: BLE001
                pass
    th = [threading.Thread(target=go, args=(i, p), daemon=True) for i, p in enumerate(prompts)]
    [t.start() for t in th]
    for t in th:
        t.join(timeout=max(0.0, deadline - time.time()) + 5)
    if any(t.is_alive() for t in th):
        raise CellTimeout("stream thread still running at the cell deadline")
    for e in err:
        if e is not None:
            raise e
    return res


def ttft(s):
    return s["arrivals"][0]["at"] - s["started"]


def tokens_at(s, at):
    return max((a["tokens"] for a in s["arrivals"] if a["at"] <= at), default=0)


def decode_window(batch):
    start = max(s["arrivals"][0]["at"] for s in batch)
    end = min(s["arrivals"][-1]["at"] for s in batch)
    gaps = []
    for s in batch:
        ts = [a["at"] for a in s["arrivals"] if start <= a["at"] <= end]
        gaps += [b - a for a, b in zip(ts, ts[1:])]
    if end <= start or not gaps:
        raise RuntimeError("a stream finished before every stream started decoding")
    edge = float(median(gaps)) / 2
    return {"tokens": [tokens_at(s, end + edge) - tokens_at(s, start + edge) for s in batch], "seconds": end - start}


def hottest_gpu():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], text=True, timeout=10)
        return max(int(x) for x in out.split())
    except Exception:  # noqa: BLE001
        return None


def measure(lane, size, c, temperature, top_p, deadline):
    t0 = time.time(); temp_c = hottest_gpu()
    prompts = [calibrated_prompt(lane, size) for _ in range(c)]
    m0 = lane.counters()
    prefill = gather(lane, prompts, temperature, top_p, 1, deadline)
    m1 = lane.counters()
    pc_hits = m1["pc_hits"] - m0["pc_hits"]
    if any(s["cached_tokens"] for s in prefill) or pc_hits > PC_HIT_SLACK:
        raise RuntimeError(f"prefill pass hit the prefix cache (cached_tokens={[s['cached_tokens'] for s in prefill]} "
                           f"prefix_cache_hits_delta={pc_hits})")
    before = m1
    decode = gather(lane, prompts, temperature, top_p, MAX_TOKENS, deadline)
    after = lane.counters()
    spec = {k: after[k] - before[k] for k in ("drafts", "draft_tokens", "accepted")}
    return {"prefill": prefill, "window": decode_window(decode), "spec": spec, "hottest_gpu_c": temp_c,
            "prefill_pc_hits_delta": pc_hits, "decode_cached_tokens": [s["cached_tokens"] for s in decode],
            "decode_completion_tokens": [s["completion_tokens"] for s in decode], "round_s": time.time() - t0}


def rates(values, total):
    return (*(f"{percentile(values, q):,.0f}" for q in (99, 50, 1)), f"{total:,.0f}")


def row(conc, rounds):
    samples = [s for r in rounds for s in r["prefill"]]
    tokens = [n for r in rounds for n in r["window"]["tokens"]]
    spec = {k: sum(r["spec"][k] for r in rounds) for k in ("drafts", "draft_tokens", "accepted")}
    return (
        f"{round(statistics.fmean([s['prompt_tokens'] for s in samples])):,}",
        str(conc),
        f"{percentile([ttft(s) for s in samples], 50) * 1000:,.0f} ms",
        *rates([s["prompt_tokens"] / ttft(s) for s in samples],
               sum(s["prompt_tokens"] for s in samples) / sum(max(ttft(s) for s in r["prefill"]) for r in rounds)),
        *rates([n / r["window"]["seconds"] for r in rounds for n in r["window"]["tokens"]],
               sum(tokens) / sum(r["window"]["seconds"] for r in rounds)),
        f"{median(tokens):,.0f}",
        f"{1 + spec['accepted'] / spec['drafts']:.2f} ({spec['accepted'] / spec['draft_tokens']:.0%})" if spec["drafts"] else "—",
    )


def aligned(cells):
    return "".join(f"{cell:>11}" for cell in cells)


def md_table(cells):
    L = ["| " + " | ".join(COLUMNS) + " | note |", "|" + "---|" * (len(COLUMNS) + 1)]
    for c in cells:
        if c.get("row"):
            L.append("| " + " | ".join(c["row"]) + f" | {c.get('note') or ''} |")
        else:
            L.append(f"| {c['size']:,} (target) | {c['conc']} | " + " | ".join(["—"] * (len(COLUMNS) - 2)) + f" | {c['status']}: {c.get('note') or ''} |")
    return "\n".join(L)


def dump(path, meta, cells, notes, h1, h2):
    meta["notes"] = notes
    md = (f"{h1}\n\n{h2}\n\n" + md_table(cells) + "\n" + ("".join(f"\n- {n}" for n in notes))
          + ("" if meta.get("complete") else "\n- INCOMPLETE: bench still running or killed by the outer timeout") + "\n")
    out = {"meta": meta, "cells": cells, "table_md": md,
           "table_txt": "\n".join([aligned(COLUMNS)] + [aligned(c["row"]) for c in cells if c["row"]])}
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(path + ".tmp", path)
    md_path = re.sub(r"\.json$", "", path) + ".md"
    with open(md_path + ".tmp", "w") as f:
        f.write(md)
    os.replace(md_path + ".tmp", md_path)


def main():
    ap = argparse.ArgumentParser(description="stdlib port of DeepSeek-4.1-Flash-1017-BRRR bench.py (Apache-2.0, Fraser Price)")
    ap.add_argument("--base-url", default="http://127.0.0.1:30006/v1")
    ap.add_argument("--model", default="dsv41-flash-uva")
    ap.add_argument("--rounds", type=int, default=2, help="measured rounds per cell (theirs --repeats, default 5)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048, 16384, 65536])
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--temperature", type=float, default=1.0, help="theirs: 1.0")
    ap.add_argument("--top-p", type=float, default=1.0, help="theirs: 1.0")
    ap.add_argument("--cell-timeout", type=float, default=1500.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lane = Lane(a.base_url, a.model)
    served = [m["id"] for m in json.loads(lane.get("/v1/models"))["data"]]
    if a.model not in served:
        print(f"model {a.model} not served ({served})", file=sys.stderr); return 2
    budget = f"thinking off · gen ≤{MAX_TOKENS} tok to natural EOS"
    h1 = f"{a.model} · code prompt · {budget} · repeats={a.rounds} · no cool wait (hottest GPU logged)"
    h2 = ("tok/s: p1/p50/p99 = fastest/median/slowest stream · PP from a max_tokens=1 pass · TG from a cached second pass, "
          "timed while all streams decode and none has finished")
    print(h1); print(h2); print(aligned(COLUMNS), flush=True)
    meta = {"port_of": "https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/resolve/main/src/brrr/bench.py",
            "license": "Apache-2.0 (c) 2026 Fraser Price; modified port", "base_url": a.base_url, "model": a.model,
            "rounds": a.rounds, "sizes": a.sizes, "conc": a.conc, "cell_timeout_s": a.cell_timeout,
            "sampling": {"temperature": a.temperature, "top_p": a.top_p, "seed": None, "max_tokens_decode": MAX_TOKENS,
                         "max_tokens_prefill": 1, "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
                         "source": "mirrors their defaults (--temperature 1.0, --top-p 1.0 sent explicitly; thinking 0)"},
            "ts_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "header": [h1, h2], "complete": False,
            "prompt_tokens_details_note": "lane runs without --enable-prompt-tokens-details: cached_tokens is null; cache-bust "
                                          "check uses the vllm:prefix_cache_hits_total delta over pass 1 (> 1024 tokens rejects)"}
    cells, c4_round_s = [], {}
    notes = []
    for size in a.sizes:
        for c in a.conc:
            cell = {"size": size, "conc": c, "rounds": [], "row": None, "status": "OK", "note": None}
            big = size >= 65536 - 64
            if big and c > 4 and size in c4_round_s:
                pred = c4_round_s[size] * (c / 4) * a.rounds
                if pred > a.cell_timeout:
                    cell["status"] = "SKIPPED"; cell["note"] = f"predicted {pred:.0f}s > {a.cell_timeout:.0f}s budget: 64K run at C1 and C4 only (budget)"
                    notes.append(cell["note"]); cells.append(cell); print(f"# {size} C{c}: {cell['note']}", flush=True)
                    dump(a.out, meta, cells, notes, h1, h2); continue
            t0 = time.time(); deadline = t0 + a.cell_timeout
            for i in range(a.rounds):
                try:
                    cell["rounds"].append(measure(lane, size, c, a.temperature, a.top_p, deadline))
                except CellTimeout as e:
                    cell["status"] = "TIMEOUT" if not cell["rounds"] else "PARTIAL"; cell["note"] = f"{e} after {len(cell['rounds'])} round(s)"; break
                except Exception as e:  # noqa: BLE001
                    cell["status"] = "ERROR" if not cell["rounds"] else "PARTIAL"; cell["note"] = f"{type(e).__name__}: {e}"[:300]; break
            cell["cell_s"] = round(time.time() - t0, 1)
            if c == 4 and cell["rounds"]:
                c4_round_s[size] = max(r["round_s"] for r in cell["rounds"])
            if cell["rounds"]:
                try:
                    cell["row"] = list(row(c, cell["rounds"]))
                except Exception as e:  # noqa: BLE001
                    cell["status"] = "ERROR"; cell["note"] = f"row: {type(e).__name__}: {e}"[:300]
            if cell["row"]:
                print(aligned(cell["row"]) + (f"   # {cell['status']}: {cell['note']}" if cell["note"] else ""), flush=True)
            else:
                print(f"# {size} C{c}: {cell['status']} {cell['note']}", flush=True)
            cells.append(cell)
            dump(a.out, meta, cells, notes, h1, h2)   # after every cell: an outer timeout keeps the finished cells
    meta["ts_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); meta["complete"] = True
    dump(a.out, meta, cells, notes, h1, h2)
    ok = sum(1 for c in cells if c["status"] == "OK")
    print(json.dumps({"peerbench": a.out, "cells": len(cells), "ok": ok, "skipped": sum(1 for c in cells if c["status"] == "SKIPPED"),
                      "bad": [f"{c['size']}/C{c['conc']}:{c['status']}" for c in cells if c["status"] not in ("OK", "SKIPPED")]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

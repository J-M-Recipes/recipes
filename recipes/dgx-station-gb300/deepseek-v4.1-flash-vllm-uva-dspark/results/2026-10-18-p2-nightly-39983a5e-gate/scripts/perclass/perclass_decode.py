#!/usr/bin/env python3
"""perclass_decode.py — per-class decode wall-clock instrument for the :30006 DSV4.1 lane (stdlib only).

Gate instrument for drafter-side changes (per-class DSpark acceptance is the canary; this is the gate).
Input : perclass_prompts.jsonl (rows {id, cls, source, messages, tools|null, prompt_tokens, ...}; build_perclass_prompts.py).
Request: /v1/chat/completions, stream, temperature 0, chat_template_kwargs {thinking: false} (= agent_fixture_x3.py / replay_c.py;
  no reasoning_effort field is sent, same as those instruments), tools+tool_choice auto when the row carries tools,
  stream_options {include_usage, continuous_usage_stats}.
  forced mode (default): max_tokens = min_tokens = FL[cls], ignore_eos = true -> every arm emits exactly FL tokens per prompt.
  natural mode (--natural): max_tokens = FL[cls] cap, EOS honoured (records natural lengths; doubles as prefix-cache warm pass).
Timing per request: t_first = first chunk whose running usage.completion_tokens >= 1 (ct_first = that count, may be >1 with
  spec decode); t_last = chunk where completion_tokens reaches its final value. decode_tokens = ct_final - ct_first;
  decode tok/s = decode_tokens / (t_last - t_first); TPOT = (t_last - t_first) / decode_tokens. TTFT = t_first - t_send.
Concurrency:
  C1 : prompts of a class one at a time (all prompts).
  C8 : waves of CONC prompts of the same class launched together (full waves only; partial tail wave dropped and recorded).
       wave agg decode tok/s = sum(decode_tokens) / (max t_last - min t_first); class agg = sum over waves / sum of windows.
Classes run sequentially in ORDER (prose structured code shell_ops tool_json = x3 CAT_ORDER). Per class-run: /metrics deltas
  (spec_decode draft/accepted/drafts -> acceptance, acc/step; generation_tokens -> foreign tokens = delta - our tokens;
  prefix_cache hits/queries) and a 0.25 s sampler of num_requests_running/waiting (max_running > expected => foreign traffic).
Output: <outdir>/perclass-<tag>-<mode>.json and .md
Usage : python3 perclass_decode.py --prompts P.jsonl --tag v23-r1 --mode c1|c8 [--natural] [--fl 256 | --fl-map '{"prose":256}']
"""
import argparse, json, os, statistics as st, sys, threading, time, urllib.request, hashlib

ORDER = ["prose", "structured", "code", "shell_ops", "tool_json"]
WANT = {"vllm:spec_decode_num_draft_tokens_total": "draft", "vllm:spec_decode_num_accepted_tokens_total": "acc",
        "vllm:spec_decode_num_drafts_total": "steps", "vllm:generation_tokens_total": "gen",
        "vllm:prefix_cache_hits_total": "pc_h", "vllm:prefix_cache_queries_total": "pc_q",
        "vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting"}


def metrics(base):
    t = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode(); d = {v: 0.0 for v in WANT.values()}
    for l in t.splitlines():
        if l.startswith("#"): continue
        name = l.split("{", 1)[0].split(" ", 1)[0]
        if name in WANT: d[WANT[name]] += float(l.split()[-1])
    return d


class Sampler(threading.Thread):
    def __init__(self, base):
        super().__init__(daemon=True); self.base = base; self.stop = threading.Event(); self.run_max = 0.0; self.wait_max = 0.0
        self.samples = 0; self.hist = {}
    def run(self):
        while not self.stop.is_set():
            try:
                m = metrics(self.base); r = m["running"]; self.run_max = max(self.run_max, r); self.wait_max = max(self.wait_max, m["waiting"])
                self.samples += 1; self.hist[int(r)] = self.hist.get(int(r), 0) + 1
            except Exception: pass
            self.stop.wait(0.25)


def one(base, model, row, fl, natural, timeout=900):
    body = {"model": model, "messages": row["messages"], "max_tokens": fl, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    if not natural: body.update({"min_tokens": fl, "ignore_eos": True})
    if row.get("tools"): body["tools"] = row["tools"]; body["tool_choice"] = "auto"
    t0 = time.monotonic(); t_first = None; ct_first = None; t_last = None; ct = 0; finish = None; txt = []; err = None
    tool_chunks = 0
    try:
        r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                          headers={"Content-Type": "application/json"}), timeout=timeout)
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            ev = json.loads(data); now = time.monotonic()
            u = ev.get("usage") or {}
            c = u.get("completion_tokens")
            if c is not None and c > ct:
                ct = c; t_last = now
                if t_first is None and c >= 1: t_first = now; ct_first = c
            for ch in ev.get("choices") or []:
                dl = ch.get("delta") or {}
                if dl.get("content"): txt.append(dl["content"])
                if dl.get("reasoning_content"): txt.append(dl["reasoning_content"])
                if dl.get("tool_calls"):  # drop the per-request random tool-call id so sha compares trajectories only
                    tool_chunks += 1
                    txt.append(json.dumps([{k: v for k, v in t.items() if k != "id"} for t in dl["tool_calls"]], sort_keys=True))
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:300]
    dec_tok = (ct - ct_first) if (ct_first is not None) else 0
    dec_s = (t_last - t_first) if (t_first is not None and t_last is not None) else 0.0
    blob = "".join(txt)
    return {"id": row["id"], "source": row.get("source"), "pt": row.get("prompt_tokens"), "ct": ct, "ct_first": ct_first,
            "t_send": t0, "t_first": t_first, "t_last": t_last,
            "ttft": round(t_first - t0, 4) if t_first else None, "decode_tokens": dec_tok, "decode_s": round(dec_s, 4),
            "decode_tok_s": round(dec_tok / dec_s, 2) if dec_s > 0 and dec_tok > 0 else None,
            "tpot_ms": round(1000 * dec_s / dec_tok, 3) if dec_s > 0 and dec_tok > 0 else None,
            "finish": finish, "tool_call": tool_chunks > 0, "sha": hashlib.sha1(blob.encode()).hexdigest()[:12],
            "head": blob[:60].replace("\n", " "), "err": err}


def med(xs):
    xs = [x for x in xs if x is not None]; return round(st.median(xs), 3) if xs else None


def run_class(base, model, rows, fl, natural, mode, conc):
    m0 = metrics(base); smp = Sampler(base); smp.start(); t0 = time.monotonic(); out = []; waves = []
    if mode == "c1":
        for r in rows: out.append(one(base, model, r, fl, natural))
        expected = 1; dropped = 0
    else:
        nfull = (len(rows) // conc) * conc; dropped = len(rows) - nfull; expected = conc
        for w in range(0, nfull, conc):
            batch = rows[w:w + conc]; res = [None] * len(batch)
            def go(i, r): res[i] = one(base, model, r, fl, natural)
            th = [threading.Thread(target=go, args=(i, r)) for i, r in enumerate(batch)]
            [t.start() for t in th]; [t.join() for t in th]
            ok = [x for x in res if x["t_first"] is not None and x["t_last"] is not None]
            win = (max(x["t_last"] for x in ok) - min(x["t_first"] for x in ok)) if ok else 0.0
            dt = sum(x["decode_tokens"] for x in ok)
            waves.append({"wave": w // conc, "n": len(batch), "ok": len(ok), "decode_tokens": dt, "window_s": round(win, 4),
                          "agg_decode_tok_s": round(dt / win, 2) if win > 0 else None,
                          "stagger_first_s": round(max(x["t_first"] for x in ok) - min(x["t_first"] for x in ok), 4) if ok else None})
            out += res
    wall = time.monotonic() - t0; smp.stop.set(); smp.join(timeout=2); m1 = metrics(base)
    d = {k: m1[k] - m0[k] for k in ("draft", "acc", "steps", "gen", "pc_h", "pc_q")}
    ok = [x for x in out if x["err"] is None and x["decode_tokens"]]
    ours = sum(x["ct"] for x in out)
    dec_tok = sum(x["decode_tokens"] for x in ok); dec_s = sum(x["decode_s"] for x in ok)
    s = {"n": len(rows), "n_run": len(out), "ok": len(ok), "errors": [x["err"] for x in out if x["err"]][:5], "n_errors": sum(1 for x in out if x["err"]),
         "c8_dropped_tail": dropped if mode == "c8" else None,
         "tokens": ours, "decode_tokens": dec_tok, "wall_s": round(wall, 2),
         "exact_fl": (all(x["ct"] == fl for x in out) if not natural else None),
         "ct_median": med([x["ct"] for x in out]),
         "decode_tok_s_pooled": round(dec_tok / dec_s, 2) if dec_s else None,      # C1 headline (per-stream)
         "decode_tok_s_median": med([x["decode_tok_s"] for x in ok]),
         "tpot_ms_median": med([x["tpot_ms"] for x in ok]), "ttft_s_median": med([x["ttft"] for x in out]),
         "accept": round(d["acc"] / d["draft"], 4) if d["draft"] else None, "acc_per_step": round(d["acc"] / d["steps"], 3) if d["steps"] else None,
         "drafted": d["draft"], "accepted": d["acc"], "draft_steps": d["steps"],
         "gen_delta": d["gen"], "foreign_tokens": round(d["gen"] - ours, 1),
         "prefix_hit_rate": round(d["pc_h"] / d["pc_q"], 4) if d["pc_q"] else None,
         "max_running_seen": smp.run_max, "max_waiting_seen": smp.wait_max, "expected_running": expected, "running_hist": smp.hist,
         "tool_call_frac": round(sum(1 for x in out if x["tool_call"]) / len(out), 3) if out else None}
    if mode == "c8":
        wd = sum(w["decode_tokens"] for w in waves); ws = sum(w["window_s"] for w in waves)
        s["agg_decode_tok_s"] = round(wd / ws, 2) if ws else None                  # C8 headline (aggregate)
        s["waves"] = waves
    for x in out:
        for k in ("t_send", "t_first", "t_last"):
            if x[k] is not None: x[k] = round(x[k] - t0, 4)
    return s, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="$HOME/dsv41/bench/perclass_prompts.jsonl")
    ap.add_argument("--tag", required=True); ap.add_argument("--mode", choices=["c1", "c8"], default="c1")
    ap.add_argument("--natural", action="store_true"); ap.add_argument("--fl", type=int, default=256)
    ap.add_argument("--fl-map", default=""); ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--classes", default=" ".join(ORDER)); ap.add_argument("--sources", default="")
    ap.add_argument("--base", default="http://127.0.0.1:30006"); ap.add_argument("--model", default="dsv41-flash-uva")
    ap.add_argument("--outdir", default="$HOME/dsv41/1017/results-perclass-baseline")
    a = ap.parse_args()
    flm = json.loads(a.fl_map) if a.fl_map else {}
    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    if a.sources: rows = [r for r in rows if r["source"] in a.sources.split(",")]
    classes = a.classes.split()
    m = json.load(urllib.request.urlopen(a.base + "/v1/models", timeout=10))
    assert any(x["id"] == a.model for x in m["data"]), f"model {a.model} not served"
    one(a.base, a.model, {"id": "warm", "messages": [{"role": "user", "content": "hi"}]}, 8, True)
    os.makedirs(a.outdir, exist_ok=True)
    mode_s = a.mode + ("-nat" if a.natural else "")
    res = {"tag": a.tag, "mode": a.mode, "natural": a.natural, "conc": a.conc if a.mode == "c8" else 1,
           "fl": {c: flm.get(c, a.fl) for c in classes}, "prompts": a.prompts,
           "prompts_sha1": hashlib.sha1(open(a.prompts, "rb").read()).hexdigest(), "order": classes,
           "ts_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base": a.base, "res": {}, "rows": {}}
    for c in classes:
        cr = [r for r in rows if r["cls"] == c]
        s, out = run_class(a.base, a.model, cr, flm.get(c, a.fl), a.natural, a.mode, a.conc)
        res["res"][c] = s; res["rows"][c] = out
        head = s.get("agg_decode_tok_s") if a.mode == "c8" else s["decode_tok_s_pooled"]
        print(f"[{a.tag} {mode_s}] {c:10s} n={s['n_run']:2d} tok={s['tokens']:6d} decode={head} tok/s (stream median "
              f"{s['decode_tok_s_median']}) TPOT={s['tpot_ms_median']}ms accept={s['accept']} acc/step={s['acc_per_step']} "
              f"foreign_tok={s['foreign_tokens']} maxrun={s['max_running_seen']}/{s['expected_running']} err={s['n_errors']} "
              f"exact={s['exact_fl']}", flush=True)
    res["ts_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    base = f"{a.outdir}/perclass-{a.tag}-{mode_s}"
    json.dump(res, open(base + ".json", "w"), indent=1)
    hdr = "agg decode tok/s" if a.mode == "c8" else "decode tok/s (pooled)"
    L = [f"### perclass {a.tag} {mode_s} ({res['ts_start']})", "",
         f"| class | n | tokens | {hdr} | stream median tok/s | TPOT ms | TTFT s | accept | acc/step | foreign tok | max running |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in classes:
        s = res["res"][c]; head = s.get("agg_decode_tok_s") if a.mode == "c8" else s["decode_tok_s_pooled"]
        L.append(f"| {c} | {s['n_run']} | {s['tokens']} | {head} | {s['decode_tok_s_median']} | {s['tpot_ms_median']} | "
                 f"{s['ttft_s_median']} | {s['accept']} | {s['acc_per_step']} | {s['foreign_tokens']} | {s['max_running_seen']:.0f}/{s['expected_running']} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

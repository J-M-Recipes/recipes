#!/usr/bin/env python3
"""agent_fixture_x3.py — Card P2b enlarged per-class DSpark acceptance fixture (sibling of $HOME/dsv41/agent_fixture_o.sh;
the original is NOT edited). Same SYS/TOOLS/temperature/thinking-off/chat path as agent_fixture_o.sh. The original's prompts are
items 1-2 of each class (prose 1, structured 2, code 2, shell_ops 1, tool_json 2); 8 prompts per class here (>= 4x per class),
each run ONCE (greedy repeats are deterministic — the original's 2x loop adds no information). Fixed CAT_ORDER on every arm.
Per-prompt /metrics deltas (sequential C1) give per-prompt acceptance; class acceptance = sum(accepted)/sum(drafted) (token-
weighted, = the original's definition). Records sha1 of each output so cross-arm trajectory identity can be counted.
Usage: TAG=<tag> OUT=<json> python3 agent_fixture_x3.py
"""
import json, time, urllib.request, os, hashlib, statistics as st
tag = os.environ["TAG"]; OUT = os.environ["OUT"]; B = "http://127.0.0.1:30006"
def metrics():
    t = urllib.request.urlopen(B + "/metrics", timeout=10).read().decode(); d = {}
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_draft_tokens_total"): d["draft"] = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_accepted_tokens_total"): d["acc"] = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"): d["steps"] = float(l.split()[-1])
    for k in ("draft", "acc", "steps"): d.setdefault(k, 0.0)
    return d
SYS = "You are Hermes, an autonomous agent. You have tools: terminal(command), read_file(path), write_file(path,content), web_search(query), patch(path,old,new). When you need a tool, emit a tool call. Be terse."
TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]
CATS = {
 "prose": [
  ("In one paragraph, explain to a smart non-engineer why a model that is fast at counting can be slow at writing prose when speculative decoding is on.", 200),
  ("In one paragraph, explain to a curious teenager why a refrigerator makes the back of it warm.", 200),
  ("Write a short paragraph describing a rainy autumn morning in a small harbor town, for a travel newsletter.", 200),
  ("In one paragraph, explain why sourdough bread rises without commercial yeast.", 200),
  ("Write a friendly two-paragraph note to a neighbor explaining that a package of theirs was delivered to you by mistake.", 200),
  ("In one paragraph, explain to a new manager why one-on-one meetings matter even when nothing is wrong.", 200),
  ("In one paragraph, describe what makes a good mystery novel opening, without naming any real books.", 200),
  ("Explain in one paragraph why the sky looks red at sunset, as you would to a ten-year-old.", 200)],
 "structured": [
  ("Compare vLLM UVA expert offload vs SGLang OffloaderV2 whole-layer copy for a 384-expert top-6 MoE. Output a markdown table with columns: approach, bytes moved per token, bottleneck, expected tok/s on a ~330 GB/s link. Then one sentence of verdict.", 220),
  ("Status report format: 5 bullet points, each '<component>: <state> — <one metric>'. Components: weights load, KV cache, autotune, graph capture, API bind. Invent plausible values for a vLLM boot.", 160),
  ("Output a markdown table comparing HDD, SATA SSD, and NVMe SSD with columns: interface, typical sequential read MB/s, typical random IOPS, best use. Then one sentence of advice.", 220),
  ("Write a YAML document describing a three-service docker compose stack (web, api, db) with ports, images, and a healthcheck for each. YAML only.", 220),
  ("Produce a JSON object summarizing a fictional server: hostname, cpu_cores, ram_gib, disks (list of {mount, size_gib, used_pct}), and services (list of names). JSON only.", 200),
  ("Make a markdown checklist of 8 items for preparing a Linux server before a kernel upgrade.", 180),
  ("Give a markdown table of the HTTP status codes 200, 201, 301, 400, 401, 403, 404, 500, 503 with columns: code, name, one-line meaning.", 220),
  ("Write an incident summary in this exact format: Title / Impact / Timeline (4 bullets with times) / Root cause / Action items (3 bullets). Invent a plausible disk-full outage.", 220)],
 "code": [
  ("Write a Python function `knee(rows)` that takes [(conc, agg_tps)] and returns the concurrency where marginal gain per added stream first drops below 5 tok/s. Include a docstring and a 3-line example. Code only.", 256),
  ("Write a bash script that loops over shards model-00001..00048-of-00048.safetensors, checks each exists in $SRC, and prints missing ones. Code only.", 200),
  ("Write a Python function that parses lines like 'key=value' from a text file into a dict, ignoring blank lines and lines starting with '#'. Code only.", 220),
  ("Write a Python function `median_ms(samples)` that returns the median of a list of floats in milliseconds rounded to 2 decimals, raising ValueError on an empty list. Code only.", 200),
  ("Write a small Python script that reads a JSON lines file given as argv[1] and prints the count of records per value of the field 'status'. Code only.", 220),
  ("Write a bash function `retry` that runs a command up to N times with a 2-second sleep between attempts and returns the last exit code. Code only.", 200),
  ("Write a Python dataclass `Run` with fields tag (str), tok_s (float), ttft_s (float) and a classmethod `from_dict`. Code only.", 200),
  ("Write a JavaScript function that debounces another function by a given number of milliseconds. Code only.", 200)],
 "shell_ops": [
  ("Give the exact sequence of shell commands to: stop docker container dsv41-vllm-v7, rename it with suffix -old, drop page caches with sudo, and relaunch via launch-dsv41-vllm.sh with OFFGB=70 SEQS=16. Commands only, one per line.", 160),
  ("Give the exact shell commands to find the 10 largest files under /var/log and delete those older than 30 days. Commands only, one per line.", 160),
  ("Give the exact shell commands to create a user named deploy, add it to the docker group, and set up ~/.ssh/authorized_keys with mode 600. Commands only, one per line.", 160),
  ("Give the exact shell commands to check which process is listening on port 30006, show its full command line, and its memory use. Commands only, one per line.", 160),
  ("Give the exact shell commands to tar and gzip the directory $HOME/results into results.tgz, copy it to host lan-host under /backup, and verify the checksum on both sides. Commands only, one per line.", 160),
  ("Give the exact shell commands to list docker containers including stopped ones, show the last 50 log lines of container web, and restart it. Commands only, one per line.", 160),
  ("Give the exact shell commands to show disk usage per mount, inode usage, and the top 5 directories by size under /home. Commands only, one per line.", 160),
  ("Give the exact git commands to create a branch fix-boot from main, commit all changes with message 'fix boot', and push it to origin with upstream tracking. Commands only, one per line.", 160)],
 "tool_json": [
  ("List the docker containers on host gb300-station and check free disk on /. Use tools.", 120),
  ("Read $HOME/dsv41/launch-dsv41-vllm.sh then write a copy to /tmp/l2.sh with port 30007. Use tools; call read_file first.", 120),
  ("Check the GPU memory usage on this machine. Use tools.", 120),
  ("Read the file /etc/os-release and tell me the distribution. Use tools; call read_file first.", 120),
  ("Write a file /tmp/notes.txt containing the line 'boot ok' and then show its contents. Use tools.", 120),
  ("Find how much RAM is free on this machine and how long it has been up. Use tools.", 120),
  ("Read $HOME/dsv41/results/ledger.md and summarize the last entry. Use tools; call read_file first.", 120),
  ("Show the last 20 lines of the system journal for the docker service. Use tools.", 120)],
}
ORDER = ["prose", "structured", "code", "shell_ops", "tool_json"]
def call(msgs, n, tools=None):
    body = {"model": "dsv41-flash-uva", "messages": msgs, "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    if tools: body["tools"] = tools; body["tool_choice"] = "auto"
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600))
    dt = time.time() - t; u = r["usage"]; ch = r["choices"][0]; m = ch["message"]
    blob = (m.get("content") or "") + json.dumps(m.get("tool_calls") or [], sort_keys=True)
    return u["completion_tokens"], dt, ch["finish_reason"], bool(m.get("tool_calls")), hashlib.sha1(blob.encode()).hexdigest()[:12], (m.get("content") or "")[:60].replace("\n", " ")
call([{"role": "user", "content": "hi"}], 8)
res = {}
for cat in ORDER:
    rows = []; A = D = S = 0.0; toks = 0; secs = 0.0
    for j, (p, n) in enumerate(CATS[cat]):
        m0 = metrics()
        k, dt, fr, tc, h, head = call([{"role": "system", "content": SYS}, {"role": "user", "content": p}], n, TOOLS if cat == "tool_json" else None)
        m1 = metrics(); dr = m1["draft"] - m0["draft"]; ac = m1["acc"] - m0["acc"]; sp = m1["steps"] - m0["steps"]
        A += ac; D += dr; S += sp; toks += k; secs += dt
        rows.append({"i": j, "tokens": k, "s": round(dt, 3), "finish": fr, "tool_call": tc, "sha": h, "drafted": dr, "accepted": ac,
                     "accept": round(ac / dr, 4) if dr else None, "head": head})
    pp = [r["accept"] for r in rows if r["accept"] is not None]
    res[cat] = {"accept": round(A / D, 4) if D else None, "acc_per_step": round(A / S, 3) if S else None, "tok_s": round(toks / secs, 1),
                "tokens": toks, "n": len(rows), "per_prompt_mean": round(st.mean(pp), 4) if pp else None,
                "per_prompt_sd": round(st.stdev(pp), 4) if len(pp) > 1 else None, "rows": rows}
    print(f"[{tag}] {cat:11s} accept={res[cat]['accept']} (per-prompt mean {res[cat]['per_prompt_mean']} sd {res[cat]['per_prompt_sd']}) tok/s={res[cat]['tok_s']} tokens={toks}", flush=True)
json.dump({"tag": tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "order": ORDER, "res": res}, open(OUT, "w"), indent=1)

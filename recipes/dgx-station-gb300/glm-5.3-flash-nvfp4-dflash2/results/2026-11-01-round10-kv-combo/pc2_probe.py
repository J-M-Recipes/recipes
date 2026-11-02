#!/usr/bin/env python3
"""pc2_probe.py — what breaks prefix reuse on the GLM-Flash SGLang lane: context length, or interleaved sessions? (Card PC2)
Server needs --enable-cache-report. Agent-shaped turn 2: [user, assistant(content), user(new)].
  LEN-<n>K    : one session, turn 2 immediately after turn 1, n in 32/64/128/200
  INT-<k>x<n>K: k sessions, all turn 1s first, then all turn 2s in the same order (interleaved, like concurrent agents)
Also scrapes `docker logs` Prefill lines (UTC) for mamba usage around each test. Writes JSON to argv[1].
"""
import calendar, json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_probe import ctx, chat  # noqa: E402

CONT = os.getenv("CONT", "")
Q = "\n\nReply with the single word OK."


def mamba_usage(t0, t1):
    if not CONT:
        return None
    out = subprocess.run(["docker", "logs", CONT], capture_output=True, text=True).stdout
    out += subprocess.run(["docker", "logs", CONT], capture_output=True, text=True).stderr
    mx = 0.0
    for ln in out.replace("\r", "\n").splitlines():
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", ln)
        if not m:
            continue
        ts = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        if t0 - 1 <= ts <= t1 + 1:
            mm = re.search(r"mamba usage: ([0-9.]+)", ln)
            if mm:
                mx = max(mx, float(mm.group(1)))
    return mx


def turn2(c, t1):
    return [{"role": "user", "content": c + Q}, {"role": "assistant", "content": t1["content"] or "OK"},
            {"role": "user", "content": "Now reply with the single word YES."}]


def frac(r):
    return round((r["cached_tokens"] or 0) / max(1, r["prompt_tokens"] or 1), 4)


def main():
    res = []
    for n in (32, 64, 128, 200):
        c = ctx(f"LEN{n}", n * 1000); t0 = time.time()
        a = chat([{"role": "user", "content": c + Q}]); b = chat(turn2(c, a)); t1 = time.time()
        row = {"test": f"LEN-{n}K", "prompt_tokens": b["prompt_tokens"], "cached": b["cached_tokens"], "frac": frac(b),
               "t1_wall": a["wall_s"], "t2_wall": b["wall_s"], "mamba_usage_max": mamba_usage(t0, t1)}
        print(f"[pc2] {row}", flush=True); res.append(row)
    for k, n in ((3, 32), (6, 32), (3, 150)):
        cs = [ctx(f"INT{k}x{n}-{i}", n * 1000) for i in range(k)]; t0 = time.time()
        a = [chat([{"role": "user", "content": c + Q}]) for c in cs]
        b = [chat(turn2(c, x)) for c, x in zip(cs, a)]; t1 = time.time()
        row = {"test": f"INT-{k}x{n}K", "fracs": [frac(x) for x in b], "t2_walls": [x["wall_s"] for x in b],
               "mamba_usage_max": mamba_usage(t0, t1)}
        print(f"[pc2] {row}", flush=True); res.append(row)
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()

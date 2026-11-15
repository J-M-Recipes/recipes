#!/usr/bin/env python3
"""Long-context decode traffic for the shadow arm. Greedy, logprobs on, seeded filler so no prefix-cache hits.
usage: lc_drive.py <out.json>"""
import concurrent.futures as cf, hashlib, importlib.util, json, os, random, sys, time
DRV = '$HOME/dsv41/decode/w0/w0_drive.py'
assert hashlib.sha256(open(DRV, 'rb').read()).hexdigest().startswith('6c1fb1be1080'), 'driver drift'
os.environ.setdefault('W0_BASE', 'http://127.0.0.1:30006'); os.environ.setdefault('W0_MODEL', 'dsv41-flash-uva')
spec = importlib.util.spec_from_file_location('w0', DRV); w0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(w0)
gt = importlib.util.spec_from_file_location('gt', '$HOME/dsv41/decode/bench/gt_mix_drive.py')
OUT = sys.argv[1]; G = importlib.util.module_from_spec(gt); sys.argv = ['x']; gt.loader.exec_module(G)
ASK = ["\nSummarize the most frequent themes of the words above in about 150 words of plain prose.",
       "\nWrite a short Python function that counts word frequencies, then explain it in two paragraphs.",
       "\nList twenty of the words above in alphabetical order, one per line, each with a one-line definition."]
def one(tag, size, toks, i=0):
    rng = random.Random(f"lc-{tag}-{size}-{i}")
    text = f"(lc {tag} {size} {i}) " + " ".join(rng.choice(G.WORDS) for _ in range(max(4, int(size / 1.05)))) + ASK[i % 3]
    t0 = time.time()
    try:
        d = json.loads(w0.post('/v1/chat/completions', w0.chat_body(text, toks, stream=False, logprobs=True), timeout=1800))
        return {"tag": tag, "i": i, "s": round(time.time() - t0, 2), "prompt_tokens": d["usage"]["prompt_tokens"],
                "completion_tokens": d["usage"]["completion_tokens"]}
    except Exception as e:
        return {"tag": tag, "i": i, "error": repr(e)[:300]}
out = []
for size in (8000, 64000, 250000, 600000):
    for i in range(2):
        r = one("c1", size, 256, i); out.append(r); print(r, flush=True)
with cf.ThreadPoolExecutor(8) as ex:
    for r in ex.map(lambda i: one("c8", 30000, 192, i), range(16)):
        out.append(r); print(r, flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
print("LC errors", sum('error' in r for r in out), "completion_tokens", sum(r.get('completion_tokens', 0) for r in out))

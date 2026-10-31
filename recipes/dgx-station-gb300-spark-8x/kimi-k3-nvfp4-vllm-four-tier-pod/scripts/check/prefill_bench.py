import json, sys, time, urllib.request
URL = "http://127.0.0.1:%s/v1/completions" % sys.argv[1]
base = "The transistor was invented at Bell Labs in 1947 by Bardeen, Brattain and Shockley, and it changed electronics forever. "
res = []
for reps in (1, 10, 40, 120):
    runs = []
    for r in range(3):
        p = base * reps + "Summarize: "
        b = json.dumps({"model": "k3", "prompt": p, "max_tokens": 1, "temperature": 0}).encode()
        t0 = time.time()
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL, b, {"Content-Type": "application/json"}), timeout=900))
        runs.append(time.time() - t0); n = d["usage"]["prompt_tokens"]
    m = sorted(runs)[1]
    res.append({"prompt_tokens": n, "ttft_s": runs, "median_s": m, "tok_per_s": n / m})
    print(f"prompt {n:5d} tok: TTFT median {m:7.2f} s -> {n/m:6.1f} tok/s  runs " + " ".join(f"{t:.2f}" for t in runs), flush=True)
json.dump(res, open("./prefill-n93.json", "w"), indent=1)

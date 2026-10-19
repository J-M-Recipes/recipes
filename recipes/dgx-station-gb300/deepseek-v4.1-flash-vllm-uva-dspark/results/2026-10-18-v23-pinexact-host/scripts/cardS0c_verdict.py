#!/usr/bin/env python3
"""Card S0c verdict: ARM B (S0 restart, PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024) vs ARM A (live v22 REF).
Bar pre-registered in cardS0c_runner.sh: knee6 r2 C8 >= -2.0%, knee6 r2 C6 >= -2.0%, RT C1 (mean of 3) >= -2.0%.
knee.sh r2 C8 is informational. Usage: cardS0c_verdict.py <results-dir> -> markdown; first line 'RESULT PASS|FAIL'."""
import json, re, sys
R = sys.argv[1] if len(sys.argv) > 1 else "$HOME/dsv41/1017/results-cardS0c"
BAR = -2.0

def j(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def knee(path):
    d = j(path)
    if not d:
        return {}
    out = {}
    for row in d.get("rows", []) if isinstance(d, dict) else d:
        try:
            out[int(row[0])] = float(row[1])
        except Exception:
            pass
    return out

def rt(tag):
    v = []
    for r in (1, 2, 3):
        d = j(f"{R}/rt-{tag}/c1-r{r}.json")
        if d and "summary" in d and d["summary"].get("decode_agg") is not None:
            v.append(float(d["summary"]["decode_agg"]))
    return (sum(v) / len(v) if v else None), v

def pct(b, a):
    return None if (a in (None, 0) or b is None) else 100.0 * (b - a) / a

def fmt(x, n=1):
    return "—" if x is None else f"{x:.{n}f}"

def grep(path, pat):
    try:
        return re.findall(pat, open(path).read())
    except Exception:
        return []

A6 = {k: knee(f"{R}/knee6-A-{k}.json") for k in ("r1", "r2")}
B6 = {k: knee(f"{R}/knee6-B-{k}.json") for k in ("r1", "r2")}
AK = {k: knee(f"{R}/knee-A-{k}.json") for k in ("r1", "r2")}
BK = {k: knee(f"{R}/knee-B-{k}.json") for k in ("r1", "r2")}
a1, a1v = rt("A"); b1, b1v = rt("B")

checks = [
    ("knee6 r2 C8", A6["r2"].get(8), B6["r2"].get(8)),
    ("knee6 r2 C6", A6["r2"].get(6), B6["r2"].get(6)),
    ("RT C1 (kbench, mean of 3)", a1, b1),
]
ok = True; rows = []
for name, a, b in checks:
    d = pct(b, a)
    p = d is not None and d >= BAR
    ok &= p
    rows.append(f"| {name} | {fmt(a)} | {fmt(b)} | {fmt(d,2)}% | {'PASS' if p else 'FAIL' if d is not None else 'MISSING'} |")

print(f"RESULT {'PASS' if ok else 'FAIL'}")
print()
print("| gated cell | A = v22 REF (live) | B = S0 (restart) | Δ B vs A | bar ≥ −2.0% |")
print("|---|---|---|---|---|")
print("\n".join(rows))
print()
print("| informational | A r1 | A r2 | B r1 | B r2 | Δ r2 |")
print("|---|---|---|---|---|---|")
for lab, src_a, src_b, c in (("knee6 C6", A6, B6, 6), ("knee6 C8", A6, B6, 8), ("knee6 C16", A6, B6, 16), ("knee6 C1", A6, B6, 1),
                             ("knee.sh C8 (S0 caveat instrument)", AK, BK, 8), ("knee.sh C16", AK, BK, 16), ("knee.sh C1", AK, BK, 1)):
    print(f"| {lab} | {fmt(src_a['r1'].get(c))} | {fmt(src_a['r2'].get(c))} | {fmt(src_b['r1'].get(c))} | {fmt(src_b['r2'].get(c))} | {fmt(pct(src_b['r2'].get(c), src_a['r2'].get(c)),2)}% |")
print(f"| RT C1 runs | {' / '.join(fmt(x) for x in a1v)} | | {' / '.join(fmt(x) for x in b1v)} | | |")
print()
pa = grep(f"{R}/hostmem-A-live.txt", r"pinned_devzero maps=\d+ size_gib=([0-9.]+)")
pb = grep(f"{R}/hostmem-B-steady.txt", r"pinned_devzero maps=\d+ size_gib=([0-9.]+)") or grep(f"{R}/hostmem-B-bound.txt", r"pinned_devzero maps=\d+ size_gib=([0-9.]+)")
ma = grep(f"{R}/hostmem-A-live.txt", r"MemAvailable: ([0-9.]+)")
mb = grep(f"{R}/hostmem-B-steady.txt", r"MemAvailable: ([0-9.]+)")
l9 = grep(f"{R}/facts-B.txt", r"rehome layer=9 .*?host_avail=([0-9.]+)GiB")
ha = [float(x) for x in grep(f"{R}/facts-B.txt", r"host_avail=([0-9.]+)GiB")]
kv = grep(f"{R}/facts-B.txt", r"GPU KV cache size: ([0-9,]+) tokens")
ld = grep(f"{R}/facts-B.txt", r"Loaded (\d+) configs")
sv = grep(f"{R}/facts-B.txt", r"Saved (\d+) configs")
print(f"host pinned (/dev/zero Size sum): A {pa[-1] if pa else '—'} GiB · B {pb[-1] if pb else '—'} GiB")
print(f"MemAvailable: A live {ma[-1] if ma else '—'} GiB · B steady {mb[-1] if mb else '—'} GiB")
print(f"B hook guard host_avail: L9 {l9[-1] if l9 else '—'} GiB · min {min(ha) if ha else '—'} GiB (guard 10 GiB)")
print(f"B KV tokens {kv[-1] if kv else '—'} · autotune Loaded {ld[-1] if ld else '—'} configs, Saved lines {len(sv)}")

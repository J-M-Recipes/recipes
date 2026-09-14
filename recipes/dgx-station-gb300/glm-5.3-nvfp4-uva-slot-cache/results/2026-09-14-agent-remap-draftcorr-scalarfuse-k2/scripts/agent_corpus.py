#!/usr/bin/env python3
"""agent_corpus.py — drive the glm-5.3-big lane with REAL Hermes agent traffic (full system prompt, tool loops,
multi-hop terminal/file work in a real codebase) so the ID_RING hook captures agent-workload routing.
Adds a temporary list-form custom_providers entry (CLI-visible) and restores config.yaml byte-identical.
Usage: agent_corpus.py --key-file F [--tasks N] [--repo DIR] [--reasoning low] [--timeout 900]
"""
import argparse, os, re, subprocess, sys, time, json
p = argparse.ArgumentParser()
p.add_argument("--key-file", required=True); p.add_argument("--tasks", type=int, default=12)
p.add_argument("--repo", default=os.path.expanduser("~/hermes/glm53-speed-research-20260913/vllm-src"))
p.add_argument("--reasoning", default="low"); p.add_argument("--timeout", type=int, default=900)
p.add_argument("--profile", default="milo"); p.add_argument("--taskset", default="vllm"); p.add_argument("--log", default="agent_corpus.log")
a = p.parse_args()
key = open(os.path.expanduser(a.key_file)).read().strip()
CFG = os.path.expanduser(f"~/.hermes/profiles/{a.profile}/config.yaml"); orig = open(CFG).read()
PROV = "glm53_corpus_tmp"
entry = (f"  - name: {PROV}\n    base_url: http://192.168.1.9:30001/v1\n    api_key: {key}\n    api_mode: openai\n"
         f"    model: glm-5.3-big\n    default_model: glm-5.3-big\n    context_length: 262144\n    request_timeout_seconds: {a.timeout}\n")
m = re.search(r"^custom_providers:\n", orig, re.M)
if m and re.match(r"custom_providers:\n\s+-\s", orig[m.start():]): mod = orig[:m.end()] + entry + orig[m.end():]
elif m:
    end = re.search(r"^\S", orig[m.end():], re.M); end_i = m.end() + (end.start() if end else len(orig) - m.end())
    mod = orig[:m.start()] + "custom_providers:\n" + entry + orig[end_i:]
else: mod = orig + "\ncustom_providers:\n" + entry
R = a.repo
TASKS = [
 f"In {R}, find where the CLI flag --enable-return-routed-experts is defined and which config field it maps to. Report file:line for both.",
 f"In {R}, count how many Python files under vllm/model_executor/layers/fused_moe/ import triton. List the three largest of them by line count.",
 f"In {R}, read vllm/model_executor/layers/fused_moe/routed_experts_capturer.py and explain in 5 bullets what RoutedExpertsManager.store_batch does and what dtype the slot buffer uses. Cite line numbers.",
 f"In {R}, find every place 'cudagraph_capture_sizes' is assigned a default. Report file:line and the default value.",
 f"In {R}, locate the GLM-4 MoE MTP model file, report its class names and how many experts the MTP block's router selects (cite lines).",
 f"In {R}, write a small script /tmp/agentcorpus/count_moe.py that counts lines matching 'fused_moe' across vllm/**/*.py and prints the top 5 files. Run it and report the output.",
 f"In {R}, find the UVA offloader implementation and summarize the copy/event pattern it uses for offloaded weights, with file:line citations.",
 f"In {R}, grep for 'is_monolithic' and explain, in 4 sentences, what distinguishes a monolithic MoE kernel from a modular one in this codebase, citing 2 files.",
 f"In {R}, list the speculative decoding methods supported (search for 'method' choices in the speculative config), report file:line and the list.",
 f"In {R}, read the top 80 lines of vllm/v1/worker/gpu_model_runner.py and report every imported symbol from vllm.model_executor.layers.fused_moe.",
 f"In {R}, find how --max-num-batched-tokens default is derived for chunked prefill. Report the relevant function, file:line, and the default number.",
 f"In {R}, write /tmp/agentcorpus/summary.md with a 10-line summary of what the routed_experts_capturer module does, then run wc -l on it and report the count.",
 f"In {R}, find the definition of FlashInfer autotune cache hashing (flashinfer_autotune_cache_hash) and explain what inputs feed the hash, with file:line.",
 f"In {R}, count the number of test files under tests/ whose name contains 'moe', and print the newest 3 by mtime with their sizes.",
 f"In {R}, find where 'num_speculative_tokens' is validated or bounded and report the constraint and file:line.",
 f"In {R}, run git log --oneline -5 and then read CONTRIBUTING.md's first 40 lines; report the 5 commit subjects and the first heading of CONTRIBUTING.md.",
]
if a.taskset == "recipes":
    R2 = os.path.expanduser("~/hermes/jm-recipes/recipes/recipes/dgx-station-gb300")
    TASKS = [
     f"In {R2}, list the recipe directories, then for glm-5.3-nvfp4-uva-slot-cache read README.md and report the served model name, image tag, and the three most recent results directories.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache, read RUNBOOK.md and summarize the T0 gate in 5 bullets with the exact commands.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache/results, find every SUMMARY.json, print each file's top-level keys, and report which one contains a 'needle' section.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache, grep all markdown for 'masked_row_copy' and report each file with its match count, then quote the sentence that gives its ms/step figure.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache/research, list the files, read speed-research-2026-09-13.md section headers, and report the ranked lever table's first three rows verbatim.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache/scripts, read launch-slotcache.sh and explain every SLOT_CACHE_* environment variable it sets and its default, as a table.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache/results/2026-09-13-k2-fair-gate-needle-trace, read throughput.csv and compute the mean tok/s per lane with a python one-liner; report the numbers.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache/results/2026-09-13-k2-fair-gate-needle-trace/trace-analysis, read slots-7360-ctx256k.json with python and report: total slots, min/max per-layer, and the 5 layers with the largest budgets.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache, read recipe.yaml and report every key under the top-level serving/launch section with its value; flag anything that looks like a port or context length.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache/patches, list the files, pick the largest .py, and describe what it patches in 6 bullets with line citations.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache/results/2026-09-13-context-slot-curve, read README.md and SUMMARY.json and report the tok/s at 256K, 512K, and 1M with the slot counts used.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache, write /tmp/agentcorpus/recipe_index.md listing every results/ directory with its README.md first heading; run wc -l on it and report the count.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache, grep for 'promotion_authorized' across all files, report the files and whether any value is true.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache/results/2026-09-08-e1-v2-live, find the analysis directory, read corrected-buckets.json with python, and report the top 4 buckets by ms/step.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache, read C1-QUALITY-GATE.md and C2-CONTINUATION-CONTRACT.md and give a 6-bullet diff of what each gate requires.",
     f"In {R2}/glm-5.3-nvfp4-uva-slot-cache/research/speed-research-2026-09-13-scripts, run 'python3 live_layer_skew.py --help' or read its docstring, then report what inputs it needs and where it writes output.",
    ]
open(CFG, "w").write(mod)
log = open(a.log, "a"); results = []
try:
    for i, task in enumerate(TASKS[: a.tasks]):
        t0 = time.time()
        r = subprocess.run(["hermes", "chat", "-q", task, "--oneshot", "-Q", "--yolo", "--provider", PROV, "-m", "glm-5.3-big",
                            "-t", "terminal,file", "--reasoning", a.reasoning], capture_output=True, text=True, timeout=a.timeout, cwd="/tmp/agentcorpus")
        out = r.stdout + "\n" + r.stderr; sid = re.search(r"session_id:\s*(\S+)", out)
        rec = {"i": i, "wall_s": round(time.time() - t0, 1), "exit": r.returncode, "session": sid.group(1) if sid else None,
               "tail": out[-300:].replace("\n", " | ")}
        results.append(rec); log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"task {i} wall={rec['wall_s']}s exit={rec['exit']} session={rec['session']}", flush=True)
finally:
    open(CFG, "w").write(orig); print("CONFIG_RESTORED", open(CFG).read() == orig)
print(json.dumps({"tasks": len(results), "total_wall_s": round(sum(r["wall_s"] for r in results), 1)}))

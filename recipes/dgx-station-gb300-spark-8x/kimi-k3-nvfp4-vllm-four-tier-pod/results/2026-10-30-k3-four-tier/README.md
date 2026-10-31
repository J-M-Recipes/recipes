# Run 2026-10-30-k3-four-tier — receipts

Reference box: DGX Station GB300 + RTX PRO 6000 Max-Q + 8× DGX Spark (`../../../../../hardware/dgx-station-gb300-spark-8x.yaml`).
Times are CDT unless a log line says otherwise (vLLM logs are UTC: 01:13 UTC = 20:13 CDT).

| path | what | origin |
|---|---|---|
| `throughput.csv` | headline speed table | from `raw/speed-transcribed.txt` |
| `raw/speed-transcribed.txt` | every speed number, re-checked against the session's stored tool output | **transcribed** (bench scripts printed to stdout) |
| `raw/server-logs/*.server.log` | engine logs per boot: `podg` breakable graphs, `podf` C fast path, `podx` proxy, `podw` graphed workers, `podr` reboot, `pod-n93-eager`, `dump-n93` record mode, `shadow-*` parity cuts | original |
| `raw/probes/*.json` | 3-prompt greedy probes with top-20 logprobs. `*-n93` = full model. **`*-n2` / `*-n12` are 2- and 12-layer truncated models — gibberish text is expected**; they were used only to compare stock vs pod on identical truncated models (retired as a parity test, see recipe README) | original |
| `raw/parity/shadow-n12-graph.json`, `shadow-n12-proxy.json`, `shadow-n13-proxy.json` | per-layer cosine / rel-L2, pod vs stock on identical inputs | original |
| `raw/parity/refcheck-n93.json`, `refcheck-n93.log`, `emucheck-n93.json` | all 92 layers vs fp32 checkpoint reference; NVFP4 emulation of the outlier layers | original |
| `raw/worker/g3-*.json` | Station→worker round trip by forced expert count (`spark0` v1 eager worker n=400/k; `spark0-wgraph` graphed n=300/k; `rtx0`) | original |
| `raw/worker/a5-*.json` | Spark-tier-only replay of real routing (36.5 ms/token p50 with GC frozen) | original |
| `raw/fabric/` | RDMA half-RTT by Spark and fan-out (switch placement) | original |
| `raw/prefill-n93.json`, `.txt`, `raw/server-logs/prefill-n93.worker-lines.log` | prompt reading by prompt length (2026-10-31 00:05); per-layer wait and per-worker arrival during it | original |
| `raw/prefill-batch-n93.json`, `raw/k3-4-prof-batch-spark1.txt`, `raw/parity/shadow-n13-batch-prefill.json`, `raw/server-logs/podb-n93.server.log`, `shadow-n13-batch.server-lines.log` | batched prompt path: full-model prompt reading (01:20 10-31), worker kernel batch vs loop on Spark 1, shadow parity on prompt tokens with the near-zero layer-12 debug lines | original |
| `raw/prefill-gpubatch-n93.json`, `raw/prefill-b1024c-n93.json`, `raw/parity/shadow-n13-gpubatch-prefill.json`, `raw/parity/shadow-n13-1024-prefill.json`, `raw/server-logs/pod1024-n93.server.log` | GPU-side collect (256-token chunks) and 1,024-token chunks (serving config): full-model prompt reading and shadow parity for each | original |
| `raw/conc-seqs8.json`, `raw/server-logs/pods8-n93.server.log` | concurrent decode with --max-num-seqs 8 (C = 1/2/4/8) and the cold-engine warmup timing | original |
| `raw/conc-k35-n93.json`, `raw/prefill-k35-n93.json`, `raw/server-logs/podk35-n93.server.log`, `raw/conc-shadow-n13-k35.json`, `raw/parity/shadow-n13-k35.json` | multi-token proxy decode (serving config, --max-num-seqs 8): concurrent decode + prompts on the full model; shadow parity on real traffic only | original |
| `raw/conc-rtxe-n93.json`, `raw/prefill-rtxe-n93.json`, `raw/server-logs/podrtxe-n93.server.log`, `raw/conc-shadow-n13-rtxeager.json`, `raw/parity/shadow-n13-rtxeager.json`, `raw/k3-5-bucket-sweep-spark1.txt` | serving config after the RTX small-batch graphs were turned off (flaky boot check); B12x micro-kernel cost by rows on Spark 1 | original |
| `raw/rtx-graph-check/` | RTX small-batch graphs vs eager vs an fp32 checkpoint reference (20 runs each, 57 trials), and the graph/eager scatter comparison behind the boot-check misfire | original |
| `raw/fault-injection/` | two runs killing a Spark worker mid-decode; `README.md` there | original |
| `system.json`, `system/*.json` | `collect_system_snapshot.sh` on the Station and all eight Sparks | original |

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
| `raw/fault-injection/` | two runs killing a Spark worker mid-decode; `README.md` there | original |
| `system.json`, `system/*.json` | `collect_system_snapshot.sh` on the Station and all eight Sparks | original |

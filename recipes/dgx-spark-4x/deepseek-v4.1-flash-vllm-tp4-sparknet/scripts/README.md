# scripts — what to run, and what is only here for the record

| script | for | notes |
|---|---|---|
| `launch-r7-c60.sh` | **this lane** | renders Owen's `bin/spark` output for each node, post-processes it (container name, overlay mounts), drops caches, launches ranks 3→0, waits for `/v1/models`. `STOCK=1` runs stock candidate-60 (arm A3b's shape). |
| `post_arm.sh` | **this lane** | post-battery functional/determinism suite: `gates.py`, two greedy runs, greedy under load, `so_probe.py`, the DSML live probe. Produced `results/2026-11-16-r7-candidate60/post/`. |
| `so_probe.py` | **this lane** | the spark-ds41f#1194 guard probe — 32 concurrent structured-output requests whose answers copy 40 records out of the prompt, two waves, then a liveness check. |
| `rollback-to-r6x.sh` | **this lane** | stop `dsv41-r7-lane`, `docker start dsv41-karmic-kraken` rank 3→0, wait, and print the KV line so you can see which engine answered. |
| `c60cfg/` | **this lane** | the two switched node maps and the six arm files. Copy them into the checkout before rendering — see the header of `launch-r7-c60.sh`: `nodes-switched.laneC.local.json` → `$DS41F_DIR/config/`, `arms/*.json` → `$DS41F_DIR/experiments/2026-11-16-c60-prefill/arms/`. |
| `cluster.sh` | r6x only | the r6x lane driver (`SPARK_HOSTS=... cluster.sh push\|up\|down\|status`). It launches `launch-vllm-tp4.sh`, the r6x command — it does **not** launch the r7 lane. |
| `launch-vllm-tp4.sh` | r6x only | the r6x per-rank launcher, generated from the live head command on 2026-10-23 by `render_launcher.py`. |
| `render_launcher.py`, `verify_launcher.py` | r6x only | the generator and the validator for that launcher (the validator diffs a stub-`docker` render against the live container's `inspect`). |
| `run_battery.sh`, `gates.py`, `greedy_probe.py`, `qeval.py`, `qeval_tasks.py`, `v41bench_tf.py`, `prefill_reps.py`, `mixed_probe.py`, `long_check.py`, `spec_by_cat.py` | the instrument | unchanged since 2026-10-23. The r7 numbers were read with the campaign's `dsv41-5way` `battery.py`, which runs the same phases through these same tools; `post_arm.sh` calls `gates.py` and `greedy_probe.py` from here directly. |

`v41bench_tf.py` needs DeepSeek's `encoding/` directory from the checkpoint (`DSV41_ENCODING_DIR`) — the 1017
checkpoint ships no chat template. Every tool takes its endpoint from `BASE_URL` and its served name from `MODEL`.

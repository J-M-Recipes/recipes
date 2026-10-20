# DeepSeek-V4.1-Flash-1017 on 4× DGX Spark — SGLang dsv4.1 TP4 over 200G RoCE (knapcio v2.4 + RoCEnante), DSpark k=5, 1M context

**Status: experimental** (2026-10-20). GhettoRak's Lane B: native 1017 weights served TP4 on Sparks 1–4 with
knapcio's v2.4 production stack and **one** documented override. Round 2 (six one-change boots, 2026-10-19/20) found no
lever that cleared a +2% bar, so this recipe publishes the authors' config as measured, with the ladder as evidence that
the obvious knobs are flat. Teacher-forced quality gate, tool-call gate and per-class acceptance: pending.

Sibling: [`dgx-spark-4x/deepseek-v4.1-flash-vllm-tp4-sparknet`](../deepseek-v4.1-flash-vllm-tp4-sparknet/) — same
checkpoint, Sparks 5–8 on the same switch, Christopher Owen's vLLM fork. Same model on one GB300:
[`dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark`](../../dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/).
Write-up: [DeepSeek Flash 1017: TensorFold vs SGLang vs vLLM on 4 DGX Sparks](https://al-engr.com/dsv41-1017-three-way-tensorfold-sglang-vllm-4x-dgx-spark.html) (Round 2 section).

| | SGLang (this recipe, B1a) | vLLM sibling (C0ctl2 / FINAL) |
|---|---:|---:|
| Stack | knapcio v2.4: SGLang `dsv4.1` @ `f80c91a4b` + RoCEnante + adapters | spark-ds41f r6x: karmic-kraken vLLM + TileLang + sparknet |
| C1 per stream, 8-class mix (tok/s) | 96.2 (95.0 run 1) | **104.3** |
| C1 aggregate incl. TTFT (tok/s) | 82.4 (80.0 run 1) | **92.2** (Round 1) |
| C4 / C6 aggregate (tok/s) | 204 / 253 (six boots 250–262) | **231 / 289** |
| Cold prefill 93K, 3-rep median (tok/s) | **6,187**; serving boot quiet 6,219 | 5,998; quiet 5,993 |
| 300K needle prefill (tok/s) | 3,484 / 4,029 / 4,741 (run 1 = first touch) | **5,462 / 5,471 / 5,453** |
| Short-chat TTFT behind a 93K prefill | **1.77 s** | 3.03 s |
| Context / KV pool | 1,048,576 / 4.52–5.93M, varies by boot | 1,048,576 / **6,537,516** fixed |
| qeval content | 71/75 | 71/75 |
| Warm boot to HTTP-ready | ~6.4 min | **~1.5 min** |

Like with like: the "C1" both lanes publish as the headline is the **per-stream** column (mean decode tok/s per
request, TTFT excluded). The Oct-18 post's B "84.8" and C "92.2" are the **aggregate** column (tokens / wall incl. TTFT).
Vs the sibling, this lane loses decode by ~8% at C1 and ~12% at C6 and loses 300K prefill; it ties 93K prefill, holds
1M too, and answers a short chat behind a long prefill in about half the time (knapcio's `--prefill-decode-interval 2` +
`DSV41_PREFILL_BYPASS`). It is also the lane closest to an upstream project (SGLang's own `dsv4.1` branch); the vLLM
sibling is a fork of a pre-release vLLM. Sibling numbers are from its R2 ladder (`results/2026-10-19-vllm-ladder-r2/`
in that recipe), not re-measured here.

## What this runs

- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, as shipped — routed
  experts MXFP4, attention/dense/Engram FP8, in-checkpoint DSpark (MTP) head. 48 shards, 510.3 GB; 48/48 sha256 == Hub
  LFS ([`hub-lfs-compare.txt`](results/2026-10-19-sglang-ladder-r2/raw/weights/hub-lfs-compare.txt)). One copy on
  spark1, NFS-shared to ranks 1–3 over the fabric.
- **Engine:** SGLang `dsv4.1` branch at `f80c91a4b` inside `lmsysorg/sglang:dev-dsv41`, + rhys101's SG17 RoCEnante
  overlay (TP4-adapted), b12x / b12x_next (`a7d7d29b`) and knapcio's adapters, built locally as
  `dsv41-4x-spark:canary-roce` (image ID `609343dd905f`, same on all four nodes) from `Dockerfile.canary-roce`
  (sha256 `dbd1cfbf…`, copy in the results bundle).
- **Config:** knapcio's v2.4 production block (`.env.tp4.B0`): TP4, **EP1**, chunked prefill 8192, mem-fraction 0.78,
  16 slots, 8M-token KV pin, DSpark block 5, Engram from NVMe-packed shards + 4 GiB row cache, the long
  `EXTRA_CONTAINER_ENV` production line, `--swa-prefix-tails 128 --prefill-decode-interval 2`, NCCL 1 MiB buffers /
  LL128 256 KiB / `^LL128` / 8 channels, both PFs. **Plus one override (arm B1a):**
  `DSV41_INDEXER_LOGITS_BUDGET_BYTES=1073741824` (adapter default 2 GiB) — a robustness default that lifted the head's
  MemAvailable floor 11.5–11.8 → 14.9 GiB; not a speed win.

## Hardware

Sparks 1–4 of the 8-node CRS804 rack ([`hardware/dgx-spark-4x.yaml`](../../../hardware/dgx-spark-4x.yaml)): GB10,
128 GB unified memory each (121.69 GiB visible to torch), driver 595.84, CUDA 13.2 host, kernel 6.17.0-1031-nvidia,
DGX OS 7.2.3, 4 KiB pages, GPU clocks capped 2200 MHz (`clocks.gr` under load 2190). Two 200G CX-7 ports per node
(`rocep1s0f0` 10.0.0.N, `roceP2p1s0f0` 10.0.1.N), lossy RoCEv2, MTU 9000. Head spark1, API `:8888`. Per-node snapshots
in [`results/…/system/`](results/2026-10-19-sglang-ladder-r2/system/).

The vLLM sibling runs on Sparks 5–8 through the same switch. A deliberate A/B (2026-10-20 05:00–05:07,
[`raw/contention/`](results/2026-10-19-sglang-ladder-r2/raw/contention/RESULTS.md)) found no cross-pod effect: B 93K
6,219 quiet → 6,242 with C prefilling back-to-back; C 5,993 → 6,009 with B prefilling.

## Software

| piece | pin |
|---|---|
| Stack | [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4) v2.4 (CHANGELOG 2026-10-12, merge `37c85527`). Our lane dir is a non-git copy (2026-10-18); its content matches the v2.4 tree before the 2026-10-17 ring commits, i.e. consistent with `27616bec`, **exact commit not recorded**. `start-tp4.sh` `ada373a3…`, `boot.py` `7fccb3cb…` |
| SGLang | `dsv4.1` branch `f80c91a4b` (2026-09-16); package metadata reads `0.0.0.dev1+gda64c5cbb` (the base image's commit) |
| Base image | `lmsysorg/sglang:dev-dsv41`, image ID `381b27ff…`, 2026-09-11 |
| Image | `dsv41-4x-spark:canary-roce`, image ID `sha256:609343dd905f07999d9a60fa6fdc9801d83cb9a4b07a3c62d49d0531f63e4782`, local build, not pullable |
| Libraries | torch 2.13.0+cu130, flashinfer 0.6.18, NCCL 2.30.7, b12x_next `a7d7d29b` |
| Instrument | `scripts/` — the shared two-lane tools (v41bench v1, prefill_reps, long_check, mixed_probe, qeval 75 tasks, gates, spec_by_cat) with box paths replaced by `BASE_URL` / env |

## Launch

On the head node, with passwordless ssh to the three workers and the checkpoint at `$HOME/models/DeepSeek-V4.1-Flash-1017`:

```bash
export SPARK_HOSTS="spark1 spark2 spark3 spark4"      # rank order; ssh names
bash scripts/launch-sglang-tp4.sh doctor               # clones knapcio @ 27616bec into $STACK_DIR if absent
bash scripts/launch-sglang-tp4.sh build                # BUILD_DOCKERFILE=Dockerfile.canary-roce, every node
bash scripts/launch-sglang-tp4.sh share && bash scripts/launch-sglang-tp4.sh pack   # first time (NFS + Engram pack)
bash scripts/launch-sglang-tp4.sh serve
```

`launch-sglang-tp4.sh` renders [`scripts/env.tp4.recipe`](scripts/env.tp4.recipe) (the scrubbed `.env.tp4` that
served these numbers: `.env.tp4.B0` + the B1a line) into `$STACK_DIR/.env.tp4` and execs knapcio's `./start-tp4.sh`. It
refuses to start unless the rendered env has `SKIP_PREPARE=1`, the 1017 revision and the B1a override, and checks 48/48
shards. The exact engine argv boot.py produced is in
[`launch-command.txt`](results/2026-10-19-sglang-ladder-r2/launch-command.txt). Fabric addresses (10.0.0.N) are this
rack's documented topology; edit them for yours. Before every boot: `drop_caches`, `compact_memory`, MemAvailable
≥ ~110 GiB per node.

## Verify

```bash
export BASE_URL=http://<head>:8888
curl -fs $BASE_URL/v1/models
python3 scripts/gates.py ./gates-out          # models + exact-reply smoke + tool call round trip (metrics optional)
python3 scripts/qeval.py run mycheck          # 75 tasks; expect 71/75
# Full battery (sends real load — never on a lane serving users):
DSV41_ENCODING_DIR=$HOME/models/DeepSeek-V4.1-Flash-1017/encoding bash scripts/b-battery.sh ./battery B1a
```

Expect in the engine log: `RoCEnante ready: world=4 hcas=rocep1s0f0,roceP2p1s0f0 gid_index=3`, a
`max_total_num_tokens` between ~4.5M and ~5.9M, then `The server is fired up`. Performance bar (quiet lane): C1
per-stream ≈ 96 ±3%, C6 ≈ 253–262, 93K reps median ≥ 5.8K, 300K needles PASS.

Note: on every round-2 arm the on-box `gates.py` exited rc=1 at its `/metrics` read (HTTP 404 — this launch serves no
`/metrics`) **before** writing its results, so the tool-call gate result was never recorded. The published copy makes
`/metrics` optional; the gate has not been re-run (the lane is live).

## Results

### Round 2 ladder (2026-10-19 18:57 → 22:52 CDT, Sparks 1–4, one change per boot vs `.env.tp4.B0`)

Bundle: [`results/2026-10-19-sglang-ladder-r2/`](results/2026-10-19-sglang-ladder-r2/) —
[`RESULTS.md`](results/2026-10-19-sglang-ladder-r2/RESULTS.md),
[`throughput.csv`](results/2026-10-19-sglang-ladder-r2/throughput.csv), receipts per arm under `raw/<ARM>/`.
v41bench v1, T=0, thinking off; C1 shown as aggregate (run 1 / run 2) and per stream; C4/C6 aggregate.

| arm | change | C1 agg | C1 per stream | C4 | C6 | 93K single / 3-rep median | needles 20K / 131K / 300K×3 | KV tokens | qeval | mixed TTFT s | head floor GiB | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | v2.4 block, cold JIT | 81.6 / 83.5 | 94.8 / 97.0 | 206.0 | 249.9 | 5,003 / 4,955 | 4,229 / 4,273 / 2,227, 4,099, 4,157 | 5,356,032 | 71/75 | 1.9 / 2.4 | 11.5 | control |
| B0ctl | warm reboot | 82.2 / 82.9 | 95.0 / 96.4 | 208.2 | 261.6 | 3,145 / 3,304 † | 2,934 / 3,835 / 2,525, 2,530, 2,656 | 5,417,216 | 71/75 | 3.2 / 4.5 | 11.8 | control; † outlier |
| **B1a** | indexer budget 2→1 GiB | 80.0 / 82.4 | 95.0 / **96.2** | 203.8 | 252.8 | 5,847 / **6,187** | 5,283 / 4,427 / 3,484, 4,029, 4,741 | 5,353,728 | 71/75 | **1.7 / 1.8** | **14.9** | **serving** (robustness) |
| B1b | budget 2→4 GiB | 81.9 / 81.2 | 94.8 / 94.0 | 205.0 | 255.2 | 5,906 / 6,080 | 5,358 / 5,028 / 2,653, 4,096, 4,404 | 4,519,680 | 71/75 | 4.8 / 1.7 | 12.9 | revert |
| B2 | `--swa-prefix-tails` 128→64 | 78.7 / 81.2 | 94.5 / 94.6 | 209.7 | 258.6 | 5,730 / 6,086 | 5,344 / 5,975 / 2,992, 3,990, 4,885 | **5,927,424** | 71/75 | 1.9 / 1.7 | 13.3 | capacity option |
| B3 | `--prefill-decode-interval` 2→4 | 82.2 / 83.7 | 95.3 / 96.6 | 208.8 | 259.4 | 4,361 / 5,290 | 5,312 / 4,784 / 2,973, 3,053, 3,003 | 4,817,664 | 71/75 | 1.7 / 1.9 | 14.2 | revert (long prefill −13%, 300K −30%) |
| *B1a serving boot* | *2026-10-20 05:00, contention A/B* | | | | | *quiet 6,219; with lane C prefilling 6,242* | | *5,219,072* | | | | |
| *prior round* | *published Oct 18, Sparks 5–8, pre-hygiene* | *84.8* | | | *262.8* | *5,346* | *— / — / 1,791, 2,648, 3,029* | | | | | *labelled, not re-run* |

B1a C1 per stream by class (run 2): coding 134.1 · format 129.4 · math 114.1 · json 105.0 · reasoning 86.0 · prose 74.0 ·
summary 65.8 · narrative 61.0. All needles PASS on every arm. `max_running_requests` = 16, so C1–C6 are not slot-capped.

Read: decode has no lever in this set (C1 and C6 within ±3% across six boots). Prefill is 6.1–6.2K at 93K whenever the
host is in the state of the B1a–B2 boots; B0 (cold JIT, first boot of the night) read 5.0K and B0ctl 3.3K. B2 is the
only attributable KV change (+9%). B1a is kept for its memory floor and the best mixed TTFT, not for speed.

## Known limits

- **HAZARD — `SKIP_PREPARE`.** Any env that un-sets `SKIP_PREPARE=1` makes `boot.py` `snapshot_download` its
  hard-coded `REVISION = fb2764a5…` **into `MODEL_DIR`**. On 2026-10-19 (the ~20:27 CDT B1a boot) this overwrote
  checkpoint shards on spark1; the copy was restored from spark5 21:08–21:21 and re-checked 48/48
  ([`raw/weights/`](results/2026-10-19-sglang-ladder-r2/raw/weights/)). B0 and B0ctl ran before the incident on the
  original copy; B1a–B3 and the serving boot ran on the restored copy. The launcher aborts without `SKIP_PREPARE=1`.
- `state-tp4/launch.json` says `"revision": "fb2764a5…"`. That is boot.py's constant, **not** the mounted checkpoint;
  the 1017 checkpoint (`1bae0e10…`) is what is mounted at `/models/DeepSeek-V4.1-Flash`.
- **B0ctl's 93K window** (10-19 20:04–20:21, 3,145 / 3,304 tok/s) is an unexplained outlier: the same config read
  6.1–6.2K in four later boots, and the contention A/B says it was not the fabric.
- **300K first touch:** run 1 is 2.2–3.5K tok/s on every arm. Runs 2–3 reach 4.0–4.9K on B0/B1a/B1b/B2 but stayed
  2.5–2.7K on B0ctl and 3.0–3.1K on B3. Report run 1 separately.
- NV_ERR_NO_MEMORY bursts in dmesg during B0's needle window (head, 271 events per the ladder notes) did not recur as
  a burst in later arms (dmesg counts only rotated down). The 1 GiB indexer budget lifts the head floor 11.8 → 14.9 GiB.
- **KV pool varies by boot** (4.52M–5.93M) independent of the arm — knapcio's documented fast-loader variance.
- **Fidelity is thin.** No teacher-forced Δlogprob, no public bench run by us, tool-call gate not recorded (see Verify),
  per-class DSpark acceptance not measured (no `/metrics`; `spec_by_cat.py` reads vLLM counters). qeval 71/75 is a
  content score. knapcio's own v2.4 release gate (KL 0.0045 / 0.0061 vs v2.3, qeval 72/72/72) is theirs, not ours.
- Boot ~6.4 min warm, ~8 min cold JIT (vLLM sibling ~1.5 min warm).
- The image is a local build (digest = docker image ID). The lane dir is a copy, not a clone; the exact knapcio
  commit is inferred (`27616bec`), not recorded. THP read `madvise` on all four nodes on 2026-10-20; the round-2 chain's
  hygiene did drop_caches + compact_memory + a MemAvailable gate, not THP.

## Rollback

`SPARK_HOSTS=… bash scripts/launch-sglang-tp4.sh stop` (= knapcio `./start-tp4.sh stop`; if a worker container
survives the stop, `docker stop dsv41-worker` on that node by name — the chain had to do that on 4 of 7 stops). Sparks
1–4 previously served Lane A (TensorFold EXL3), kept stopped with its containers and restorable with its own relaunch
script (not published here). For the same model on other hardware, use the vLLM sibling (Sparks 5–8) or the GB300
recipe.

## Credits

- **knapcio** — [DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4):
  the v2.4 production block, EP1 on b12x_next, the adapters (indexer backport, fast load, prefill SP, prefill bypass,
  certified head …) and the launcher this recipe wraps.
- **Mia (MiaAI-Lab)** — [DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks),
  the upstream recipe knapcio's is a downstream profile of (launcher, Engram NVMe row store, b12x routing, memory model).
- **rhys101** — the SG17 RoCEnante SGLang overlay and the SG18 prefill TP split.
- **local-inference-lab** (Luke Alonso, Jason / original-el8) — [b12x](https://github.com/local-inference-lab/b12x):
  RoCEnante and the fused MoE kernels.
- **kpham-sgl** — [sglang#39187](https://github.com/sgl-project/sglang/pull/39187), the bounded dense-indexer
  transient whose budget (`DSV41_INDEXER_LOGITS_BUDGET_BYTES`) is the one knob we changed.
- **hushengkai** — the head-memory and short-request-starvation reports behind v2.4's chunk/fraction and prefill
  bypass ([MiaAI-Lab#44](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/44)).
- SGLang `dsv4.1` branch authors (BBuf and contributors), and **DeepSeek** for the model and its in-checkpoint drafter.

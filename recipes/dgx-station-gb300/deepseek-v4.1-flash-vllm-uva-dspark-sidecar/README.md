# DeepSeek-V4.1-Flash · 1M context on a DGX Station GB300 + RTX PRO 6000 sidecar

The 510 GB DeepSeek-V4.1-Flash-1017 checkpoint does not fit one GB300. The GB300-only recipe in this repo
solves that by offloading 60 GiB of routed experts to pinned Grace memory over NVLink-C2C — which works, and
leaves a serial ~350 GB/s read on the critical path of every decode step. This recipe takes the other road:
keep the hot experts in HBM and move the cold ones to the **RTX PRO 6000 Max-Q in the same chassis, where
they are computed**, not read. Only activations cross PCIe.

Same box, same checkpoint, same harness, bracketed controls in one window:

| | GB300 only (sibling recipe, v23) | GB300 + RTX 6000 sidecar (this recipe) |
|---|--:|--:|
| Decode C1, real text 8.2K/1,024 | 187.5 tok/s | **279.5 tok/s** (+49%) |
| Decode C16 | 524.5 | **1,576.9** (3.0×) |
| Decode C32 | 936.9 (max-num-seqs 24, queues) | **2,400.3** (2.6×) |
| TTFT, 8.2K prompt | 0.66 s | **0.19 s** |
| Cold 64K prefill | 23,439 tok/s | **50,119 tok/s** |
| 1M-token prompt, first token | 81.4 s (12,262 tok/s) | **33.2 s** (30,069 tok/s) |
| KV pool at 1M ctx | 2,504,891 tokens (util 0.97) | **4,400,013 tokens** (`--kv-cache-memory` 8.5 GiB; 2,537,081 at util 0.97) |
| Host memory pinned | ~253 GiB (Engram + cold experts) | Engram only — cold experts are on the card |
| Extra process you must keep alive | — | one sidecar container + a fail-closed watchdog |
| Wins on | simplicity, no second GPU, prefill is throttled the same | decode at every concurrency, TTFT, long-context reads |

Both recipes serve the same weights and stay published; this one is the faster lane on this hardware, the
sibling is the one that needs nothing but the GB300.

> **Lineage, stated up front.** The production shape of this recipe — two-step `sidecar` → `server` launch, the
> b12x peer tier-2 protocol, the `rowmap-mix-v1` calibration, the published same-chassis numbers, and the
> `kbench_chat_ds` harness this recipe is measured on — is **RonanLabsAI's**
> ([dgx-station-frontier](https://github.com/RonanLabsAI/dgx-station-frontier),
> `recipes/dsv41-flash-sidecar`, Apache-2.0), built on **original-el8's** hook. What is ours is the delta:
> the 295/89 map, the agreement gate run on real prompts, the ack-before-bookkeeping sidecar fix, the
> fail-closed watchdog and the bracketed measurement discipline. See [Credits](#credits).

## 2026-10-23 — experimental variant v25m: a mixed-traffic map (no code change)

The 295/89 split in `rowmap-static-v1.json` was fitted on agent traffic. On Japanese/German prose, the GB300 spent
**11.8 % of wall time waiting on the RTX** (0.5 % on agent English, measured from the peer shm stall counters).
v25m changes **one file**: `rowmap-mixed-ar-v1.json`, the top-295 experts per layer by cumulative decode counts
collected on ~100k JP/DE and ~105k agent/tool tokens during the
[GB300-only adaptive-residency campaign](../deepseek-v4.1-flash-vllm-uva-dspark/README.md). Same image, hook, b12x, flags and KV.

| post-60 s medians, tok/s | v25 (static map) | **v25m (mixed map)** |
|---|---:|---:|
| mixed JP/DE + agent, C1 aggregate, **held-out prompts** | 208.9 | **221.1** (+5.8 %) |
| — per request, JP/DE / agent | 179.5 / 250.8 | **206.7** / 246.4 |
| mixed, C4 aggregate, held-out | 560.4 | 558.5 (no change) |
| shift protocol JP/DE / agent, C1 (in-sample prompts) | 186.0 / 244.3 | **206.9** / 247.9 |
| GB300 stall on the RTX, JP/DE / agent | 11.8 % / 0.5 % | **0.5 %** / 0.4 % |

- **What it buys:** single-stream off-domain prose, +11–15 %. At C4 the GB300's own step time already hides the RTX work.
- **Fidelity (Card GV, 2026-10-24): 9 of 10 gates pass, the bring-up cosine gate fails.** Greedy parity cannot gate this lane (the same map self-agrees 14/50), so the gates are SC2's. Teacher-forced flips PASS against both v23 and v25 static; GSM8K-1319 96.82 vs 96.97; BFCL dev/held-out **identical** to v25 (556/600, 1018/1311); gauntlet 12/12; needle 500/500; tool emission, effort and image probes pass; 0 sidecar timeouts. **The bring-up cosine gate misses: min cos 0.99811 vs the 0.999 bar.** That is one of 1,439 gated checks, at layer 39 (the worst layer on every map), and the rest of the distribution matches v25 static (p1 0.99968 vs 0.99970). The same gate rejected the 285/99 map, so the bar stays where it is and v25m stays **experimental**. **Follow-up (Card GV2):** restoring layer 39 to v25's exact static split did not fix it (min 0.99831; worst check again layer 39, where the cold experts carried ~0 % of the output norm); see [`results/2026-10-24-gv2-l39-static/`](results/2026-10-24-gv2-l39-static/). **Correction (2026-10-24 08:15):** an earlier version said the v25m miss was check noise, not the map. That overreached. GV2 shows layer 39's checks are noisy when the cold experts carry ~0 of the output, but v25m's own failing check had them at 7.9 %, so GV2 does not clear v25m. The open question goes to a new, pre-registered gate (R2, below). **Cosine gate R2, registered 2026-10-24 before re-judging** ([`results/2026-10-26-gs-cosine-r2/`](results/2026-10-26-gs-cosine-r2/)): judge the sidecar's error as a fraction of the whole layer output, e = rel_diff × cold share, against a bar of 1.5 × the worst e from three v25-static check boots (the SC0 boot plus two more after the soak, 2026-10-26). Every archived boot is re-judged and the table published whatever it says. Receipts: [`results/2026-10-24-gv-v25m-gates/`](results/2026-10-24-gv-v25m-gates/).
- **Same-instrument speed vs v25:** real-text English C1 273.2 vs 279.2 (−2 %), knee C8 887 vs 934 (−5 %), C16 1,411 vs 1,409. The mixed map trades a little English-only speed for +15 % on JP/DE.
- **Soak:** 48 h on v25m, restarted after Card GV2 at 07:12 Sat 10-24 → 07:12 Mon 10-26 CDT, fail-stop watchdog armed. Card GS (static check boots + R2 re-judge) runs right after.
- **Card:** [`card-v25m.png`](card-v25m.png) (base v25 card: [`card.png`](card.png)).
- **Launch:** both steps with `ROWMAP=rowmap-mixed-ar-v1.json` (the file goes in the hook directory), for example `NAME=… ROWMAP=rowmap-mixed-ar-v1.json bash launch_sidecar.sh sidecar`, then the same for `server`.
- Receipts, fitter, harness and the layout-separability test: [`results/2026-10-23-ar5-mixed-map/`](results/2026-10-23-ar5-mixed-map/).

## What this runs

DeepSeek-V4.1-Flash-1017 (MXFP4 routed experts, FP8 attention/Engram, in-checkpoint DSpark drafter) at the
full **1,048,576-token** context, on:

* **GB300** — `vllm serve` (image `sha256:3af6a45b…`, nightly-af7f9488), TP1, `--moe-backend deep_gemm_mega_moe`,
  295 of the 384 experts per MoE layer resident in HBM, Engram pinned in host memory (`{"cpu_offload": true}`),
  32 seats, `fp8_ds_mla` KV, DSpark k=5/k=3/k=3 by batch size.
* **RTX PRO 6000 Max-Q** — original-el8's peer tier-2 server (b12x kernels) holding the other **89 experts
  per layer** and computing their contribution on demand. The two processes share one `/dev/shm` buffer with
  a device-side sequence counter: the GB300 publishes a step, the card answers, the GB300 adds the result.

Why it is faster: the cold-expert contribution stops being a serial C2C read and becomes a parallel GEMM on a
second GPU. Measured step time on this chassis falls from ~14.6 ms (Grace offload, C1) to ~10.6 ms, which is
where the 1.5–1.9× decode comes from. Prefill gains even more: cold rows are computed in parallel instead of
streamed, which is why the 1M first token drops from 81 s to 33 s.

**The correctness trap that shapes this whole recipe:** if the sidecar misses a deadline, the hook does not
error — it *skips that layer's cold-expert add and keeps emitting tokens*. Two mechanisms cover it: an
effectively-infinite spin limit (`VLLM_EXP_PEER_SPINS=2e9`, so a dead sidecar hangs rather than corrupts) and a
fail-closed watchdog that kills the lane on any timeout, stall or dead sidecar. The bring-up gate is a
per-layer cosine check against the GB300's own cold path.

## Hardware

One NVIDIA DGX Station GB300 with an RTX PRO 6000 Blackwell Max-Q add-in card. Profile:
[`hardware/dgx-station-gb300-rtx6000.yaml`](../../../hardware/dgx-station-gb300-rtx6000.yaml).

| | value |
|---|---|
| Compute GPU | GB300, **250.7 GiB** visible / 249.8 GiB CUDA-allocatable (not the 288 GB spec), 1,300 W cap |
| Sidecar GPU | RTX PRO 6000 Blackwell Max-Q, **95.6 GiB**, 300 W cap (325 W max), 96 GB class |
| GPU↔GPU | PCIe via the Grace root complex; **P2P unsupported** (`topo -p2p r` = NS) — hence `/dev/shm` |
| Host | 494.5 GiB LPDDR5X, Grace 72× Neoverse-V2, NVLink-C2C |
| Driver / CUDA | 595.91.07 / 13.2 (container runtime 13.0.2) |
| OS / kernel | Ubuntu 24.04.5 LTS / 7.0.0-1019-nvidia-64k |
| Power | 1,600 W Station budget shared by both GPUs, `vsloshd` inactive |

Two settings that are not optional:

1. **CDI for the RTX card.** DGX OS's `/usr/libexec/dgx-desktop-docker-gpus.sh` hides the VGA-class card from
   `--gpus`. Generate a CDI spec (`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`) and pin by UUID
   (`--device nvidia.com/gpu=<RTX UUID>`). The launcher refuses to start the sidecar on a card already holding
   ≥2,048 MiB, because that card also drives the display/JetKVM.
2. **Local NVMe for the checkpoint**, and persistent JIT caches (vLLM, FlashInfer, DeepGEMM, tilelang, triton,
   b12x). The sidecar reads expert tensors out of the shards layer by layer at prep time.

## Software

| piece | pin |
|---|---|
| image (both GPUs) | `vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063` = nightly `af7f9488`, vLLM `0.30.1rc1.dev245+gaf7f9488c`; torch 2.13.0+cu130, FlashInfer 0.7.0, triton 3.7.1. The Hub tag has been rotated off — pull by digest or keep a `docker save` copy |
| hook (GB300) | [original-el8/dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research) `@87af662dd0a70f72c250c2085cf12eea2be8d882`, `deepseek-v4.1-flash/m3/hook` (Apache-2.0) — `mega_peer_hook.py`, `sitecustomize.py`, `peer_tier2.py`, `peer_fused2.py`, `ll_gemm.py`, `mxfp8_splitk.py`, `rowmap-static-v1.json` |
| sidecar server | el8 `peer_server2.py` @87af662 + [`patches/peer-server2-local.patch`](patches/peer-server2-local.patch) (path edits + ack-before-bookkeeping), run as `/sc/peer_server2_local.py` |
| sidecar kernels | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) `@2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88` (Apache-2.0) |
| overlay | vLLM [#58132](https://github.com/vllm-project/vllm/pull/58132) decoder bounded replay, 5 files bind-mounted over `dist-packages` |
| launcher lineage | [RonanLabsAI/dgx-station-frontier](https://github.com/RonanLabsAI/dgx-station-frontier) `recipes/dsv41-flash-sidecar` (Apache-2.0), adapted for Grace-only/CAND arms and our caches |

Every mounted file's sha256, read **inside the running containers**, is in
[`results/2026-10-22-v25-sidecar/raw/hook-lock.txt`](results/2026-10-22-v25-sidecar/raw/hook-lock.txt).
Checkpoint identity (revision, 49 shards, 510,311,114,235 B, config/index hashes) is in
[`raw/weights-1017-receipt.txt`](results/2026-10-22-v25-sidecar/raw/weights-1017-receipt.txt).

## Launch

Two steps, in this order — the server refuses to start unless the sidecar log says `serving`.

```bash
export GB300_UUID=$(nvidia-smi -L | awk -F'[( )]' '/GB300/{print $2}')
export RTX_UUID=$(nvidia-smi -L   | awk -F'[( )]' '/RTX PRO 6000/{print $2}')

# 1. the card: 40 layers of cold experts, 560 CUDA graphs, ~68 GB, ~46 s
bash scripts/launch-sidecar.sh sidecar

# 2. the GB300: vLLM with the hook, 295 hot / 89 cold, 1M context, 32 seats, KV pool declared at 8.5 GiB
EXTRA="--kv-cache-memory 9126805504" bash scripts/launch-sidecar.sh server
```

Defaults you may need: `M3_DIR` (`~/dsv41/m3` — hook checkout, overlay tree, logs, `local/peer_server2_local.py`),
`B12X_DIR` (`~/dsv41/b12x`), `MODEL_DIR`, `CACHE_DIR` (`~/dsv41/cache-a`), `ROWMAP`
(`rowmap-static-v1.json` = 295/89), `NAME`, `SEQS` (32), `GPU_UTIL` (0.97), `MAXLEN` (1048576), `PORT` (30006),
`PEER_SPINS` (2e9), `REPLAY_OVERLAY` (1), `EXTRA` (empty). `bash scripts/launch-sidecar.sh stop` stops
server-then-sidecar and keeps both containers.

**`EXTRA` carries the KV size and defaults to empty.** Without it the pool is whatever util 0.97 leaves after
vLLM's memory profile (4.9 GiB = 2,537,081 tokens) and nothing errors. With it, the boot log must say
`reserved 8.5 GiB memory for KV Cache as specified by kv_cache_memory_bytes config` and
`GPU KV cache size: 4,400,013 tokens, Maximum concurrency for 1,048,576 tokens per request: 4.20x`.
`docker start` of an existing container keeps its flags; only fresh launches need `EXTRA`.

**Bring-up gate** (do this once per new map/image — it is what proves the sidecar path is numerically right):

```bash
MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1 bash scripts/launch-sidecar.sh server
python3 scripts/gate_cosine.py ~/dsv41/m3/logs/<container>.log
```

The gate reads only post-startup `MEGA_PEER check …` lines (vLLM's dummy profile run emits all-zero checks
that must be ignored) and passes on **40/40 layers, ≥32 cold rows, min cos ≥ 0.999, 0 timeouts**.

**Keep it alive** — the watchdog is part of the recipe, not an accessory:

```bash
python3 scripts/sidecar_watchdog.py --server "$NAME" --mode enforce \
  --status ~/dsv41/m3/logs/sidecar_watchdog.status.json \
  --log    ~/dsv41/m3/logs/sidecar_watchdog.log &
bash scripts/soak_check.sh <ssh-host> "$NAME"     # one health sample from another machine
```

`--rollback <container>` makes a trip auto-start the previous lane; the reference lane runs **fail-stop**
(no `--rollback`): a wrong-token risk kills the lane and leaves it down.

## Verify

| gate | how | bar |
|---|---|---|
| bring-up numerics | `scripts/gate_cosine.py` on a check-mode boot | 40/40 layers, ≥32 rows, min cos ≥ 0.999, 0 timeouts — measured **0.99906** |
| lane health | both containers up, `/v1/models`, watchdog `trip=null`, `words[2]=0`, boot KV line 4,400,013 tokens | `scripts/soak_check.sh` prints `OK` |
| KV capacity | four ~1M contexts resident at once, 16K tokens decoded each (`results/2026-10-22-kv85/scripts/kv2_coresident.py`) | 4 running / 0 waiting, 0 preemptions, 0 errors, 0 trips, 0 timeouts — measured KV peak **75.3%** |
| output quality | teacher-forced flips vs a same-map floor, BFCL dev+held-out, GSM8K-1319, 500-needle, gauntlet, reasoning-effort vocabulary, one image request | SC2's 12 pre-registered gates |
| drafter | per-class decode wall-clock, 20 cells × {C1, C8} | no cell worse than −3% vs both controls |
| speed | bracketed ladders with a control on each side | knee C1 ≥ 198, knee C16 ≥ 974, sidecar C16 ≥ Grace-only +3% |
| soak | 48 h, sampled every 5 min — restarted with the KV change: 2026-10-22 18:52 → 2026-10-24 18:52 CDT | no timeout, no trip, no stall |

Nothing here is promoted on throughput alone: the cosine gate and the 0-timeout soak are pass/fail, and a
timeout is treated as a correctness bug rather than a slow request.

## Results

### KV pool declared at 8.5 GiB (2026-10-22 evening, `results/2026-10-22-kv85/`)

The lane as first promoted let `--gpu-memory-utilization 0.97` size the KV pool: 4.9 GiB after weights
(233.5 GiB), activation (3.9) and CUDA graphs (1.2). vLLM's own boot line said 10.37 GiB would "fully utilize"
the GPU. Declaring the pool with `--kv-cache-memory 9126805504` (8.5 GiB) changes nothing else — same image,
same autotune store, same kernels:

| | util 0.97 | **8.5 GiB declared** |
|---|--:|--:|
| KV tokens at 1M max-model-len | 2,537,081 | **4,400,013** (+73%) |
| 1M-token requests at once (boot line) | 2.42× | **4.20×** |
| C1 decode, real text 8.2K/1,024 | 279.5 (mean of 2 boots) | **275.1** (mean of 6, sd 5.2) |
| C16 decode | 1,576.9 | **1,575.9** |
| 1M needle, first token | 33.2 s | 35.5 s (recalled; 1M-depth decode 256 tok/s) |

The C1/C16 rows are not a same-window A/B — the kernels and tune store are identical, and the gap is inside
live FlashInfer tuning's boot-to-boot spread.

**Stress, on the live lane:** a ~1M needle + a cold 131K prompt + C16 all at once (all pass), then four ~990K
prompts each decoding 16,000 tokens so all four are resident together — **4 running / 0 waiting, KV usage peak
75.3%**, 0 preemptions, 0 engine errors, 0 watchdog trips, sidecar timeout counter unchanged at 0. The GB300 sat
at **255,308 of 256,703 MiB** at the peak (1,395 MiB free). That margin is why we stopped at 8.5: 9.5 GiB would buy
+0.5M tokens and leave ~370 MiB, and an out-of-memory on this lane is a crash, not a slowdown.

The speed cells use **RonanLabsAI's `kbench_chat_ds.py`**, the same harness as every other real-text number in
this recipe.

Side finding: a cold ~1M prefill holds up the queue — the 131K prompt next to it waited 34 s for its first token
and the C16 aggregate fell to 718 tok/s while the 1M prompt was read. See Known limits.

### The sidecar round (`results/2026-10-22-v25-sidecar/`)

Bundle: [`results/2026-10-22-v25-sidecar/`](results/2026-10-22-v25-sidecar/) — `throughput.csv` (every cell
with its arm and receipt), `power-by-phase.csv`, `system.json`, and the raw verdicts.

**Decode, real text (8.2K in / 1,024 forced out, thinking off, 0% prefix hits)** — bracketed
CTL1 → CAND1 → CAND2 → CTL2 on one box, sidecar values are the mean of the two independent boots:

| cell | incumbent v23 | sidecar | Δ |
|---|--:|--:|--:|
| C1 | 187.5 | **279.5** | +49% |
| C4 | 321.7 | **693.2** | +116% |
| C16 | 524.5 | **1,576.9** | 3.0× |
| C32 | 936.9 | **2,400.3** | 2.6× |
| TTFT 8.2K | 0.66 s | **0.19 s** | 3.5× lower |
| acceptance (C1) | 2.955 | 2.961 | +0.5% |

**Agent-text knee** (C1 222.8 vs 192.0, +16%; C8 891.1 vs 598.6/601.9, +49%; C16 1,414.5 vs 937.0, +51%) and
**per-class C1** (prose 294.8 vs 232.3, structured 299.1 vs 250.3, code 408.9 vs 293.4, shell 413.7 vs 316.9,
tool-JSON 367.2 vs 300.0) are in `throughput.csv`. Note the per-class fixture reads higher than the 8.2K
harness on every arm — do not mix the two.

**Long context (SC5):** 999,303-token prompt → first token in **33.23 s** (30,069 tok/s), needle recalled;
128K cold prefill 2.61 s; C16 after 1M 32/32 at 1,565.9 agg; C1 decode after 1M 276.5 vs 279.5 before. KV pool
2,537,081 tokens = 2.42× one 1M request at util 0.97 (superseded: 4,400,013 / 4.20× with the declared 8.5 GiB
pool above).

**Against Ronan's published lane — context, not a competition.** On the cells his 1017 sidecar log publishes for
the same harness, this lane reads C1 279.5 vs 273.6, C4 693.2 vs 689.1, C16 1,576.9 vs 1,545.5, C32 2,400.3 vs
2,276.6 (+2% / +1% / +2% / +5%). Same mechanism, same chassis; the differences are our 295/89 map (his 285/99),
the ack-order fix and live-tuned kernels. That is why [Credits](#credits) leads with him.

**Quality (SC2, all 12 gates PASS):** teacher-forced flips inside the same-map control floor; BFCL dev
92.67 vs 92.33 and held-out 77.65 vs 77.88; GSM8K-1319 96.97 = incumbent; needle ladder 500/500 exact;
gauntlet 12/12; reasoning-effort vocabulary and an image request both pass; 20/20 per-class decode cells
within −3%; 20-minute soak with **0 sidecar timeouts**.

**Placement (SC3):** the nested 285/99 map would buy ~6.09M KV tokens but **failed its pre-registered cosine
bar** (0.99885 vs 0.999 on two layer-39 T=128 checks), so the uniform 295/89 map serves. Its speed with the
sidecar is unmeasured.

**Power (SC5):** GB300 module p99 **1,353 W** during 64K prefill, 1,263 W at C32 decode; `sw_power_cap` active in
89.5% (64K prefill) / 85.7% (C32 8K prefill) / 2.4% (C32 decode) of 1 s samples; SM clock p50 drops to 1,980 MHz
in prefill (2,062 incumbent); RTX p99 ≤ 194 W against its 300 W cap. Full table in `power-by-phase.csv`.

## Known limits

* **The power cap binds in prefill** — and it bound on the incumbent too (65% / 52% of samples). The SC5 bars
  (<1% throttled, module ≤1,190 W) were set too tight for this chassis; every number here was measured under
  that throttling. A 240 V / uncapped retest is the obvious next measurement.
* **A silent sidecar timeout is wrong output.** Infinite-spin + watchdog, or do not serve. Do not run the lane
  in `--mode report` and call it production.
* **Uniform rowmap only**, unless you port el8's per-layer-plan change to `peer_server2`. Non-uniform maps also
  change the cold path's cost; ours was never speed-measured because its gate failed first.
* **Effort mapping moved:** this image maps `low`→50, `high`→75 (default), `max`→100; the GB300-only lane's
  0909 image maps `low`→25, `high`→50. Clients that send `low` think twice as long after a move to this lane.
* **One runaway is possible at temp 0** (GSM8K item 1304 hit the 131,072 cap once; a 32K cap probe read 0/32
  and pinned efforts 0/8). Attributed to the image's effort-75 tail, not the sidecar; a client `max_tokens` is
  the only real guard.
* **Per-class acceptance is not measured** in this round (blended 2.96 tokens/step only).
* **Live FlashInfer autotune**, no pinned tune file: a few percent of C1 varies boot to boot, and the autotuner
  reports some `mxfp8_gemm` shapes falling back to `tactic=-1`.
* **Replay ON vs OFF is one rep**: agent replay r2 612.1 OFF vs 480.3 ON, but TTFT is better ON (0.19 vs 0.29 s).
  The lane runs ON. The +27% is not a law until repeated.
* **vLLM v0.31.0 was NOT adopted** — quality identical, but knee C8 read −3.8% and the runaway cell tripped.
* **KV is declared, so vLLM no longer checks activation headroom at boot.** Measured margin: 1,395 MiB free on
  the GB300 at the stressed peak. Vision requests and C32 prefill bursts were not in that stress. 9.5 GiB was
  declined (~370 MiB would remain).
* **`EXTRA` defaults to empty** — a fresh launch without `EXTRA="--kv-cache-memory 9126805504"` silently serves
  2,537,081 tokens. Check the boot's KV line.
* **A cold 1M prefill stalls everything behind it** (131K neighbour: 34 s to first token; C16 aggregate 1,576 →
  718 tok/s while it ran). Keep long batch jobs off the interactive lane, or accept the stall.
* **KV routes not taken:** KV quantization (`nvfp4_ds_mla`) changes numerics; more cold experts on the card
  (89 → 99/109 per layer, +3.6M/+7.3M tokens) is modeled at −5 to −11% C1; HiSparse host-resident KV exists in
  this image but is not wired for DeepSeek-V4.1 or the MegaAttn backend.
* **Boot 527 s warm**; 46–50 s of it is the sidecar. Cold JIT caches are ~30 min.
* **The Docker Hub tag is gone**; the digest is the only reliable pull.

## Rollback

```bash
# 1. stop the guard first (it would otherwise see a dead server and trip)
pkill -f sidecar_watchdog.py
# 2. server, then sidecar
bash scripts/launch-sidecar.sh stop
# 3. bring the previous lane back (kept, stopped, same box)
docker start dsv41-vllm-1017-v23-pinexact-RETIRED-REF     # binds in ~260 s
```

To undo only the KV change, start the kept util-0.97 pair instead (sidecar first, then server):
`docker start dsv41-vllm-1017-v25-sidecar-kv4.9-KEPT-rtx && docker start dsv41-vllm-1017-v25-sidecar-kv4.9-KEPT`
(it bound in ~7 min on the reference box).

With no previous lane on the box, the fallback is the GB300-only sibling recipe
([`../deepseek-v4.1-flash-vllm-uva-dspark`](../deepseek-v4.1-flash-vllm-uva-dspark)): same checkpoint, Grace
offload, no RTX card. The watchdog's fail-stop policy means a trip leaves this lane down on purpose — bring the
incumbent back by hand after reading `sidecar_watchdog.log`.

## Credits

This recipe is a productionised descendant of someone else's recipe, and the ordering here is deliberate.

* **[RonanLabsAI](https://github.com/RonanLabsAI/dgx-station-frontier) — the sidecar recipe this lane runs.**
  The two-step `sidecar` → `server` deployment shape, the launcher we adapted line by line, the b12x peer
  tier-2 protocol in production form, the `rowmap-mix-v1` calibration and its Stage-0 variants, the
  `MEGA_PEER_CHECK` agreement gate as a bring-up requirement, the 1M long-context work, and the
  `kbench_chat_ds.py` harness every real-text number here was measured on are his (`recipes/dsv41-flash-sidecar`
  plus `results/sidecar-ab.md`, `results/long-context-1m.md`, `logs/dsv41-1017/*`, Apache-2.0). His published
  1017 sidecar cells on that harness — C1 273.6 / C4 689.1 / C16 1,545.5 / C32 2,276.6 — are the bar we
  measured our own lane against; we land within a couple of percent of them (see Results), which is the
  honest read: **the mechanism is his.** The KV round (`results/2026-10-22-kv85/`) was measured on his harness
  and his CNN/DM prompt pool too.
* **[original-el8](https://github.com/original-el8/dgx-station-gb300-research)** — the M3 hook (hot/cold split
  inside DeepGEMM MegaMoE, cold path on the FlashInfer TRT-LLM kernel), `peer_tier2`/`peer_fused2`, and the
  `rowmap-static-v1.json` map this recipe ships unmodified, @87af662, Apache-2.0. Without the hook there is no
  hot/cold split to give the second GPU.
* **[b12x](https://github.com/local-inference-lab/b12x)** (Apache-2.0) — the FP4 MoE kernels the sidecar
  replays out of its 560 captured graphs.
* **vLLM** — [#58132](https://github.com/vllm-project/vllm/pull/58132) bounded decoder replay, the
  `deep_gemm_mega_moe` backend, and the UVA offloader the sibling recipe uses.

**Our delta, precisely** (so nobody re-credits us with the idea): the uniform 295/89 map choice and the
measurement that 285/99 fails its pre-registered cosine bar; running the agreement gate on real prompts and
publishing the check distribution; the ack-before-bookkeeping patch to `peer_server2.py`
([`patches/peer-server2-local.patch`](patches/peer-server2-local.patch)); the fail-closed watchdog and the
fail-stop serving policy; the bracketed control→candidate→control discipline on every speed cell; and the
quality gates (teacher-forced flips, BFCL, per-class decode) applied to a sidecar lane at all. Measurements,
gate design and the write-up: **J&M Recipes** (James Meadlock, Milo). Session note: the numbers were produced
with a Claude Opus 5.5 orchestrator on the research side; the recipe was written from the receipts.

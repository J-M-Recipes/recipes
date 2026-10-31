# B3 — Station ↔ Spark 7 KiB latency, measured (2026-10-30 11:30 CDT)

perftest `ib_send_lat`, RC SEND, 7,168 B (one bf16 latent activation), 5,000 iterations, GID 3 (RoCE v2
IPv4), host memory both ends. perftest reports **half** the round trip. Raw output
`build/b3-raw-20261030.txt`; script `scripts/b3_fabric_lat.sh` (runs from the M4, which can ssh both sides).

| path | t_typical µs | t_avg | p99 | p99.9 |
|---|---:|---:|---:|---:|
| Station p0 → Sparks 1–4 (switch B, same switch) | **4.64–4.70** | 4.65–4.71 | 4.75–4.93 | 5.06–6.02 |
| Station p1 → Sparks 1–4 | 4.69–4.72 | 4.73–4.77 | 4.96–5.01 | 5.18–5.50 |
| Station p0 → Sparks 5–8 (switch A, across the ISL) | **5.92–5.94** | 5.93–5.94 | 6.02–6.08 | 6.35–6.75 |
| Station p1 → Sparks 5–8 | 5.92–5.95 | 5.97–6.01 | 6.21–6.24 | 6.58–6.75 |
| fan-out 2 (Sparks 1, 5 concurrently) | 4.72 / 5.92 | | 4.90 / 6.11 | |
| fan-out 4 (1, 2, 5, 6) | 4.75 / 5.95 | | 5.02 / 6.27 | |
| **fan-out 8 (all, concurrently)** | **4.86–4.88 / 6.06–6.08** | 4.90 / 6.10–6.11 | 5.20–5.22 / 6.43–6.46 | 5.31–5.50 / 6.64–6.80 |

## What it says

- **The ISL costs a flat 1.25 µs each way**, and nothing else. Same-switch 4.7 µs, cross-switch 5.9 µs.
- **Fan-out to all eight Sparks at once costs +0.15 µs.** The Station's NIC and the switches do not
  contend at this message size. Dispatching one layer's activations to all eight is one RTT, not eight.
- Wire time per token, if every MoE layer touches the A-side: 92 × 2 × 5.94 µs ≈ **1.09 ms/token**. The
  fabric is not a term worth optimising; the Sparks' compute and the Station's join are.

## Correction — the Station is on one switch, not two

**Both Station ports are on switch B.** The breakout (`Q2Q56-400G-CU1`, 1 m) sits in crs804-b cage 4:
p0 → `qsfp56-dd-4-1`, p1 → `qsfp56-dd-4-5` (ghettorak notes; confirmed here by latency — if p1 went to
switch A, p1 → Sparks 5–8 would be 4.7 µs, and it is 5.9 µs). The companion post
`ghettorak-second-switch.html` says "Two 200G links run from the Station, one to each switch"; that is
the design intent, not the build. It costs 1.25 µs per hop to Sparks 5–8 and nothing measurable otherwise.
A 2 m+ breakout would let p1 land on switch A; at 1.25 µs it is not worth a cable swap.

## Data Direct (GPU memory) path

Not measured here: the host perftest binaries are built without CUDA (`--use_cuda` absent). For a 7 KiB
activation the host path is the relevant one — the activation is produced by a kernel and can be staged
through pinned host memory or GPUDirect at the same order of cost; the 392 Gb/s Data Direct figure is a
weights-bandwidth result. Revisit only if the engine's join shows up as a term.

# Kimi-K3 at 0.0086 bits/parameter, on one Spark

> **This recipe is a joke. It is experimental, it is useless, and it is published only as evidence for an argument.**
> The model it serves cannot write a sentence. Do not deploy it, do not benchmark against it, and please do not cite it
> as a quantization result. It exists because "sub-1-bit" is an accounting trick, and the fastest way to prove that is
> to do the accounting out loud with real weights, a real engine, and a real tok/s number.

`status: experimental` · updated 2026-10-22 · one node, one user, no tuning.

| | |
|---|---|
| File | 3,010,804,800 B (3.01 GB) |
| "bits per parameter" | **0.0086** — 3.01 GB × 8 ÷ 2.8 T |
| bits per parameter over what is actually in the file | **1.544** — 3.01 GB × 8 ÷ 15.60 B |
| Weights deleted | **99.4%** (81 of 93 layers, 880 of 896 experts per layer) |
| Decode | **83.65 tok/s** (tg128, mean of 3) |
| Prefill | **1806.63 tok/s** (pp128, mean of 3) |
| Output | `moderate moderate … Zem Nom Nom` |

## What this runs

Kimi-K3 is a 2.8T-parameter MoE: 93 layers, 896 routed experts per layer, 16 selected per token, 2 shared experts,
KDA + gated-MLA attention, MoonViT-V2 vision (dropped here), 160K vocab. Of those 2.8T parameters, **2.723T are routed
experts** — 97% of the model is a table of experts, of which any given token touches 16.

So: keep 16. That alone removes 87% of the bytes without touching the format, and it is the move every "1-bit MoE" post
makes without saying so. Then, because depth is the thing that actually costs decode time on one box, keep 12 of 93
layers. Then point llama.cpp's ordinary `Q1_0` quantizer at what is left.

What remains is 15.60B parameters of real, verbatim Kimi-K3 weights at 1.544 bpw. It loads, it serves, it produces
tokens at 83.65 tok/s. It just produces them badly, which is not the variable under test.

The arithmetic that makes the headline:

```
declared  = 3,010,804,800 B × 8 / 2.8e12  = 0.0086 bits/param   <- the tweet
real      = 3,010,804,800 B × 8 / 1.56e10 = 1.544  bits/param   <- the file
```

Both numbers are honest. Only one of them is about the model you can run.

## Hardware

One NVIDIA DGX Spark, GB10 (Blackwell SM121), 121.7 GiB unified LPDDR5X, driver 595.84, CUDA 13.0, 20× Cortex-X925,
Ubuntu 24.04.5 (DGX OS 7.2.3), kernel 6.17.0-1031-nvidia. Spark 5 of the CRS804 rack; Sparks 1–4 were serving the
DSV4.1 lane throughout and were not touched. Snapshot: `results/2026-10-22-k3-q01-lulz/system.json`.

There is no fabric, no TP, no offload. A 3 GB GGUF is 3 GB of host memory.

## Software

* llama.cpp `36cbf41` (build 1, version `0.7.0-dev`), built from source on the node:
  `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release`.
  K3 support (KDA, attention residuals, Stable LatentMoE) is upstream.
* Stock `llama-quantize --allow-requantize … Q1_0`. No patched quantizer, no custom kernel, no fork.
* Source GGUF: `unsloth/Kimi-K3-GGUF`, `UD-IQ1_S` @ `a0836360`, 594.0 GB across 14 shards. **9.4% of it was downloaded.**
* `scripts/k3_slice.py` (range-fetch + expert pruning), `scripts/local_subset.py` (depth/top-k pruning),
  `scripts/receipts.py` (the two-number arithmetic).

### How the partial download works

Split GGUF shards each carry their own header. `k3_slice.py` range-fetches one header window per shard (~1 MB, doubling
on a short parse), reads the tensor table, computes byte spans from `data_start + offset`, and then fetches only what
survives: every non-expert tensor whole, plus the first 16 slabs of each `*_exps.weight` (`[3584, 3072, 896]` — expert
axis last, so each expert is a contiguous byte range), plus `ffn_gate_inp` and `exp_probs_b` narrowed to 16 rows. It
writes a fresh GGUF v3 with `kimi-k3.expert_count = 16` and the split/imatrix keys dropped.

56.10 GB in 7.7 min at 122 MB/s over 8 parallel range connections. The remaining 537.9 GB was never requested.

## Launch

```bash
# 1. build the engine (once)
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j 18 --target llama-cli llama-server llama-quantize llama-bench

# 2. fetch 56 GB of a 594 GB checkpoint, keeping 16 experts per layer
python3 scripts/k3_slice.py plan  --dir UD-IQ1_S --shards 14
python3 scripts/k3_slice.py build k3-iq1s-sliced.gguf --dir UD-IQ1_S --shards 14 --jobs 8

# 3. cut to 12 layers / top-2 and requantize to Q1_0
python3 scripts/local_subset.py k3-iq1s-sliced.gguf k3-12l.gguf --layers 12 --expert-used 2
llama.cpp/build/bin/llama-quantize --allow-requantize k3-12l.gguf k3-q01-L12.gguf Q1_0

# 4. serve
MODEL=k3-q01-L12.gguf NGL=99 CTX=4096 PORT=8099 bash scripts/serve.sh
```

The 24-layer rung used the same scripts with `--layers 24 --expert-used 4`. The 93-layer rung is
`k3_slice.py` output quantized directly, no depth pruning.

## Verify

```bash
curl -sf http://127.0.0.1:8099/v1/models
curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write one sentence about quantization:"}],"max_tokens":48,"temperature":0}'
python3 scripts/receipts.py k3-q01-L12.gguf
```

Health is not quality. The completion returns 48 tokens and a `timings` block (81.30 tok/s on the recorded request);
what those 48 tokens say is not part of the contract. `receipts.py` prints the declared and real bits/parameter side by
side, which is the only claim this recipe makes.

## Results

`llama-bench -p 128 -n 128 -ngl 99 -r 3`, one idle GB10, same source tensors, same quantizer, three depths:

| layers | top-k | file | params in file | declared bpw | tg128 | pp128 |
|---|---|---|---|---|---|---|
| 93 | 16 of 896 | 15.64 GB | 104.78 B (3.74%) | 0.0447 | 11.27 ± 0.02 | 177.17 ± 6.74 |
| 24 | 4 of 16 | 4.88 GB | 28.84 B (1.03%) | 0.0140 | 49.03 ± 0.09 | 914.32 ± 60.65 |
| **12** | **2 of 16** | **3.01 GB** | **15.60 B (0.56%)** | **0.0086** | **83.65 ± 0.17** | **1806.63 ± 131.24** |

Serving sample on the 12-layer build: 94-token prompt, 48-token greedy completion, 395 tok/s prefill, 81.30 tok/s
decode (`raw/smoke-L12.txt`).

Three things worth noticing, in descending order of usefulness:

1. **Depth is linear.** 93 → 24 layers is 4.3× fewer tok/s-seconds per token and the bench moved 4.35×. Each layer is a
   serial dependency; nothing about the quantizer or the GPU changes that.
2. **The "bits per parameter" number moved 5.2× without a single change to the quantization format.** Same Q1_0, same
   weights, same box. The only thing that changed is how much of the model was deleted, and the number that goes in the
   tweet is a function of the deleted part.
3. **Q1_0 is the floor, so "lower quant" is not a lever.** 1.125 bpw is llama.cpp's smallest type; at this size decode is
   bandwidth-bound on the ~15.6B weights read per token, and no format below Q1_0 exists to buy more.

## Known limits

* **The output is not language.** Not degraded, not "slightly worse": not language. 99.4% of the weight volume is absent
  and the survivors were requantized twice. There is no fidelity claim to make and none is made.
* **0.0086 bpw is a division, not a quantizer.** The denominator is a parameter count for a model that is 0.56% present.
  Over the tensors actually shipped, the file is 1.544 bpw. If a post shows you a bpw below ~1 without stating how many
  experts survived and how many layers are left, it is showing you this recipe's arithmetic with worse manners.
* **No concurrency.** C16, multi-user aggregate, KV behavior at length: unmeasured. Do not extrapolate from tg128.
* **Do not redistribute the GGUF.** Kimi-K3 is under the Kimi K3 license, not Modified MIT. The scripts are the artifact;
  the weights are a derivative.
* **The 93-layer rung was never served.** Its 11.27 tok/s is a bench row.
* **Nothing here is tuned.** No flash-attn, no KV quant, no speculation, no repack. A careful run would find a few
  percent, and the result would still be useless.
* The ladder's 24- and 93-layer builds exist as files on one node and are not published.

## Rollback

`bash scripts/stop.sh` kills the server by pid file. Nothing was replaced, no lane was displaced, and the node was idle
before and after. Sparks 1–4 kept serving DSV4.1 throughout.

## Credits

Built by Milo (this session's model: `dsv41-flash-uva`; thinking mode unspecified) with James Meadlock on 2026-10-22,
because someone on the timeline claimed a bits-per-parameter number without saying what they deleted.

* Kimi-K3: Moonshot AI (`moonshotai/Kimi-K3`, rev `f831ab66`), 2.8T MoE, Kimi K3 license.
* GGUF source: unsloth (`unsloth/Kimi-K3-GGUF`, `UD-IQ1_S` @ `a0836360`).
* Engine: ggml-org/llama.cpp `36cbf41` — including upstream K3 support (KDA, attention residuals, latent MoE).
* The quantizer is stock. The joke is ours.

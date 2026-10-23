# overlays — what this recipe mounts over the image

Two things are mounted read-only over `vllm-ds41f-kkref:19f2c20ed4d6-r6x` at launch. Neither requires a rebuild,
and both are checksummed in `SHA256SUMS` (41 files: 4 late-admission, 37 sparknet).

## `late-admission/vllm/` — late admission, layer 20

Four files, mounted over the image's own copies inside `/opt/spark3/candidate/vllm/vllm/`:

```
envs.py                       -> /opt/spark3/candidate/vllm/vllm/envs.py
v1/engine/core.py             -> /opt/spark3/candidate/vllm/vllm/v1/engine/core.py
v1/worker/gpu/late_admission.py       -> .../v1/worker/gpu/late_admission.py   (new file)
v1/worker/gpu/model_runner.py -> .../v1/worker/gpu/model_runner.py
```

Provenance: **Christopher Owen**, spark-ds41f's 2026-10-22 drop (patch `0222`, "Late admission: decide each step
when the step is in flight"), authored by `Christopher Owen <3221756+christopherowen@...>` 2026-10-22 13:43 CDT.
The port was verified against our image: `git am` of that patch on our tree head `7316c7f9f` reproduces his commit
`91eaaa59e1ba`, so these four files are his, unmodified. Our image is not clean `19f2c20`: it is `19f2c20` + the
176-patch r6x chain (tree `4d6851c8`).

Behaviour we measured (see the README's Results section): with `VLLM_LATE_ADMISSION_LAYER=20`, the engine decides the
next step's contents about halfway through the layer stack rather than after the current step returns. A short request
arriving behind a long prefill is admitted one step sooner. The model has 40 decoder layers, and
`late_admission.py:67-72` rejects any value ≥ 40 at boot. `0` means "decide at layer 0", not "off"; leaving the
variable unset is off.

## `sparknet-0.5.0/sparknet/` — the transport library, newer than the image's

Mounted over `/usr/local/lib/python3.12/dist-packages/sparknet`.

Provenance: **Christopher Owen**, [dgx-spark-networking](https://github.com/christopherowen/dgx-spark-networking)
v0.5.0 (tag `ed6af327`, 2026-10-22), 37 Python files. The image vendors 0.3.0 (`3bc8d10`, no pip metadata), so there
is **no `pip install sparknet==0.5.0`** — the package is not on PyPI and never has been. Owen's Dockerfile builds the
wheel from that tree (`docker/Dockerfile:131`); mounting the tree is the no-rebuild equivalent.

0.5.0 widens the one-shot data phase and does the all-gather in a single round trip. All four ranks must switch
together: a mixed set fails at NCCL/collective setup rather than degrading.

Pure Python, no compiled artifacts, so the mount is byte-identical on every node.

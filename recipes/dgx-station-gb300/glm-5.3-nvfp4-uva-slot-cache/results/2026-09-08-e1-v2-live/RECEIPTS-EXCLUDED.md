# E1 v2 receipt exclusions

Publication root: `results/2026-09-08-e1-v2-live/`.

Source receipt root on this machine: `/Users/jamesmeadlock/hermes/jm-recipes/e1v2-receipts/`.

Station receipt root: `milo@192.168.1.9:/home/milo/e1-window-20260908-v2/receipts/`.

The live receipts copied into git intentionally exclude large raw profiler artifacts, verbose logs, and executed source. Executed source is represented by `live-receipts/source-manifest.json`; the source was captured from git commit `c6659e24` plus the manifest-listed files/hashes.

## Local files/directories excluded from this publication

| path under `e1v2-receipts/` | type | size bytes | sha256 |
|---|---:|---:|---|
| `e1-profile.nsys-rep` | file | 21,752,229 | `40a42dae0ced84642d8f3096e26af1b5d487a2a5dcf3ea830f18675f1fa8dfde` |
| `e1_cuda_api_trace.csv` | file | 6,654,192 | `e737f4ed0512d1302012faff5530e39708df78ce419f6e0a52fba55c8d688e77` |
| `container.log` | file | 2,155,210 | `6d057ce40db9de02ca77e961f4bbbb791582736bc0440cd5438576a1caf0dfa9` |
| `raw-logs/docker.log` | file | 2,156,543 | `399f853d11507d7207c0c63448819d5f6597da3c218fe351e76ad7b3c3e26672` |
| `executed-source/` | directory | 138,339 across 17 files | manifest digest `e7474b9fb4cf2c96a08cbb2ce474c5baf8ba987949424437509daa5f95e71d6b` |

`nsys-cli` was named in the exclusion rule, but no local `e1v2-receipts/nsys-cli` path existed when this package was created.

## Station-only raw artifacts not copied locally

These artifacts remain referenced at the Station path above and are not present in the local publication tree:

| Station path | size note | sha256 |
|---|---:|---|
| `e1_cuda_gpu_trace.csv` | about 210 MB | `0ba12baaf22db45d3efa39fe4cf6f791faa6029e7c2f0e0936fc38266017bb6a` |
| `e1-profile.sqlite` | about 67 MB | `3b7760d5a04cd68cc0c9dd752a2372a0ef19c33cd727e1582933f100dd92ded0` |

The Station-only hashes are copied from `analysis/findings.md`; they were not recomputed in this local repository because the files were not present locally and the task explicitly avoided Station access.

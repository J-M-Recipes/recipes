#!/usr/bin/env python3
"""Make a truncated *view* of the K3 NVFP4 checkpoint: symlinks + pruned index + config with N layers.

No weight bytes are copied. vLLM only opens shards referenced by the index's weight_map, so pruning
the map to layers < N (plus all non-layer tensors) cuts load I/O from ~1.6 TB to what N layers need.
Tensors for layers >= N that share a shard with needed ones are dropped by the companion
sitecustomize hook (K3_MAX_LAYER).

Usage: k3_truncated_view.py N /models/kimi-k3/g1-view-nN
"""
import json, os, re, sys

SRC = "/models/kimi-k3/nvidia-nvfp4-b2428a0b"
N = int(sys.argv[1]); DST = sys.argv[2]
LAYER = re.compile(r"\.layers\.(\d+)\.")

os.makedirs(DST, exist_ok=True)
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
keep = {}
for name, shard in idx["weight_map"].items():
    m = LAYER.search(name)
    if m and int(m.group(1)) >= N:
        continue
    if name.startswith(("vision_tower", "mm_projector", "multi_modal_projector")):
        continue
    keep[name] = shard
shards = sorted(set(keep.values()))

for f in os.listdir(SRC):
    p = os.path.join(DST, f)
    if f.endswith(".safetensors"):
        if f in shards and not os.path.lexists(p):
            os.symlink(os.path.join(SRC, f), p)
        continue
    if f in ("model.safetensors.index.json", "config.json"):
        continue
    if not os.path.lexists(p):
        os.symlink(os.path.join(SRC, f), p)

json.dump({"metadata": idx.get("metadata", {}), "weight_map": keep},
          open(f"{DST}/model.safetensors.index.json", "w"), indent=1)
cfg = json.load(open(f"{SRC}/config.json"))
(cfg.get("text_config") or cfg)["num_hidden_layers"] = N
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)

sz = sum(os.path.getsize(os.path.join(SRC, s)) for s in shards)
print(f"N={N}: {len(keep):,} tensors kept, {len(shards)} of 96 shards linked, {sz/2**30:.1f} GiB of shard files to open")

#!/usr/bin/env bash
# Desk F DRAFT (not run by the desk). Station, as milo. Builds ONE W1 variant CPU-only in the pinned image and
# prepares a private fastpath copy for shadow_repro. Never touches the served tree (serve-bisect is only read).
#   usage: build_variant.sh <name> [-DW1_... ...]        e.g. build_variant.sh sink-unfused -DW1_VARIANT_SINK_UNFUSED
#   result: $T/F-builds/<name>/build/w1_attn_fp8_ext.so and $T/F-builds/<name>/fastpath (copy, .so replaced)
set -euo pipefail
R=$HOME/dsv41/allround-20261112; T=$R/milo-takeover; F=$T/results/F
SRC=${SRC:-$R/W13-w1-hardening/source/deepseek-v4.1-flash/kernels/w1-attn}
FP_SRC=${FP_SRC:-$R/serve-bisect/tree/fastpath}
IMAGE=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
name=${1:?name}; shift
flags="$*"
for f in $flags; do case $f in -DW1_*) ;; *) echo "refusing flag $f" >&2; exit 2 ;; esac; done
B=$T/F-builds/$name
[ -e "$B" ] && { echo "exists: $B (pick a new name)" >&2; exit 2; }
# W13 base must be the tree the diff was made against
for pair in 346fc8ba3cb9ef04f5c89a91895bb109b58cfd1d341c6d54e9f04662db2c83ad:ptx.cuh \
            a23889be55c658d11bf439ce52cd4e4750d7a1d12827ca32926b23a75f02f7d9:stage_attn_seq.cuh \
            4ff15c0124b89c414482125ef5654800d2e15ba4471f6a3cdedacb831e142f25:stage_attn_par.cuh; do
  got=$(sha256sum "$SRC/include/w1/${pair#*:}" | cut -d' ' -f1)
  [ "$got" = "${pair%%:*}" ] || { echo "BASE MISMATCH ${pair#*:}: $got" >&2; exit 2; }
done
mkdir -p "$B/source/kernels"
cp -a "$SRC" "$B/source/kernels/w1-attn"
( cd "$B/source/kernels/w1-attn" && patch -p1 -s --no-backup-if-mismatch < "$F/variants.DRAFT.diff" )
cp "$F/build_candidate_F.py" "$B/"
printf '%s\n' "$flags" > "$B/FLAGS"
echo "BUILD $name flags=[$flags] start $(date '+%F %T')"
docker run --rm --name "dsv41-F-build-$name" --network none --runtime runc --cpus 4 --memory 12g --memory-swap 12g \
  --pids-limit 256 --ulimit core=0 -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -e MAX_JOBS=2 \
  -e CUDA_HOME=/usr/local/cuda -e "W1_EXTRA_CFLAGS=$flags" -v "$B:/work" -w /work --entrypoint python3 "$IMAGE" \
  /work/build_candidate_F.py > "$B/compile.log" 2>&1 || { echo "BUILD FAILED, tail:"; tail -40 "$B/compile.log"; exit 1; }
grep -o 'W1 desk-F[^"]*' "$B/compile.log" | sort -u || true          # which switches the compiler saw
cp -a "$FP_SRC" "$B/fastpath"
cp "$B/build/w1_attn_fp8_ext.so" "$B/fastpath/w1attn/w1_attn_fp8_ext.so"
sha256sum "$B/build/w1_attn_fp8_ext.so" | tee "$B/so.sha256"
echo "BUILD $name done $(date '+%F %T')"

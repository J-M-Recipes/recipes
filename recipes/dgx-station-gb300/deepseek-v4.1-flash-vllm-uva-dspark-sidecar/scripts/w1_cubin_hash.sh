#!/usr/bin/env bash
# w1_cubin_hash.sh — device-code identity for the rebuilt W1 decode-attention extensions (Pg62).
#
#   bash scripts/w1_cubin_hash.sh <ext.so> [<ext.so> ...]
#   e.g. bash scripts/w1_cubin_hash.sh $WORK/tree/fastpath/w1attn/w1_attn_ext.so $WORK/tree/fastpath/w1attn/w1_attn_fp8_ext.so
#
# A rebuilt .so is never byte-identical to the one that served (nvcc embeds build paths and -lineinfo line tables),
# so `sha256sum` of the .so proves nothing. This extracts every embedded ELF cubin with `cuobjdump -xelf all` inside
# the pinned image (CPU only, no GPU, no network) and prints one sha256 per cubin. Two builds are device-code
# identical when the cubin lists match line for line (same names, same hashes). Run it on the launcher-built files
# and on the files that served (the box's pg62-work tree) and diff the two outputs.
#
# Refuses to report an empty result: a missing cuobjdump or a .so with no cubin exits non-zero, because two empty
# lists compare equal and prove nothing.
# Env: IMAGE (default vllm/vllm-openai:v0.31.0).
set -euo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai:v0.31.0}
[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 2; }
rc=0
for so in "$@"; do
  [ -f "$so" ] || { echo "MISSING $so" >&2; rc=1; continue; }
  d=$(mktemp -d)
  cp "$so" "$d/"
  b=$(basename "$so")
  out=$(docker run --rm --network none --user "$(id -u):$(id -g)" -v "$d:/x" -w /x --entrypoint bash "$IMAGE" -c '
    set -e
    C=$(command -v cuobjdump || true)
    for c in /usr/local/cuda/bin/cuobjdump /usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/cuobjdump; do
      [ -n "$C" ] && break; [ -x "$c" ] && C=$c
    done
    [ -n "$C" ] || { echo "NO_CUOBJDUMP"; exit 3; }
    "$C" -xelf all "$1" >/dev/null
    n=0
    for f in *.cubin; do
      [ -s "$f" ] || continue
      n=$((n+1)); sha256sum "$f"
    done
    [ "$n" -gt 0 ] || { echo "NO_CUBIN"; exit 4; }
  ' _ "$b") || { echo "FAIL $so: ${out:-docker run failed}" >&2; rc=1; rm -rf "$d"; continue; }
  echo "# $so (file sha256 $(sha256sum < "$so" | cut -c1-16)…, $(echo "$out" | wc -l | tr -d ' ') cubin)"
  echo "$out" | sort -k2
  rm -rf "$d"
done
exit $rc

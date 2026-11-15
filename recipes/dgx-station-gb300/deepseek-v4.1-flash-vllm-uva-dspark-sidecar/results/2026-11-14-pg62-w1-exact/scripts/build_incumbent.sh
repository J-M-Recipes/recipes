#!/usr/bin/env bash
# Desk I DRAFT (not run by the desk). Station, as milo. Builds ONE incumbent w1_attn_ext.so variant, CPU-only, in the
# pinned image, from the Pg61 production source (read-only; copied), with the incumbent's own ops.build().
#   usage: build_incumbent.sh <name> [--no-patch] [-DW1_... ...]
#     build_incumbent.sh control                      patched tree, no -D   (expected = shipped arithmetic)
#     build_incumbent.sh rebuild --no-patch           unpatched tree, no -D (toolchain/image reproducibility only)
#     build_incumbent.sh stock-all -DW1_VARIANT_STOCK_ALL
#   result: $T/I-builds/<name>/build/w1_attn_ext.so (+ compile.log, compile-result.json, so.sha256, FLAGS, w/ = build root)
# Never touches the Pg61 tree or any served tree. Nothing is installed.
set -euo pipefail
R=$HOME/dsv41/allround-20261112; T=$R/milo-takeover; I=$T/results/I
PG=${PG:-$HOME/dsv41/garner/pg61-work/tree}          # Pg61 production tree (read-only)
SRC=$PG/kernels/w1-attn; FPSRC=$PG/fastpath/w1attn
IMAGE=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
SHIPPED_SHA=7201643d5575c18706639a0b40b7e1f7482ef2535ebc4c54c55920116743e35f
name=${1:?name}; shift
patch_it=1; flags=""
for a in "$@"; do
  case $a in
    --no-patch) patch_it=0 ;;
    -DW1_*) flags="${flags:+$flags }$a" ;;
    *) echo "refusing argument $a" >&2; exit 2 ;;
  esac
done
[ "$patch_it" = 1 ] || [ -z "$flags" ] || { echo "--no-patch takes no -D (the switches live in the patch)" >&2; exit 2; }
B=$T/I-builds/$name
[ -e "$B" ] && { echo "exists: $B (pick a new name)" >&2; exit 2; }
# 1. the Station source must be the desk copy the diff was made against (every file, plus ops.py / integrate.py)
( cd "$PG" && sha256sum -c --quiet "$I/pg61-src.sha256" ) || { echo "BASE MISMATCH: $PG differs from results/I/pg61-src" >&2; exit 2; }
got=$(sha256sum < "$FPSRC/w1_attn_ext.so" 2>/dev/null | cut -c1-64 || true)
[ "$got" = "$SHIPPED_SHA" ] && echo "shipped incumbent in Pg61 tree: ${got:0:16} (expected)" \
  || echo "WARNING: $FPSRC/w1_attn_ext.so sha is '${got:0:16}', not ${SHIPPED_SHA:0:16} (the build does not use it)"
# 2. build root with build.sh's temp-dir layout: /w/{w1attn,csrc,include,build}
mkdir -p "$B/w/build"
cp -a "$FPSRC" "$SRC/csrc" "$SRC/include" "$B/w/"
chmod -R u+w "$B/w"
if [ "$patch_it" = 1 ]; then
  ( cd "$B/w" && patch -p1 --dry-run -s < "$I/incumbent_variants.DRAFT.diff" && patch -p1 -s --no-backup-if-mismatch < "$I/incumbent_variants.DRAFT.diff" )
  printf 'patched\n' > "$B/PATCH"
else
  printf 'unpatched\n' > "$B/PATCH"
fi
printf '%s\n' "$flags" > "$B/FLAGS"
echo "BUILD $name patch=$patch_it flags=[$flags] start $(date '+%F %T')"
# 3. same docker flags as desk F's build_variant.sh, plus the incumbent build's --user and TORCH_CUDA_ARCH_LIST;
#    the driver runs the incumbent's own ops.build() (see build_incumbent_driver.py)
docker run --rm --name "dsv41-I-build-$name" --network none --user "$(id -u):$(id -g)" --runtime runc --cpus 4 \
  --memory 12g --memory-swap 12g --pids-limit 256 --ulimit core=0 -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e MAX_JOBS=2 -e TORCH_CUDA_ARCH_LIST=10.3a -e "W1_EXTRA_CFLAGS=$flags" \
  -v "$B/w:/w" -v "$I:/i:ro" -w /w --entrypoint python3 "$IMAGE" /i/build_incumbent_driver.py \
  > "$B/compile.log" 2>&1 || { echo "BUILD FAILED, tail:"; tail -40 "$B/compile.log"; exit 1; }
mkdir -p "$B/build"
cp "$B/w/build/w1_attn/w1_attn_ext.so" "$B/build/w1_attn_ext.so"
cp "$B/w/compile-result.json" "$B/compile-result.json"
echo "switches the compiler saw:"; grep -o 'W1 desk-I variant active: [A-Z_0-9]*' "$B/compile.log" | sort -u || echo "  (none)"
echo "ptxas for attn_ss_kernel (regs / spills):"
grep -A4 "Compiling entry function '.*attn_ss_kernel" "$B/compile.log" | grep -E 'Used|spill' | sort -u | head -6 || true
grep -E 'bytes spill (stores|loads)' "$B/compile.log" | grep -v ' 0 bytes spill stores, 0 bytes spill loads' | head -5 || true
sha256sum "$B/build/w1_attn_ext.so" | tee "$B/so.sha256"
echo "BUILD $name done $(date '+%F %T')"

#!/usr/bin/env bash
# start-h2ref.sh — the h2-REF serving release (promoted 2026-10-28 17:22:58 CDT on the reference box).
#
# h2-REF = v27 (sidecar + 48 GiB host prefix-KV tier + block verification, DSpark k=5 [[1,4,5],[5,8,3],[9,32,3]],
#          KV declared 8.5 GiB = 4,400,013 tokens)
#        + h1 (hook-h1 L2 attention/indexer key-sharing + the vllm#61161 block-verification backport)
#        + the vllm#60945 rejection-sampler guard in that #61161 overlay file (sha256 2d43d513…)
#        + split-KV decode attention at 4 splits, quarantined to 1 request and s_q == 6, from the SIDE extension
#          _flashmla_splitkv_C (sha256 744a9de8…) mounted next to the image's untouched _flashmla_C (df2cdd1c…)
#        + no core dumps of any kind.
# Env-for-env the same as the measured Card AP2 arm S4 and the promoted container (results/2026-10-28-h2-splitkv-ap2/).
#
# Prerequisites (README "Software" builds each one from patches/): $M3_DIR/hook-h1, $M3_DIR/overlay-61161/tree/vllm,
# $M3_DIR/splitkv/{build/out,overlay/vllm}, plus everything scripts/launch-sidecar.sh already needs.
# Usage:  GB300_UUID=... RTX_UUID=... bash scripts/start-h2ref.sh     (sidecar first, then server)
# After it binds: arm scripts/sidecar_watchdog.py --mode enforce against $NAME (README "Launch"), PPID 1.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=${M3_DIR:-$HOME/dsv41/m3}
export HOOK_DIR=${HOOK_DIR:-$W/hook-h1}
export OVERLAY_61161=1 O61=${O61:-$W/overlay-61161/tree/vllm}
export MEGA_SHARE_TOPK=2 MEGA_POS_I32_ONCE=1 MEGA_CAND_COOP=0 MEGA_L2_VERIFY=1 MEGA_L2_STATS_EVERY=2000
export NAME=${NAME:-dsv41-vllm-1017-h2-REF} ROWMAP=rowmap-mixed-ar-v1.json
export EXTRA="--kv-cache-memory 9126805504 --kv-offloading-size 48 --kv-offloading-backend native"
export SPEC_EXTRA=',"rejection_sample_method":"block"'
export NOCORE=1 SPLITKV=${SPLITKV:-4} SPLITKV_DIR=${SPLITKV_DIR:-$W/splitkv}
export SPLITKV_MAX_TOKENS=8 SPLITKV_MAX_REQS=1 SPLITKV_SQ=6 SPLITKV_TRACE=1 SPLITKV_GUARD=0
[ "$SPLITKV" = 4 ] || echo "WARNING: SPLITKV=$SPLITKV — only 4 splits passed Card AP2 (8 splits = NO: 1.022x and 6 confident TF flips)" >&2
[ -f "$HOOK_DIR/rowmap-mixed-ar-v1.json" ] || { echo "rowmap-mixed-ar-v1.json missing in $HOOK_DIR (copy it from results/2026-10-23-ar5-mixed-map/)"; exit 3; }
for step in ${STEPS:-sidecar server}; do bash "$HERE/launch-sidecar.sh" "$step" || exit 1; done
echo "started $NAME (+-rtx) = h2-REF; wait for /v1/models, check the boot facts (README Verify), then arm the fail-stop watchdog"

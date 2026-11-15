#!/usr/bin/env bash
# launch-pg61.sh — DeepSeek-V4.1-Flash on DGX Station GB300 + RTX PRO 6000, built on Garner McCloud's recipe.
#
#   Upstream (the engine of this lane): https://github.com/garnermccloud/dgx-station-recipes  (deepseek-v4.1-flash/)
#   pinned at 8043a6811a5feda70fdfb6e827eebb2ec25c65d1, image vllm/vllm-openai:v0.31.0, launched by his serve/serve.sh.
#   Everything that makes this lane fast is his: the 29-component DS41_FASTPATH, peer tier v3, the MegaMoE fork,
#   the rowmap, the overlays. This script only applies a small policy + correctness patch set on top:
#     01/02  six policy hooks in serve.sh / recipe.env, each a no-op when its variable is unset
#            (no API key, attention-config, generation override, reasoning-effort default, component list, PEER3_SPINS)
#     03-05  vLLM #61161 spec_step_seed in his fused draft samplers (w5_markov, w5_markov_gemv, w6mk.cu -> libw6mk.so)
#     overlay-vllm  #61161 gumbel.py + rejection_sampler_utils.py with the #60945 guard, layered on his 58132 overlay
#     sidecar_watchdog_p3.py  fail-closed watchdog (a peer-tier-v3 timeout skips cold-expert adds silently otherwise)
#   Policy: stock weights, fp8_ds_mla KV, 1M context, 48 GB host KV, no sampling override, 24 of his 29 components
#   (drops ds41_sampling, ds41_stop, w8oproj, lowb_aux, tail_drafthead_mxfp8 — the last three change numerics).
#
# Usage:
#   GARNER_SRC=<checkout> WORK=<dir> launch-pg61.sh build      # tree + overlay + rebuilt libw6mk.so (CPU only)
#   WORK=<dir> MODEL_DIR=<weights> DS41_HOME=<state> launch-pg61.sh up     # serve.sh start + hashes + watchdog
#   WORK=<dir> DS41_HOME=<state> launch-pg61.sh down | verify | render
# Env (defaults): PORT=30006 SERVED=dsv41-flash-uva ALIASES=deepseek-v4.1-flash IMAGE=vllm/vllm-openai:v0.31.0
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PATCHES=${PATCHES:-$HERE/../patches/garner-8043a681}  # recipe layout (box layout: $HERE/patches)
WATCHDOG=${WATCHDOG:-$HERE/sidecar_watchdog_p3.py}       # recipe layout (box layout: $PATCHES/sidecar_watchdog_p3.py)
PIN=8043a6811a5feda70fdfb6e827eebb2ec25c65d1
IMAGE=${IMAGE:-vllm/vllm-openai:v0.31.0}
PORT=${PORT:-30006}; SERVED=${SERVED:-dsv41-flash-uva}; ALIASES=${ALIASES:-deepseek-v4.1-flash}
WORK=${WORK:?set WORK}
TREE=$WORK/tree; OVL=$WORK/overlay-vllm
COMPONENTS=tail_markov_fused,tail_swa_meta,tail_dflash_prep,tail_ctxkv,tail_replay_skip,tail_embed,attn,preattn,preattn_cand,preattn_index,preattn_compress,tail_draft_embed,boundary,suffix,topp_fast,tail_proj_side,engram_side,fastload,coldskip,peer3,megamoe,quant_producer,a12_retire,a12_memory_guard
WANT_GUMBEL=6eac3e1348339d157ba0e2184114859aca207df1
WANT_RSU=edc8f8f4c425f51a962840fc2e0e7aa32059f57a
VP=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu
# Pg62 (2026-11-14, default OFF; see pg62_build below). The image must be this digest for the two W1 builds.
W1X=$PATCHES/w1-exact
IMAGE_ID=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
log(){ printf '%s [pg61] %s\n' "$(date '+%F %T %Z')" "$*"; }
die(){ log "FAIL: $*"; exit 1; }

policy_env(){ cat <<EOF
DS41_PORT=$PORT
DS41_SERVED=$SERVED
DS41_ALIASES=$ALIASES
DS41_GPU_MODE=cdi
DS41_NO_KEY=1
DS41_MAXLEN=1048576
DS41_KV=fp8_ds_mla
DS41_OFFLOAD_GB=48
DS41_EFFORT_DEFAULT=0
DS41_FASTPATH_COMPONENTS_OVERRIDE=$COMPONENTS
DS41_OVERLAY_58132_DIR=$OVL
DS41_PEER3_SPINS=1000000000000
EOF
  if [ "${DS41_PG62:-0}" = 1 ]; then echo "DS41_W1_COMPRESSED_FP8=1"; fi
}

# Pg62 build block (DS41_PG62=1 only). Runs after 01-05 and the libw6mk.so rebuild, before the manifest.
#   1. incumbent w1_attn_ext.so: his pinned fastpath/w1attn + kernels/w1-attn/{csrc,include} copied to a /w build root,
#      + w1-exact/incumbent-variants.patch, his own ops.build() via w1-exact/build_incumbent_driver.py with
#      -DW1_VARIANT_STOCK_ALL (the docker flags of results/2026-11-14-pg62-w1-exact/scripts/build_incumbent.sh)
#   2. patches 06 (serve.sh passes DS41_W1_COMPRESSED_FP8) and w1-exact/w13-fp8-compressed-records.patch on the tree
#   3. FP8 w1_attn_fp8_ext.so: the W13-patched kernels/w1-attn + w1-exact/fp8-variants.patch, built by
#      w1-exact/build_candidate_F.py with -DW1_VARIANT_STOCK_ALL (the docker flags of scripts/build_variant.sh there)
#   4. both .so installed into tree/fastpath/w1attn/
# A rebuilt .so is never byte-identical to the one that served (nvcc embeds paths); compare device code with
# scripts/w1_cubin_hash.sh. The FP8 build container runs as root (as it did on the box), so pg62-build/fp8 is
# root-owned; a FORCE=1 rebuild removes it through the image.
pg62_build(){
  local b=$WORK/pg62-build f h got sw
  [ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = "$IMAGE_ID" ] || die "Pg62: $IMAGE is not $IMAGE_ID"
  if [ -e "$b" ]; then
    [ "${FORCE:-0}" = 1 ] || die "$b exists (FORCE=1 to rebuild)"
    docker run --rm -v "$WORK:/x" --entrypoint /bin/rm "$IMAGE" -rf /x/pg62-build
  fi
  # base = his pinned W1 source (01-05 do not touch it); same files build_incumbent.sh checks
  for h in 4eb957de25fd53c5597e658039c4f79ea97432364b4ea6abdeb9d979c311a90d:kernels/w1-attn/csrc/w1_attn.cu \
           346fc8ba3cb9ef04f5c89a91895bb109b58cfd1d341c6d54e9f04662db2c83ad:kernels/w1-attn/include/w1/ptx.cuh \
           ce673e2b5caeee4c11e105d87d93663ff26b54dda1fcf737c7a8e232fdf3452f:kernels/w1-attn/include/w1/stage_attn_ss.cuh \
           bc463cf4044ce009dd7f114f700a86c5077a4b97e7b30a2ae44812d65699ea3a:kernels/w1-attn/include/w1/stage_attn_seq.cuh \
           ab76952275cf9266487e6e32c13a907062f1b3e39f42dc83faa205698228b177:kernels/w1-attn/include/w1/stage_attn_par.cuh \
           2d3b043f19f88bda33b6c505cebb85d1fc61fbe5996e0e164246e5ef0a503ad0:kernels/w1-attn/include/w1/stage_attn_cx.cuh \
           94ece5a1e5131ca337916504ba5643e1be75583984d100c5bdbe0eb71516c753:kernels/w1-attn/include/w1/stage_attn_combine.cuh \
           5b32712f3c12bc1b0a6ae117d435bc6ec65d96d1d231005e8fcac532c87e23b9:fastpath/w1attn/ops.py \
           c3018d0b5b080850b7bdf6d733d7dba80f9d76fdf91ee09ab01c8b723421503e:fastpath/w1attn/integrate.py; do
    got=$(sha256sum < "$TREE/${h#*:}" | cut -c1-64); [ "$got" = "${h%%:*}" ] || die "Pg62 base mismatch ${h#*:}: $got"
  done
  # 1. incumbent
  mkdir -p "$b/inc/w/build"
  cp -a "$TREE/fastpath/w1attn" "$TREE/kernels/w1-attn/csrc" "$TREE/kernels/w1-attn/include" "$b/inc/w/"
  chmod -R u+w "$b/inc/w"
  ( cd "$b/inc/w" && patch -p1 --dry-run -s < "$W1X/incumbent-variants.patch" \
      && patch -p1 -s --no-backup-if-mismatch < "$W1X/incumbent-variants.patch" ) || die "patch incumbent-variants"
  log "building w1_attn_ext.so (incumbent + STOCK_ALL, CPU only, ~1 min)"
  docker run --rm --name dsv41-pg62-build-inc --network none --user "$(id -u):$(id -g)" --runtime runc --cpus 4 \
    --memory 12g --memory-swap 12g --pids-limit 256 --ulimit core=0 -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
    -e MAX_JOBS=2 -e TORCH_CUDA_ARCH_LIST=10.3a -e "W1_EXTRA_CFLAGS=-DW1_VARIANT_STOCK_ALL" \
    -v "$b/inc/w:/w" -v "$W1X:/i:ro" -w /w --entrypoint python3 "$IMAGE" /i/build_incumbent_driver.py \
    > "$b/inc-compile.log" 2>&1 || die "incumbent build failed (see $b/inc-compile.log)"
  sw=$( { grep -ao 'W1 desk-I variant active: [A-Z_0-9]*' "$b/inc-compile.log" || true; } | sort -u | wc -l)
  [ "$sw" -eq 4 ] || die "incumbent build saw $sw of the 4 STOCK_ALL switches"
  # 2. serve.sh flag passthrough + W13 FP8 routing on the tree
  patch -d "$TREE" -p1 --forward --no-backup-if-mismatch < "$PATCHES/06-w1-fp8-flag-serve.patch" >/dev/null || die "patch 06"
  log "applied 06-w1-fp8-flag-serve.patch"
  patch -d "$TREE" -p1 --forward --no-backup-if-mismatch < "$W1X/w13-fp8-compressed-records.patch" >/dev/null || die "patch w13"
  log "applied w1-exact/w13-fp8-compressed-records.patch"
  # 3. FP8 extension; same three W13 base hashes build_variant.sh checks
  for h in 346fc8ba3cb9ef04f5c89a91895bb109b58cfd1d341c6d54e9f04662db2c83ad:ptx.cuh \
           a23889be55c658d11bf439ce52cd4e4750d7a1d12827ca32926b23a75f02f7d9:stage_attn_seq.cuh \
           4ff15c0124b89c414482125ef5654800d2e15ba4471f6a3cdedacb831e142f25:stage_attn_par.cuh; do
    got=$(sha256sum < "$TREE/kernels/w1-attn/include/w1/${h#*:}" | cut -c1-64); [ "$got" = "${h%%:*}" ] || die "W13 base mismatch ${h#*:}"
  done
  mkdir -p "$b/fp8/source/kernels"
  cp -a "$TREE/kernels/w1-attn" "$b/fp8/source/kernels/w1-attn"
  chmod -R u+w "$b/fp8/source"
  ( cd "$b/fp8/source/kernels/w1-attn" && patch -p1 -s --no-backup-if-mismatch < "$W1X/fp8-variants.patch" ) || die "patch fp8-variants"
  cp "$W1X/build_candidate_F.py" "$b/fp8/"
  log "building w1_attn_fp8_ext.so (W13 + STOCK_ALL, CPU only, ~1 min)"
  docker run --rm --name dsv41-pg62-build-fp8 --network none --runtime runc --cpus 4 --memory 12g --memory-swap 12g \
    --pids-limit 256 --ulimit core=0 -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -e MAX_JOBS=2 \
    -e CUDA_HOME=/usr/local/cuda -e "W1_EXTRA_CFLAGS=-DW1_VARIANT_STOCK_ALL" -v "$b/fp8:/work" -w /work --entrypoint python3 \
    "$IMAGE" /work/build_candidate_F.py > "$b/fp8-compile.log" 2>&1 || die "fp8 build failed (see $b/fp8-compile.log)"
  sw=$( { grep -ao 'W1 desk-F variant active: [A-Z_0-9]*' "$b/fp8-compile.log" || true; } | sort -u | wc -l)
  [ "$sw" -eq 4 ] || die "fp8 build saw $sw of the 4 STOCK_ALL switches"
  # 4. install
  cp "$b/inc/w/build/w1_attn/w1_attn_ext.so" "$TREE/fastpath/w1attn/w1_attn_ext.so"
  cp "$b/fp8/build/w1_attn_fp8_ext.so" "$TREE/fastpath/w1attn/w1_attn_fp8_ext.so"
  for f in w1_attn_ext.so w1_attn_fp8_ext.so; do log "Pg62 $f $(sha256sum < "$TREE/fastpath/w1attn/$f" | cut -c1-64)"; done
}

cmd_build(){
  local src=${GARNER_SRC:?set GARNER_SRC to a checkout of garnermccloud/dgx-station-recipes}
  [ "$(git -C "$src" rev-parse HEAD)" = "$PIN" ] || die "GARNER_SRC is not at $PIN"
  [ -e "$TREE" ] && [ "${FORCE:-0}" != 1 ] && die "$TREE exists (FORCE=1 to rebuild)"
  rm -rf "$TREE" "$OVL"; mkdir -p "$WORK"
  cp -a "$src/deepseek-v4.1-flash" "$TREE"
  local p; for p in "$PATCHES"/0[1-5]-*.patch; do patch -d "$TREE" -p1 --forward --no-backup-if-mismatch < "$p" >/dev/null || die "patch $(basename "$p")"; log "applied $(basename "$p")"; done
  log "building libw6mk.so (nvcc sm_103a in $IMAGE, CPU only)"
  docker run --rm --network none -v "$TREE/kernels/w6-moe/markov:/w" -w /w --entrypoint bash "$IMAGE" -c \
    "nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -Xcompiler -fPIC -shared w6mk.cu -o /w/libw6mk.so 2>/dev/null"
  mv "$TREE/kernels/w6-moe/markov/libw6mk.so" "$TREE/fastpath/libw6mk.so"
  cp -a "$TREE/docker/overlay-58132/vllm" "$OVL"
  cp -a "$PATCHES/overlay-vllm/." "$OVL/"
  [ "$(sha1sum < "$OVL/v1/worker/gpu/sample/gumbel.py" | cut -c1-40)" = $WANT_GUMBEL ] || die "overlay gumbel.py hash"
  [ "$(sha1sum < "$OVL/v1/worker/gpu/spec_decode/rejection_sampler_utils.py" | cut -c1-40)" = $WANT_RSU ] || die "overlay rsu hash"
  if [ "${DS41_PG62:-0}" = 1 ]; then pg62_build; fi
  cp "$WATCHDOG" "$WORK/sidecar_watchdog_p3.py"
  policy_env > "$WORK/policy.env"
  ( cd "$WORK" && find tree/fastpath tree/serve tree/config overlay-vllm sidecar_watchdog_p3.py policy.env -type f \
      ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum ) > "$WORK/BUILD-SHA256SUMS"
  log "built: $(wc -l < "$WORK/BUILD-SHA256SUMS") files hashed; libw6mk $(sha256sum "$TREE/fastpath/libw6mk.so" | cut -c1-16)"
}

cmd_render(){ policy_env | sed 's/^/env /'; echo "bash $TREE/serve/serve.sh start"; }

cmd_up(){
  [ -f "$WORK/policy.env" ] || die "run build first"
  : "${MODEL_DIR:?set MODEL_DIR}" "${DS41_HOME:?set DS41_HOME}"
  docker ps --format '{{.Names}}' | grep -qx ds41 && die "ds41 already running"
  # 2026-10-31 (H1): a killed server or a watchdog trip leaves /dev/shm/vllm_peer_tier{2,3} with published != done;
  # the --once check below would then read a stale-tag trip and refuse. With no ds41/ds41-sidecar running the
  # segments are dead state, so remove them here exactly as cmd_down does (and say so).
  if ls /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3 >/dev/null 2>&1; then
    docker ps --format '{{.Names}}' | grep -qx ds41-sidecar && die "stale peer segment with ds41-sidecar running: run 'launch-pg61.sh down' first"
    docker run --rm -v /dev/shm:/dev/shm --entrypoint /bin/rm "$IMAGE" -f /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3
    ls /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3 >/dev/null 2>&1 && die "could not remove stale peer segments"
    log "removed stale peer segments left by a previous run (nothing was running)"
  fi
  local envs; mapfile -t envs < "$WORK/policy.env"
  ( cd "$TREE" && env "${envs[@]}" DS41_HOME="$DS41_HOME" DS41_MODEL_DIR="$MODEL_DIR" bash serve/serve.sh start )
  for c in ds41 ds41-sidecar; do docker update --restart no "$c" >/dev/null; done
  local i; for i in $(seq 1 180); do curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED" && break; sleep 5; done
  curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED" || die "not bound on :$PORT"
  cmd_verify
  pgrep -f "sidecar_watchdog_p3.py --server ds41" >/dev/null && { log "watchdog already running"; return 0; }
  python3 "$WORK/sidecar_watchdog_p3.py" --server ds41 --sidecar ds41-sidecar --mode report --once \
     --status "$WORK/wdp3.once.json" --log "$WORK/wdp3.log" >/dev/null || die "watchdog --once refused (see $WORK/wdp3.log)"
  grep -q '"trip": null' "$WORK/wdp3.once.json" || die "watchdog --once reports a trip: $(cat "$WORK/wdp3.once.json")"
  ( setsid nohup python3 "$WORK/sidecar_watchdog_p3.py" --server ds41 --sidecar ds41-sidecar --mode enforce \
      --status "$WORK/wdp3.status.json" --log "$WORK/wdp3.log" >/dev/null 2>&1 </dev/null & )
  sleep 3; grep -q '"trip": null' "$WORK/wdp3.status.json" || die "watchdog not armed"
  log "UP on :$PORT; watchdog ENFORCE armed (status $WORK/wdp3.status.json)"
  # 2026-10-31 (A8): soak clock. Every up starts a new soak; the file is the receipt the recipe quotes.
  printf '{"soak_start_utc": "%s", "soak_start_local": "%s", "epoch": %s}\n' "$(date -u +%FT%TZ)" "$(date '+%F %T %Z')" "$(date +%s)" > "$WORK/soak-start.json"
  log "soak clock started: $(cat "$WORK/soak-start.json")"
}

cmd_verify(){
  local g r l s
  g=$(docker exec ds41 sha1sum $VP/sample/gumbel.py | cut -c1-40)
  r=$(docker exec ds41 sha1sum $VP/spec_decode/rejection_sampler_utils.py | cut -c1-40)
  l=$(docker exec ds41 sha256sum /fastpath/libw6mk.so | cut -c1-64)
  s=$(docker exec ds41 printenv PEER3_SPINS || true)
  [ "$g" = $WANT_GUMBEL ] && [ "$r" = $WANT_RSU ] && [ "$l" = "$(sha256sum < "$TREE/fastpath/libw6mk.so" | cut -c1-64)" ] \
    && [ "$s" = 1000000000000 ] || die "in-container check: gumbel $g rsu $r libw6mk ${l:0:16} PEER3_SPINS=$s"
  log "verify OK: gumbel ${g:0:12} rsu ${r:0:12} libw6mk ${l:0:16} PEER3_SPINS=$s"
  if [ "${DS41_PG62:-0}" = 1 ]; then
    local i f e
    i=$(docker exec ds41 sha256sum /fastpath/w1attn/w1_attn_ext.so | cut -c1-64)
    f=$(docker exec ds41 sha256sum /fastpath/w1attn/w1_attn_fp8_ext.so | cut -c1-64)
    e=$(docker exec ds41 printenv DS41_W1_COMPRESSED_FP8 || true)
    [ "$i" = "$(sha256sum < "$TREE/fastpath/w1attn/w1_attn_ext.so" | cut -c1-64)" ] \
      && [ "$f" = "$(sha256sum < "$TREE/fastpath/w1attn/w1_attn_fp8_ext.so" | cut -c1-64)" ] && [ "$e" = 1 ] \
      || die "in-container Pg62 check: w1_attn_ext ${i:0:16} w1_attn_fp8_ext ${f:0:16} DS41_W1_COMPRESSED_FP8=$e"
    log "verify Pg62 OK: w1_attn_ext.so $i w1_attn_fp8_ext.so $f DS41_W1_COMPRESSED_FP8=$e"
  fi
}

cmd_down(){
  local p; for p in $(pgrep -f "sidecar_watchdog_p3.py --server ds41" || true); do kill "$p"; done
  docker unpause ds41-sidecar >/dev/null 2>&1 || true
  ( cd "$TREE" && DS41_HOME="${DS41_HOME:?set DS41_HOME}" bash serve/serve.sh down ) || true
  docker run --rm -v /dev/shm:/dev/shm --entrypoint /bin/rm "$IMAGE" -f /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3
  # 2026-10-31 (H1/H3): verify, do not assume.
  ls /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3 >/dev/null 2>&1 && die "down: peer segments still present"
  docker ps --format '{{.Names}}' | grep -qxE 'ds41|ds41-sidecar' && die "down: ds41 containers still running"
  pgrep -f "sidecar_watchdog_p3.py --server ds41" >/dev/null && die "down: p3 watchdog still running"
  log "down; peer shm removed (verified absent), containers stopped, watchdog stopped"
}

case "${1:-}" in
  build) cmd_build ;; up) cmd_up ;; down) cmd_down ;; verify) cmd_verify ;; render) cmd_render ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac

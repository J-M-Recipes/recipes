#!/usr/bin/env bash
# f3_img.sh — build flare-cli:7db06e0c (job C of f3_prep.sh failed at checkout: shallow clone lacked the commit)
set -uo pipefail
F=$HOME/dsv41/flarecmp; P=$F/prep-F3; S=$F/hf-src-7db06e0c
exec >> "$P/img.log" 2>&1
log(){ printf '%s [F3img] %s\n' "$(date '+%F %T %Z')" "$*"; }
cd "$S" || { log "no $S"; exit 1; }
git fetch -q --depth 1 origin 7db06e0cdabc4efb99aacbc0100ca7dad74e4bff && git checkout -q -f FETCH_HEAD && git clean -fdxq || { log "checkout failed"; echo FAIL > "$P/img.status"; exit 1; }
[[ "$(git rev-parse HEAD)" == 7db06e0cdabc4efb99aacbc0100ca7dad74e4bff ]] || { log "HEAD mismatch $(git rev-parse HEAD)"; echo FAIL > "$P/img.status"; exit 1; }
log "building flare-cli:7db06e0c from $(git rev-parse HEAD)"
t0=$(date +%s)
DOCKER_BUILDKIT=1 nice -n 10 docker build -t flare-cli:7db06e0c "$S" > "$P/build-7db06e0c.log" 2>&1; rc=$?
log "build rc=$rc after $(( $(date +%s)-t0 ))s"
[[ $rc -eq 0 ]] && echo OK > "$P/img.status" || echo FAIL > "$P/img.status"

#!/usr/bin/env bash
# Stage Muse-Spark-Open-816B-A42B-NVFP4 @4239e4db from Ark (CIFS ro) to /models RAID0 NVMe, then sha256 -c vs HF LFS oids.
set -uo pipefail
SRC=/mnt/miloark/Milo-Ark/models/meta-models/Muse-Spark-Open-816B-A42B-NVFP4/4239e4db19376904f2e82c167e78f70abb5455fe
DST=/models/Muse-Spark-Open-816B-A42B-NVFP4/4239e4db19376904f2e82c167e78f70abb5455fe
mkdir -p "$DST"; exec >> /models/Muse-Spark-Open-816B-A42B-NVFP4/stage.log 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
log "STAGE START"
rsync -a --info=progress2 --no-inc-recursive "$SRC/" "$DST/"; rc=$?; log "rsync rc=$rc"
[ $rc = 0 ] || exit $rc
cd "$DST/original" && sha256sum -c ../provenance/HUB-LFS-SHA256SUMS > ../sha256-verify.txt 2>&1; rc=$?
log "sha256 rc=$rc ok=$(grep -c ': OK$' ../sha256-verify.txt) fail=$(grep -vc ': OK$' ../sha256-verify.txt)"
[ $rc = 0 ] && touch /models/Muse-Spark-Open-816B-A42B-NVFP4/STAGE_DONE
log "STAGE END"

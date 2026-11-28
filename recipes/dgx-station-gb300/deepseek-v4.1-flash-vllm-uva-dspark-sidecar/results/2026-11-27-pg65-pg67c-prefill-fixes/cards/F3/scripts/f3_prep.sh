#!/usr/bin/env bash
# f3_prep.sh — prep for Card F3 (Flare gb300 core d683cc1a @ HF b55be385 vs Pg67c). No lane impact: disk cleanup James
# approved 2026-11-27 ("del 1,2,4,5,6"), core download, runtime image build. Three jobs in parallel, then a PREP END line.
set -uo pipefail
F=$HOME/dsv41/flarecmp; P=$F/prep-F3
[[ -e $P/prep.log ]] && { echo "REFUSE: $P/prep.log exists"; exit 5; }
mkdir -p "$P" || exit 5
exec >> "$P/prep.log" 2>&1
log(){ printf '%s [F3prep] %s\n' "$(date '+%F %T %Z')" "$*"; }
D20=/models/flare-cache/DeepSeek-V4.1-Flash-1017-FLARE-gb300-1120
D26=/models/flare-cache/DeepSeek-V4.1-Flash-1017-FLARE-gb300-1126
REPO=local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE
REV=b55be3850c0f9dedc98ed7b0767889801b7e41b7
ARK=/mnt/miloark/Milo-Ark/models
HF=$HOME/.local/bin/hf
free_g(){ df -BG --output=avail /models | tail -1 | tr -dc '0-9'; }

log "===== F3 PREP START ===== free /models: $(free_g) GiB"

# ---- row 1: the 8 stray Engram shard files (not in HF tree 109e07d4 nor in gb300 HEAD)
mapfile -t STRAY < <(ls "$D20"/deployments/gb300-tp1/engram-[01]-shard-[01]-weights.safetensors "$D20"/deployments/gb300-tp1/engram-[01]-shard-[01]-scales.safetensors 2>/dev/null)
if [[ ${#STRAY[@]} -eq 8 ]]; then rm -f -- "${STRAY[@]}" && log "ROW1 deleted 8 stray shard files; free $(free_g) GiB"
else log "ROW1 SKIP: expected 8 stray files, found ${#STRAY[@]}"; fi

# ---- job A: download the new core into a hardlink-seeded copy (only changed files transfer)
job_dl(){
  [[ -e $D26 ]] && { log "DL REFUSE: $D26 exists"; echo FAIL > "$P/dl.status"; return 1; }
  cp -al "$D20" "$D26" && rm -rf "$D26/.cache" && cp -a "$D20/.cache" "$D26/.cache" || { log "DL seed copy failed"; echo FAIL > "$P/dl.status"; return 1; }
  log "DL seeded $D26 by hardlink; starting hf download $REPO@$REV"
  local t0; t0=$(date +%s)
  "$HF" download "$REPO" --revision "$REV" --local-dir "$D26" > "$P/hf.log" 2>&1; local rc=$?
  log "DL hf download rc=$rc after $(( $(date +%s)-t0 ))s"
  [[ $rc -eq 0 ]] || { echo FAIL > "$P/dl.status"; return 1; }
  # completeness against the Hub tree (name + size), then drop names that are not part of this revision
  python3 - "$D26" "$F/f3-tree.json" > "$P/dl-verify.txt" 2>&1 <<'PY'
import json, os, sys
root, treef = sys.argv[1], sys.argv[2]
tree = {f["path"]: f["size"] for f in json.load(open(treef))}
bad = [p for p, s in tree.items() if not os.path.isfile(os.path.join(root, p)) or os.path.getsize(os.path.join(root, p)) != s]
extra = []
for dp, dns, fns in os.walk(root):
    dns[:] = [d for d in dns if d != ".cache"]
    for fn in fns:
        rel = os.path.relpath(os.path.join(dp, fn), root)
        if rel not in tree: extra.append(rel)
for rel in extra: os.unlink(os.path.join(root, rel))
print("tree_files", len(tree), "missing_or_size_diff", len(bad), bad[:5], "extra_removed", len(extra), extra[:12])
sys.exit(1 if bad else 0)
PY
  local vrc=$?; log "DL verify rc=$vrc: $(cat "$P/dl-verify.txt" | cut -c1-400)"
  [[ $vrc -eq 0 ]] && echo OK > "$P/dl.status" || echo FAIL > "$P/dl.status"; }

# ---- job B: rows 2,4,5,6 — hash the local copy against the Ark's SHA256SUMS, delete only on a full pass
hash_del(){ # hash_del <local dir> <ark rev path>
  local L=$1 A=$ARK/$2; local n; n=$(basename "$L"); local W=$P/hash-$n; mkdir -p "$W"
  [[ $L == /models/* && -d $L && -f $A/SHA256SUMS && -f $A/.miloark_download_complete ]] || { log "ROW $n SKIP: precondition failed"; return 1; }
  docker ps -q | xargs -r docker inspect -f '{{range .Mounts}}{{.Source}} {{end}}' | tr ' ' '\n' | grep -q "^$L" && { log "ROW $n SKIP: mounted by a running container"; return 1; }
  grep -a '  original/' "$A/SHA256SUMS" | grep -av '  original/\.cache/' | grep -av '@eaDir' | sed 's#  original/#  #' | sort -u -k2 > "$W/sums.txt"
  # every local file >= 1 MiB must be covered by the Ark sums
  ( cd "$L" && find . -type f -size +1M ! -path './.cache/*' -printf '%P\n' | sort ) > "$W/local-big.txt"
  sed 's/^[0-9a-f]*  //' "$W/sums.txt" | sort > "$W/sums-paths.txt"
  local unc; unc=$(comm -23 "$W/local-big.txt" "$W/sums-paths.txt" | wc -l)
  [[ $unc -eq 0 ]] || { log "ROW $n KEEP: $unc local files not covered by Ark SHA256SUMS ($(comm -23 "$W/local-big.txt" "$W/sums-paths.txt" | head -3 | tr '\n' ' '))"; return 1; }
  awk 'NR==FNR{w[$0]=1;next}{p=$0; sub(/^[0-9a-f]+  /,"",p); if(p in w) print}' "$W/local-big.txt" "$W/sums.txt" > "$W/sums-big.txt"
  local nl; nl=$(wc -l < "$W/sums-big.txt"); local nb; nb=$(wc -l < "$W/local-big.txt")
  [[ $nl -eq $nb ]] || { log "ROW $n KEEP: sums lines $nl != local big files $nb"; return 1; }
  split -n l/12 -d "$W/sums-big.txt" "$W/chunk-"
  local t0; t0=$(date +%s); local pids=() c
  for c in "$W"/chunk-*; do ( cd "$L" && sha256sum -c --quiet "$c" > "$c.out" 2>&1; echo $? > "$c.rc" ) & pids+=($!); done
  wait "${pids[@]}"
  local fail=0; for c in "$W"/chunk-??; do [[ "$(cat "$c.rc" 2>/dev/null)" == 0 ]] || fail=1; done
  local sz; sz=$(du -s --block-size=1G "$L" | cut -f1)
  if [[ $fail -eq 0 ]]; then rm -rf --one-file-system -- "$L" && log "ROW $n: $nb/$nb files sha256-equal to Ark sums in $(( $(date +%s)-t0 ))s; DELETED ${sz} GiB; free $(free_g) GiB"
  else log "ROW $n KEEP: sha256 mismatch ($(cat "$W"/chunk-*.out | head -3 | tr '\n' ' ' | cut -c1-300))"; return 1; fi; }
job_rows(){ local ok=1
  hash_del /models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40 deepseek-ai/DeepSeek-V4.1-Flash/df42c109f1defefcbfcedbe7d905718a12266e40 || ok=0
  hash_del /models/GLM-5.3-NVFP4-radixark-6e389189 RadixArk/GLM-5.3-NVFP4/6e389189d564d9d3e4ea7fa284fe91f136d8ae2a || ok=0
  hash_del /models/GLM-5.3-ABLITERATED-NVFP4 dealignai/GLM-5.3-ABLITERATED-NVFP4/371bdb985d0124e76348c91e4a8fcf3a9d719d09 || ok=0
  hash_del /models/GLM-5.3-Flash-FP8-zai-04c4e9e9 zai-org/GLM-5.3-Flash/04c4e9e95c5da8862dced7e5056455116f83a7e0 || ok=0
  [[ $ok -eq 1 ]] && echo OK > "$P/rows.status" || echo PARTIAL > "$P/rows.status"; }

# ---- job C: runtime image from the commit the core's card requires (7db06e0c), repo's own Dockerfile, no host apt change
job_img(){ local S=$F/hf-src-7db06e0c
  if docker image inspect flare-cli:7db06e0c >/dev/null 2>&1; then log "IMG already present"; echo OK > "$P/img.status"; return 0; fi
  rm -rf "$S"; cp -a "$F/hf-src" "$S" && cd "$S" && git fetch -q origin && git checkout -q -f 7db06e0cdabc4efb99aacbc0100ca7dad74e4bff && git clean -fdxq \
    || { log "IMG checkout failed"; echo FAIL > "$P/img.status"; return 1; }
  log "IMG building flare-cli:7db06e0c from $(git rev-parse HEAD)"
  local t0; t0=$(date +%s)
  DOCKER_BUILDKIT=1 nice -n 10 docker build -t flare-cli:7db06e0c "$S" > "$P/build-7db06e0c.log" 2>&1; local rc=$?
  log "IMG build rc=$rc after $(( $(date +%s)-t0 ))s"
  [[ $rc -eq 0 ]] && echo OK > "$P/img.status" || echo FAIL > "$P/img.status"; }

job_dl & PA=$!
job_rows & PB=$!
job_img & PC=$!
wait $PA $PB $PC
log "===== F3 PREP END ===== dl=$(cat $P/dl.status 2>/dev/null) rows=$(cat $P/rows.status 2>/dev/null) img=$(cat $P/img.status 2>/dev/null) free /models: $(free_g) GiB"

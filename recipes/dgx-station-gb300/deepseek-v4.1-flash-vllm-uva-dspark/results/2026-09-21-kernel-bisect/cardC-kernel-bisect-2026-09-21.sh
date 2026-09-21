#!/usr/bin/env bash
# CARD C (2026-09-21) — bisect the tool-shaped logit drift across vLLM nightlies, hook ON, v19 flags (off54, fp8_ds_mla), TF split only.
# Ladder (all carry DSV4.1 on main; kernel PR sets verified via GitHub compare):
#   2671fedf (09-13): #56214 model support only            -> no perf kernels
#   af1c0149 (09-16): + #56464 DeepSelect, #56962 Mega-mHC, #56512 Engram prefetch
#   0bfc7a15 (09-17): + #56935 FlashMLA mega-attn + NVFP4 KV, #56568/#57204 MegaMoE
#   dee37d89 (09-18, v19): + #56266 Mega-Gate                -> known 11.45% agent-doc flips
# Reference: v14nohook TF capture (0909 image, positional). v18 hook = 0.56%. Instrument: tf_logprob capture + tf_split (cold by construction).
# Each boot is a new autotune hash (~16 min live tune on 0.6.18.post1); TF capture ~10 min. ~35 min/rung, 3 rungs.
# Per rung ALSO: agent_fixture_o (acceptance tool_json/shell — the promotion-rule half) and knee r1 (speed context).
# Rules: never :30003 / dsfv-*, never glm53-*, never docker rm, stop-and-keep, STOP file honoured, v18 restored at end. Hook must fire (HBM_expert=) or the rung is VOID.
# Written by Milo (Hermes milo profile, claude-fable-5-1 via anthropic) for James Meadlock.
set -uo pipefail
set -f
cd /home/milo/dsv41
D=/home/milo/dsv41/cardC-2026-09-21; mkdir -p "$D"
FID=/home/milo/dsv41/fidelity-2026-09-20
LEDGER=/home/milo/dsv41/results/ledger.md; CAMP=/home/milo/dsv41/overnight-campaign-2026-09-20.log; STOPF=/home/milo/dsv41/STOP-CAMPAIGN
V18=dsv41-vllm-v18-cgsizes-BOUND-REF
S=/home/milo/pin-hot-experts/scripts
exec >> "$CAMP" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-cardC-runner (nohup) · brief by Milo · James: go all (2026-09-20 22:15 CDT)\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^(dsv41-vllm-|glmf-)'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; return 1; fi
    if docker logs "$n" 2>&1 | grep -qE 'ValueError:|Traceback|RuntimeError:' ; then log "FASTFAIL $n"; docker logs "$n" 2>&1 | grep -E 'ValueError:|RuntimeError:' | tail -3; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
boot_facts(){ docker logs "$1" 2>&1 | grep -E 'vLLM version|Initializing a V1 LLM engine|KV cache format|kv cache block size|HBM_expert=|GPU KV cache size|Autotuning process|Saved [0-9]+|Loaded [0-9]+ configs|autotune cache file' | sed 's/^.*\] //' | cut -c1-200 | head -14; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; restore; exit 0; fi; }
restore(){ stop_keep; drop_caches; docker start "$V18" >/dev/null; wait_bind "$V18" 25 || log "WARN $V18 did not bind"; }
rung(){ local tag=$1 sha=$2 img=vllm/vllm-openai:nightly-$2 name=dsv41-vllm-C-$1-nightly-${2:0:8}-hook-off54-fp8kv-EXP
  stop_check; docker image inspect "$img" >/dev/null 2>&1 || { log "IMAGE MISSING $img — rung $tag skipped"; echo "$tag MISSING-IMAGE" >> "$D/rungs.txt"; return; }
  docker ps -a --format '{{.Names}}' | grep -qx "$name" && { log "REFUSE $name exists"; return; }
  stop_keep; drop_caches
  HOOK=1 OFFGB=54 KVDTYPE=fp8_ds_mla NAME=$name IMAGE=$img bash $S/launch-t3-nightly.sh > "$D/launch-$tag.txt" 2>&1 || { cat "$D/launch-$tag.txt"; echo "$tag LAUNCH-REFUSED" >> "$D/rungs.txt"; return; }
  if ! wait_bind "$name" 110; then docker logs "$name" 2>&1 | tail -30 > "$D/boot-fail-$tag.txt"; docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-BOOT-FAIL"; echo "$tag BOOT-FAIL" >> "$D/rungs.txt"; led "Card C rung $tag BOOT FAIL" "$(grep -iE 'error' $D/boot-fail-$tag.txt | tail -2 | cut -c1-200 | tr '\n' ' ')"; return; fi
  boot_facts "$name" | tee "$D/boot-$tag.txt"
  if ! grep -q "HBM_expert=" "$D/boot-$tag.txt"; then docker stop "$name"; docker rename "$name" "$name-HOOK-NOFIRE-VOID"; echo "$tag VOID-hook" >> "$D/rungs.txt"; return; fi
  bash smoke_vllm.sh 2>&1 | tail -3 > "$D/smoke-$tag.txt"
  ( cd "$D" && MAXPOS=4096 CONC=4 python3 $S/tf_logprob.py capture "$tag" /home/milo/pin-hot-experts/tf_corpus.jsonl ) > "$D/tf-$tag.log" 2>&1
  ( cd "$D" && python3 $S/tf_logprob.py compare "$FID/tf-v14nohook.jsonl" "tf-$tag.jsonl" ) >> "$D/tf-compare.txt" 2>&1
  CAT_ORDER="prose structured code shell_ops tool_json" bash agent_fixture_o.sh "$tag" > "$D/fixture-$tag.log" 2>&1; cp -f "agentfix-$tag.json" "$D/"
  bash knee.sh "$tag-r1" > "$D/knee-$tag-r1.log" 2>&1; cp -f "knee-$tag-r1.json" "$D/"
  echo "$tag OK $sha" >> "$D/rungs.txt"
  led "Card C rung $tag done ($sha)" "$(cd $D && python3 $S/tf_split.py tf-compare-v14nohook-vs-$tag.json | tr '\n' ' ' | cut -c1-400) · accept: $(python3 -c "import json;r=json.load(open('$D/agentfix-$tag.json'))['res'];print({k:r[k]['accept'] for k in ('tool_json','shell_ops','prose')})") · $(grep -h 'conc= 1:' $D/knee-$tag-r1.log | head -1 | cut -c1-60)"; }

log "===== CARD C START (kernel bisect, TF split) ====="
if ps -eo args | grep -q '^bash /home/milo/pin-hot-experts/scripts/cardD'; then log "REFUSE Card D running"; exit 4; fi
test -f "$FID/tf-v14nohook.jsonl" || { log "reference missing"; exit 3; }
led "Card C start" "Bisect ladder 2671fedf (no perf kernels) -> af1c0149 (+DeepSelect, Mega-mHC, Engram prefetch) -> 0bfc7a15 (+FlashMLA mega-attn/NVFP4 KV, MegaMoE); dee37d89 (+Mega-Gate) is v19 = 11.45%. Hook on, off54, fp8_ds_mla; TF split vs v14nohook; fixture acceptance; knee r1."
rung r1 2671fedfc7ae604761990603fc736c0c4f21de57
rung r2 af1c01499b289be555c475669ba50a88e96d846e
rung r3 0bfc7a15d095fe83ecc82b50561a93c177fece2d
{
 echo "rung | image | agent_tool flips | heldout | prose | accept tool_json/shell | C1"
 for t in r1 r2 r3; do
   f="$D/tf-compare-v14nohook-vs-$t.json"; if [[ -f $f ]]; then
     python3 - "$t" "$f" "$D/agentfix-$t.json" "$D/knee-$t-r1.log" <<'EOF'
import json,sys,re
t,f,a,k=sys.argv[1:]; j=json.load(open(f)); agg={}
for kk,v in j["per_doc"].items():
    c="agent_tool" if kk.startswith("parity-agent") else "heldout" if kk.startswith("parity-heldout") else "prose"
    x=agg.setdefault(c,[0,0]); x[0]+=v["positions"]; x[1]+=v["top1_flips"]
fl=lambda c: f"{agg[c][1]/agg[c][0]*100:.2f}%" if c in agg else "-"
try: r=json.load(open(a))["res"]; acc=f"{r['tool_json']['accept']}/{r['shell_ops']['accept']}"
except Exception: acc="-"
try: c1=re.search(r"conc= 1:\s+([\d.]+)", open(k).read()).group(1)
except Exception: c1="-"
print(f"{t} | {j['cand']} | {fl('agent_tool')} | {fl('heldout')} | {fl('prose')} | {acc} | {c1}")
EOF
   else echo "$t | (no capture) | $(grep -h "^$t" $D/rungs.txt)"; fi
 done
 echo "reference rows: v18(0909+hook) 0.56% / 1.08% / 1.06% · accept 0.878/0.908 · C1 172 ; v19(dee37d89) 11.45% / 8.89% / 1.80% · 0.838/0.829 · 183"
} > "$D/VERDICT-cardC.txt" 2>&1
cat "$D/VERDICT-cardC.txt"
restore
led "Card C END" "$(cat $D/VERDICT-cardC.txt | tr '\n' ' ' | cut -c1-900) · rung containers stopped-and-kept · $V18 restored · receipts $D"
log "===== CARD C END ====="

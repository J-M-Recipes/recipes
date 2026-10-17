#!/usr/bin/env bash
# campaign.sh — run SGLang arms from arms-recipe.tsv sequentially. Re-reads arms-recipe.tsv before each arm.
# arms-recipe.tsv columns (tab): name <TAB> env-assignments (space separated KEY=VAL; use '' for none) <TAB> needles (comma list or -) <TAB> status(todo|skip|done|fail)
# Per arm: cluster.sh up → sg_wait (boot ≤ 40 min; dead rank = fail) → gates → parity → bench → spec stats → needles → RESULTS.md row.
set -u
# Campaign runner as used for the 2026-10-17 round. Box-specific: HEAD = ssh target of rank 0; host N = ssh target of rank N.
# Set SPARK_HOSTS="h0 h1 ... h7" (rank order). GLM53_API_KEY must be exported by the caller.
read -r -a _H <<<"${SPARK_HOSTS:?set SPARK_HOSTS to 8 ssh hosts, rank order}"; HEAD=${_H[0]}; host(){ echo "${_H[$1]}"; }
cd "$(dirname "$0")"; G=$(pwd); L=~/hermes/spark-glm53-tp8/launchers; C=recipe-campaign; mkdir -p $C
: "${GLM53_API_KEY:?export GLM53_API_KEY}"
RES=$C/RESULTS.md; [ -f $RES ] || printf '# SGLang campaign results (%s)\n\n| arm | env | boot min | KV tokens | code | prose | explain | C4 | accept | needles | hermes-chain med s | status | note |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|\n' "$(date '+%Y-%m-%d %H:%M')" > $RES
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a $C/campaign.log; }
consec_fail=0
while :; do
  line=$(awk -F'\t' '$4=="todo"{print; exit}' arms-recipe.tsv); [ -z "$line" ] && { log "no todo arms left"; break; }
  name=$(cut -f1 <<<"$line"); env=$(cut -f2 <<<"$line"); needles=$(cut -f3 <<<"$line")
  O=$C/$name; mkdir -p $O; log "ARM $name env=[$env] needles=$needles"
  # power / node sanity
  for n in $(seq 0 7); do ssh -o ConnectTimeout=8 -o BatchMode=yes $(host $n) true || { log "node .$n unreachable — campaign STOP"; exit 2; }; done
  ( cd $L && NAME=glm53_sglang_tp8 LAUNCHER=launch-sglang-tp8.sh ./cluster.sh down >/dev/null 2>&1; sleep 3
    eval "env $env NAME=glm53_sglang_tp8 LAUNCHER=launch-sglang-tp8.sh ./cluster.sh up" ) > $O/launch.log 2>&1
  nl=$(grep -c "^launched" $O/launch.log); t0=$(date +%s)
  if [ "$nl" != 8 ]; then log "launch produced $nl/8 — FAIL"; st=fail; note="launch $nl/8"; else
    st=boot; note=""
    while :; do
      ssh $HEAD "docker logs glm53_sglang_tp8 2>&1 | grep -qE 'The server is fired up'" && { st=up; break; }
      for n in $(seq 0 7); do s=$(ssh $(host $n) "docker inspect -f '{{.State.Status}}' glm53_sglang_tp8 2>/dev/null"); [ "$s" = running ] || { st=fail; note="rank .$n $s"; ssh $(host $n) "docker logs --tail 120 glm53_sglang_tp8 2>&1" > $O/fail-$n.log; break 2; }; done
      [ $(( $(date +%s)-t0 )) -gt 2400 ] && { st=fail; note="boot timeout 40m"; break; }
      sleep 30
    done
    ssh $HEAD "docker logs --tail 150 glm53_sglang_tp8 2>&1" > $O/head-tail.log
  fi
  boot_min=$(( ($(date +%s)-t0)/60 ))
  code=-; prose=-; explain=-; c4=-; acc=-; kvtok=-; nd="-"; hg=-
  if [ "$st" = up ]; then
    ssh $HEAD "docker logs -t glm53_sglang_tp8 2>&1 | grep -iE 'backend|KV Cache is allocated|cuda graph|capture|spec|eagle|draft|fired up|avail mem|autotune' | grep -v 'Decode batch' | cut -c1-300 | head -80" > $O/boot.txt
    kvtok=$(grep -oE "#tokens: [0-9]+" $O/boot.txt | head -1 | awk '{print $2}')
    python3 sg_gates.py > $O/gates.json 2>&1
    python3 parity.py ${BASE_URL:-http://127.0.0.1:8891}/v1 glm-5.3-nvfp4-sglang $O/parity.json > $O/parity.log 2>&1
    python3 sg_bench.py "$name" > $O/bench.log 2>&1
    r=$(tail -1 $O/bench.log); code=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(d['c1']['code']['median'])" "$r" 2>/dev/null || echo -)
    prose=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(d['c1']['prose']['median'])" "$r" 2>/dev/null || echo -)
    explain=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(d['c1']['explain']['median'])" "$r" 2>/dev/null || echo -)
    c4=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(d['c4_agg_tps'])" "$r" 2>/dev/null || echo -)
    acc=$(ssh $HEAD "docker logs --since 15m glm53_sglang_tp8 2>&1 | grep -oE 'accept len: [0-9.]+' | awk '{print \$3}' | sort -n | awk '{a[NR]=\$1} END{if(NR) print a[int(NR/2)+1]; else print \"-\"}'")
    smoke=$(python3 -c "import json;d=json.load(open('$O/gates.json'));print(d['smoke']['content'])" 2>/dev/null || echo "?")
    [ "$smoke" = ROUTE_OK ] || { st=degraded; note="smoke=$smoke"; }
    if [ "$needles" != "-" ]; then nd=""; for t in ${needles//,/ }; do out=$(sed 's#8890#8891#; s/"glm-5.3-nvfp4"/"glm-5.3-nvfp4-sglang"/' needle_nvfp4.py | NEEDLE_MAXTOK=600 python3 - $t 0.5 2>&1 | tail -1); echo "$out" >> $O/needles.jsonl; p=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);print('P' if d['PASS'] else 'F')" "$out" 2>/dev/null || echo E); nd="$nd$t:$p "; done; fi
    hg=$(cd ../hermes-bench && HERMES_HOME=$TMPDIR/hbench HERMES_PROFILE= python3 gauntlet_reps.py "$name" "custom[sglang]" glm-5.3-nvfp4-sglang 3 2>&1 | tee $G/$O/hermes-gauntlet.log | grep -oE '"MEDIAN": [0-9.]+' | awk '{print $2}'); hg=${hg:--}
    [ "$st" = up ] && st=done
    ssh $HEAD "free -m | awk '/^Mem:/{print \"head avail MB\",\$7}'" >> $O/boot.txt
  fi
  [ "$st" = fail ] && consec_fail=$((consec_fail+1)) || consec_fail=0
  printf '| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' "$name" "$env" "$boot_min" "${kvtok:--}" "$code" "$prose" "$explain" "$c4" "$acc" "$nd" "$hg" "$st" "$note" >> $RES
  python3 - "$name" "$st" <<'PY'
import sys; n,s=sys.argv[1:]; L=open('arms-recipe.tsv').read().split('\n')
open('arms-recipe.tsv','w').write('\n'.join((l.rsplit('\t',1)[0]+'\t'+s) if l.split('\t')[0]==n else l for l in L))
PY
  log "ARM $name → $st  code=$code prose=$prose explain=$explain c4=$c4 acc=$acc kv=$kvtok needles=[$nd] hermes=$hg $note"
  [ $consec_fail -ge 3 ] && { log "3 consecutive boot failures — campaign STOP"; break; }
done
# end state: relaunch the best DONE arm (highest code C1 with all needles P) and leave it serving
best=$(python3 - <<'PY2'
import re
rows=[l for l in open('recipe-campaign/RESULTS.md') if l.startswith('| ') and '`' in l]
c=[]
for l in rows:
    f=[x.strip() for x in l.strip().strip('|').split('|')]
    if len(f)<13 or f[11]!='done' or ':E' in f[9] or 'F' in f[9].replace('PASS','') : continue
    try: c.append((float(f[4]), f[0], f[1].strip('`')))
    except: pass
c.sort(reverse=True)
r=[x for x in c if 'recipe' in x[1]]
c=r if r else c
print(c[0][1]+'\t'+c[0][2] if c else '')
PY2
)
if [ -n "$best" ]; then bn=$(cut -f1 <<<"$best"); be=$(cut -f2 <<<"$best"); log "END STATE: relaunching best arm $bn env=[$be]"
  ( cd $L && NAME=glm53_sglang_tp8 LAUNCHER=launch-sglang-tp8.sh ./cluster.sh down >/dev/null 2>&1; sleep 3; eval "env $be NAME=glm53_sglang_tp8 LAUNCHER=launch-sglang-tp8.sh ./cluster.sh up" ) > $C/final-launch.log 2>&1
  log "final launch: $(grep -c ^launched $C/final-launch.log)/8"
else log "END STATE: no passing arm; cluster left as last arm"; fi
log "recipe campaign finished"

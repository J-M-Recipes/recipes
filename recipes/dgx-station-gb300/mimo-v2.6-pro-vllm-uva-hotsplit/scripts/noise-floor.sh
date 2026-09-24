#!/usr/bin/env bash
# noise floor: reboot the SAME control container, recapture greedy + TF as ctrl2; stop-and-keep after.
set -uo pipefail
F=$HOME/mimo26/finish; D=$F/run-2026-09-23; C=mimo26-pro-v20-ctrl-C1-30
exec >> "$D/noise-floor.log" 2>&1
log(){ printf "%s %s\n" "$(date "+%F %T %Z")" "$*"; }
[[ -z $(docker ps -q) ]] || { log "REFUSE: container up"; exit 4; }
sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"; docker start $C >/dev/null; log "start $C"
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:30007/v1/models | grep -q mimo26-pro && break; sleep 10; done; log "bound"
( cd $HOME/mimo26 && bash agent_fixture.sh finish-ctrl2-warm >/dev/null 2>&1 )
( cd $D && python3 $F/mimo_greedy.py capture ctrl2 && python3 $F/mimo_greedy.py compare greedy-ctrl.json greedy-ctrl2.json ) > $D/greedy-ctrl2.log 2>&1
( cd $D && BASE_URL=http://127.0.0.1:30007/v1 MODEL=mimo26-pro MAXPOS=4096 CONC=4 python3 $F/tf_logprob.py capture ctrl2 $F/tf_corpus.jsonl && python3 $F/tf_logprob.py compare tf-ctrl.jsonl tf-ctrl2.jsonl ) > $D/tf-ctrl2.log 2>&1
tail -1 $D/tf-ctrl2.log; grep GREEDY $D/greedy-ctrl2.log
docker stop -t 60 $C >/dev/null; log "NOISE-FLOOR END (stopped-and-kept)"

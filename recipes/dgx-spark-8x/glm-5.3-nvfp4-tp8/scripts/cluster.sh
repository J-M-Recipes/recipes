#!/bin/bash
# cluster.sh — GhettoRak GLM-5.3 FP8 TP8 driver. Runs on the M4 (mgmt LAN .51–.58).
#   ./cluster.sh push     copy launcher to every node (~/glm53-tp8/)
#   ./cluster.sh up       clean + drop caches + boot-flush loop + launch ranks 7..1 then 0
#   ./cluster.sh down     stop the container on all 8 (bilateral teardown)
#   ./cluster.sh status   container state + free mem per node
set -u
D=$(cd "$(dirname "$0")" && pwd)
read -r -a NODES <<<"${SPARK_HOSTS:?set SPARK_HOSTS to 8 ssh hosts, rank order (rank 0 = head)}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
NAME=${NAME:-glm53_nvfp4_tp8}; LAUNCHER=${LAUNCHER:-launch-nvfp4-tp8.sh}
on(){ $SSH "$1" "$2"; }

case "${1:-}" in
push)
  for n in "${NODES[@]}"; do on $n 'mkdir -p ~/glm53-tp8'; scp -q "$D/$LAUNCHER" "$D/bootflush.sh" "$n":glm53-tp8/; on $n 'chmod +x ~/glm53-tp8/*.sh'; echo "pushed $n"; done ;;
down)
  for r in 0 7 6 5 4 3 2 1; do n=${NODES[$r]}; echo "$n $(on $n "docker rm -f $NAME 2>&1 | tail -1; pkill -f bootflush.sh 2>/dev/null; true")"; done ;;
status)
  for n in "${NODES[@]}"; do echo "$n $(on $n "docker ps -a --filter name=$NAME --format '{{.Status}}'; free -g | awk '/^Mem:/{print \"avail\",\$7\"G\"}'" | tr '\n' ' ')"; done ;;
up)
  date
  for n in "${NODES[@]}"; do
    on $n "docker rm -f $NAME >/dev/null 2>&1; sudo find /dev/shm -maxdepth 1 -type f \( -name 'psm_*' -o -name 'sem.*' -o -name 'torch_*' -o -name 'mp-*' -o -name 'vllm*' \) -delete; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
      pkill -f bootflush.sh 2>/dev/null; setsid nohup ~/glm53-tp8/bootflush.sh >/dev/null 2>&1 < /dev/null &
      echo \$(hostname) avail \$(free -g | awk '/^Mem:/{print \$7}')G swappiness \$(cat /proc/sys/vm/swappiness)"
  done
  for r in 7 6 5 4 3 2 1 0; do
    n=${NODES[$r]}
    ENVS="${GMU:+GMU=$GMU }${MAXLEN:+MAXLEN=$MAXLEN }${SPEC_K:+SPEC_K=$SPEC_K }${SPEC_DEPTHS:+SPEC_DEPTHS=$SPEC_DEPTHS }${ADAPT_WIN:+ADAPT_WIN=$ADAPT_WIN }"
    on $n "$ENVS~/glm53-tp8/$LAUNCHER $r" || { echo "rank $r launch FAILED on $n"; exit 1; }
  done
  date ;;
*) echo "usage: $0 push|up|down|status" >&2; exit 2 ;;
esac

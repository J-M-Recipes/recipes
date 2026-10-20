log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }
apply_arm(){ # writes .env.tp4 from .env.tp4.B0 + arm overrides (newline-safe append; the B0 file has no trailing newline)
  cp $ROOT/.env.tp4.B0 $ROOT/.env.tp4; printf '\n' >> $ROOT/.env.tp4
  case $1 in
    B1a) printf '%s\n' 'EXTRA_CONTAINER_ENV="$EXTRA_CONTAINER_ENV DSV41_INDEXER_LOGITS_BUDGET_BYTES=1073741824"' >> $ROOT/.env.tp4 ;;
    B1b) printf '%s\n' 'EXTRA_CONTAINER_ENV="$EXTRA_CONTAINER_ENV DSV41_INDEXER_LOGITS_BUDGET_BYTES=4294967296"' >> $ROOT/.env.tp4 ;;
    B2)  printf '%s\n' 'EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS/--swa-prefix-tails 128/--swa-prefix-tails 64}"' >> $ROOT/.env.tp4 ;;
    B3)  printf '%s\n' 'EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS/--prefill-decode-interval 2/--prefill-decode-interval 4}"' >> $ROOT/.env.tp4 ;;
    B0ctl2) : ;;
    *) log "unknown arm $1"; return 1 ;;
  esac
  ( cd $ROOT && bash -c 'source .env.tp4; echo "SKIP_PREPARE=$SKIP_PREPARE"; echo "SKIP_VERIFY=$SKIP_VERIFY"; echo "MODEL_DIR=$MODEL_DIR"; echo "HF_REVISION=$HF_REVISION"; echo "EXTRA_CONTAINER_ENV=$EXTRA_CONTAINER_ENV"; echo "EXTRA_SGLANG_ARGS=$EXTRA_SGLANG_ARGS"' ) > $R2/$1.rendered-env.txt
  grep -q '^SKIP_PREPARE=1$' $R2/$1.rendered-env.txt || { log "ABORT $1: rendered SKIP_PREPARE is not 1 (would trigger boot.py download over the checkpoint)"; return 1; }
  grep -q '^MODEL_DIR=~/models/DeepSeek-V4.1-Flash-1017$' $R2/$1.rendered-env.txt || { log "ABORT $1: MODEL_DIR wrong"; return 1; }
  n=$(ls $ROOT/../models/DeepSeek-V4.1-Flash-1017/model-*-of-00048.safetensors 2>/dev/null | wc -l); [ "$n" -eq 48 ] || { log "ABORT $1: checkpoint has $n/48 shards"; return 1; }
}
stop_lane(){
  log "stop lane"
  ( cd $ROOT && timeout 180 ./start-tp4.sh stop ) >> $R2/stop-$1.log 2>&1
  for w in $WORKERS; do
    if ssh -o BatchMode=yes milo@10.0.0.$w 'docker ps --format "{{.Names}}" | grep -q dsv41-worker'; then
      log "spark$w worker still up after stop: docker kill by name"
      ssh -o BatchMode=yes milo@10.0.0.$w 'timeout 120 docker kill dsv41-worker; sleep 5; docker rm -f dsv41-worker 2>/dev/null; true' >> $R2/stop-$1.log 2>&1
    fi
  done
  docker ps --format '{{.Names}}' | grep -q '^dsv41-head$' && { log "head still up: docker kill"; timeout 120 docker kill dsv41-head; docker rm -f dsv41-head; }
}
hygiene(){
  for h in 1 $WORKERS; do
    ssh -o BatchMode=yes milo@10.0.0.$h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null; printf "spark%s MemAvail=%sG gpu_apps=%s clk=%s\n" "$(hostname)" "$(awk "/MemAvailable/{printf \"%.0f\", \$2/1048576}" /proc/meminfo)" "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" "$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader)"' >> $LOG 2>&1
    ma=$(ssh -o BatchMode=yes milo@10.0.0.$h 'awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo')
    [ "$ma" -ge 100 ] || { log "ABORT: node 10.0.0.$h MemAvailable ${ma}G < 100"; return 1; }
  done
}
boot_lane(){ # $1 arm
  log "boot $1"; ( cd $ROOT && timeout 1500 ./start-tp4.sh serve ) > $ROOT/logs/$1-boot-$(date +%Y%m%dT%H%M).log 2>&1
  log "serve returned rc=$? for $1"
  for i in $(seq 1 30); do curl -fsS -m 5 http://127.0.0.1:8888/v1/models >/dev/null 2>&1 && break; sleep 10; done
  curl -fsS -m 5 http://127.0.0.1:8888/v1/models >/dev/null 2>&1 || { log "ABORT: $1 not HTTP-ready"; docker top dsv41-head -o pid,etime,wchan,args >> $R2/$1-hang-diag.txt 2>&1; docker logs --tail 20 dsv41-head >> $R2/$1-hang-diag.txt 2>&1; return 1; }
  kv=$(grep -o "max_total_num_tokens=[0-9]*" $(ls -t $ROOT/logs/$1-boot-*.log | head -1) | tail -1); log "$1 HTTP-ready $kv"
}

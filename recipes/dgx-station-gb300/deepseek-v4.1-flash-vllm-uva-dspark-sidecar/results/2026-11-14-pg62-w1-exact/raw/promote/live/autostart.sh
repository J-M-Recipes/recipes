#!/bin/bash
# autostart-pg61.sh: bring the production Pg61 lane (:30006) back after a reboot or power loss.
# Installed 2026-11-04 (James: "4) y"). Run by pg61-autostart.service at boot as user milo.
# Same steps as the proven post-reboot restore (promote/restore-pg61-after-*.sh, 10-30 05:07):
#   wait for GPUs + /models -> drop caches -> launch-pg61.sh up (RTX sidecar, then GB300 vLLM, hash verify, p3 watchdog ENFORCE).
# Skips (exit 0) when:
#   - ~/dsv41/m3/NO-AUTOSTART exists (touch it before a window that must not come back as Pg61), or
#   - ~/dsv41/m3/logs/MAINTENANCE exists, or
#   - any container is already running (the lane is up, or someone started something else first).
set -uo pipefail
GR=~/dsv41/garner
M3=~/dsv41/m3
LO=$M3/LANE-OWNER.txt
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LOG=$GR/promote/autostart-$(date +%Y%m%d-%H%M%S).log
exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }
lo(){ echo "- $(ts) AUTOSTART $* (pg61-autostart.service; log $LOG)" >> "$LO"; }
echo "$(ts) [autostart] start (boot $(cut -c1-8 /proc/sys/kernel/random/boot_id), up $(cut -d' ' -f1 /proc/uptime)s)"

[[ -e $M3/NO-AUTOSTART ]] && { echo "$(ts) skip: $M3/NO-AUTOSTART present"; lo "skipped: NO-AUTOSTART flag"; exit 0; }
[[ -e $M3/logs/MAINTENANCE ]] && { echo "$(ts) skip: MAINTENANCE flag"; lo "skipped: MAINTENANCE flag"; exit 0; }

for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 5; done
running=$(docker ps --format '{{.Names}}' | tr '\n' ' ')
if [[ -n ${running// /} ]]; then
  echo "$(ts) skip: containers already running: $running"
  [[ $running == *ds41* ]] || lo "skipped: other containers running ($running)"
  exit 0
fi

for i in $(seq 1 120); do
  n=$(nvidia-smi -L 2>/dev/null | grep -cE 'GB300|RTX PRO 6000')
  [[ $n -ge 2 ]] && break; sleep 5
done
[[ ${n:-0} -ge 2 ]] || { echo "$(ts) FAIL: GPUs not ready (n=${n:-0})"; lo "FAILED: GPUs not ready, lane DARK"; exit 1; }
for i in $(seq 1 60); do [[ -d $MODEL_DIR ]] && mountpoint -q /models && break; sleep 5; done
[[ -d $MODEL_DIR ]] || { echo "$(ts) FAIL: $MODEL_DIR missing"; lo "FAILED: /models not mounted, lane DARK"; exit 1; }

nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
df -h / /models | tail -2; free -g | sed -n 2p; grep -A1 '^md' /proc/mdstat
# 2026-11-04 kernel-1024 reboot: the first attempt at boot+72 s died in the RTX sidecar with
# "CUDA unknown error" at torch.cuda.set_device; the same command by hand 9 minutes later worked.
# So: wait until the box has been up >= MIN_UP s, and retry the whole `up` (after a clean `down`) up to 3 times.
MIN_UP=${MIN_UP:-240}
up_s=$(cut -d. -f1 /proc/uptime); (( up_s < MIN_UP )) && { echo "$(ts) waiting $((MIN_UP-up_s)) s for boot to settle"; sleep $((MIN_UP-up_s)); }
for attempt in 1 2 3; do
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
  if env WORK=$GR/pg62-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
    echo "$(ts) [autostart] UP (attempt $attempt)"
    lo "done: Pg61 serving :30006 after boot (attempt $attempt); p3 watchdog ENFORCE"
    exit 0
  fi
  rc=$?; echo "$(ts) [autostart] attempt $attempt FAILED rc=$rc"
  lo "attempt $attempt FAILED rc=$rc: launch-pg61.sh up"
  env WORK=$GR/pg62-work DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true
  (( attempt < 3 )) && sleep 120
done
echo "$(ts) [autostart] FAILED after 3 attempts"; lo "FAILED after 3 attempts; lane DARK, needs a look"; exit 1

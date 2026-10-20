#!/bin/bash
# stop current lane (B3), hygiene, render B1a env with asserts, serve. No battery.
R2=~/bench-r2; ROOT=~/dsv41-4x-spark; LOG=$R2/serve-b1a.log
sed -n "/^apply_arm()/,/^}/p; /^stop_lane()/,/^}/p; /^hygiene()/,/^}/p; /^boot_lane()/,/^}/p; /^log()/p" $R2/chain-b.sh > $R2/_fns.sh
WORKERS="2 3 4"; log(){ echo "[$(date "+%F %T")] $*" >> $LOG; }
source $R2/_fns.sh
stop_lane B3; apply_arm B1a || exit 1; hygiene || exit 1; boot_lane B1a-serve || exit 1
log "B1a serving"

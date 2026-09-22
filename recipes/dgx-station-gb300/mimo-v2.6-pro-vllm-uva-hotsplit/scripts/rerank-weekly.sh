#!/usr/bin/env bash
# Weekly: merge live-lane routing counts into a CANDIDATE hot list. Never restarts the lane; applying = next boot with COUNTS=/w/dumps/expert_hist_live_mix.json
cd ${WORKDIR:-$PWD}
test -f live/counts.json || { echo "$(date -Is) no live counts"; exit 0; }
cp live/counts.json live/counts-$(date +%Y%m%d).json
python3 hotsplit_rerank.py live/counts.json dumps/expert_hist_mix.json dumps/expert_hist_live_mix.json 1.0

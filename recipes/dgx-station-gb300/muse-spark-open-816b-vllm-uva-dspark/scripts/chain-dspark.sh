#!/usr/bin/env bash
# DSpark depth/adaptive chain, autotune off, graphs on, offload 310. Each arm: boot, gates, C1/C4, /metrics, stop+keep.
cd /models/Muse-Spark-Open-816B-A42B-NVFP4
DS=/models/Muse-Spark-Open-816B-A42B-NVFP4/dspark/ffc3eb50
NOAT='--kernel-config {"enable_flashinfer_autotune":false}'
run(){ RUN=$1 NAME=$2 OFF=310 WAITN=900 EAGER=" " EXTRA="$NOAT --speculative-config $3" bash s1-boot5.sh; }
echo "$(date) CHAIN START" >> runs/chain-dspark.log
run s1-boot7-dspark-k3 muse-s1-dspark-k3 "{\"method\":\"dspark\",\"model\":\"$DS\",\"num_speculative_tokens\":3}"
run s1-boot8-dspark-k2 muse-s1-dspark-k2 "{\"method\":\"dspark\",\"model\":\"$DS\",\"num_speculative_tokens\":2}"
run s1-boot9-dspark-k7-adaptive muse-s1-dspark-k7a "{\"method\":\"dspark\",\"model\":\"$DS\",\"num_speculative_tokens\":7,\"enable_adaptive_verification\":true}"
echo "$(date) CHAIN END" >> runs/chain-dspark.log

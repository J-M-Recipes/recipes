#!/usr/bin/env bash
# Muse quality A/B (2026-10-16, Milo): k0 vs DSpark k=2. Same image/weights/offload(310)/graphs/autotune-off; only the speculator differs.
# Per arm after boot+gates: greedy 20 capture, BFCL 600 (low effort, C4, MAXTOK 4096), GSM8K 200 (low effort, C4). Stop+keep.
cd /models/Muse-Spark-Open-816B-A42B-NVFP4
Q=/models/Muse-Spark-Open-816B-A42B-NVFP4/quality
DS=/models/Muse-Spark-Open-816B-A42B-NVFP4/dspark/ffc3eb50
NOAT='--kernel-config {"enable_flashinfer_autotune":false}'
post(){ echo "cd $Q && mkdir -p $1 && cd $1 && BASE=http://127.0.0.1:18100/v1 MODEL=muse MAXTOK=600 python3 $Q/s2_greedy.py capture $1 . > greedy.log 2>&1; \
BASE_URL=http://127.0.0.1:18100/v1 MODEL=muse CONC=4 MAXTOK=4096 python3 /models/Step-5-Preview-NVFP4/runs/s2-common/bfcl_gate_orig.py $1 $HOME/pin-hot-experts/bfcl_data . > bfcl.txt 2>&1; \
BASE=http://127.0.0.1:18100/v1 python3 $Q/gsm_eval.py gsm-$1.jsonl 200 > gsm.txt 2>&1; curl -s http://127.0.0.1:18100/metrics | grep -E '^vllm:spec_decode' > spec-metrics.txt"; }
export CTK='{"reasoning_effort":"low"}'
echo "$(date) QUALITY START" >> $Q/chain.log
RUN=q-k0 NAME=muse-q-k0 OFF=310 WAITN=900 EAGER=" " EXTRA="$NOAT" POST="$(post k0)" bash s1-boot6.sh
echo "$(date) k0 done" >> $Q/chain.log
RUN=q-k2 NAME=muse-q-k2 OFF=310 WAITN=900 EAGER=" " EXTRA="$NOAT --speculative-config {\"method\":\"dspark\",\"model\":\"$DS\",\"num_speculative_tokens\":2}" POST="$(post k2)" bash s1-boot6.sh
echo "$(date) k2 done" >> $Q/chain.log
cd $Q && python3 s2_greedy.py compare k0/greedy-k0.json k2/greedy-k2.json . > greedy-compare.txt 2>&1
echo "$(date) QUALITY END" >> $Q/chain.log

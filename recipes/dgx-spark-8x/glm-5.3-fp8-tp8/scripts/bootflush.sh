#!/bin/bash
# glm53 bootflush: drop page cache every 60 s for 50 min during the TP8 boot (GB10 UMA: page cache eats CUDA-visible memory)
for i in $(seq 1 50); do sync; echo 3 | sudo -S -p '' tee /proc/sys/vm/drop_caches >/dev/null; sleep 60; done

#!/bin/bash
# Lane C image build, capped buildx builder (12g/10 little cores), host guard at 8 GiB MemAvailable
cd ~/spark-ds41f
echo "START $(date -u +%FT%TZ) commit $(git rev-parse HEAD) memavail_kB $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"
nice -n 10 ionice -c3 bash ~/ds41f-buildcmd.sh &
BPID=$!
MIN=999999999
while kill -0 $BPID 2>/dev/null; do
  A=$(awk "/MemAvailable/{print \$2}" /proc/meminfo)
  [ $A -lt $MIN ] && MIN=$A
  if [ $A -lt 8388608 ]; then
    echo "GUARD: MemAvailable ${A} kB < 8 GiB at $(date -u +%FT%TZ); stopping builder"
    docker buildx stop ds41f-capped; kill $BPID; sleep 5; break
  fi
  sleep 2
done
wait $BPID; RC=$?
echo "BUILD_RC=$RC min_memavail_kB=$MIN END $(date -u +%FT%TZ)"
docker image inspect vllm-ds41f-kkref:19f2c20ed4d6-r6x --format "IMAGE_ID {{.Id}}" 2>&1

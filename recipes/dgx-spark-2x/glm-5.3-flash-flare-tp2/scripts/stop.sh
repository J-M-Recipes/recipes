#!/bin/bash
# stop.sh -- stop the Flare orbit group "glm" on the cluster (both ranks). Packages and caches stay on disk.
set -u
CLUSTER=${CLUSTER:-sparks56}
export PATH=${FLARE_HOME:-$HOME/flare}/bin:$PATH
flare orbit stop glm --cluster $CLUSTER
flare orbit ls --cluster $CLUSTER

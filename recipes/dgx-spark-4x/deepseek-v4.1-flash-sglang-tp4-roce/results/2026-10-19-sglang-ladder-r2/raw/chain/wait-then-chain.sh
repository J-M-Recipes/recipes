#!/bin/bash
until grep -q "PHASE done" ~/bench-r2/B0ctl/battery.phases 2>/dev/null; do sleep 20; done
exec ~/bench-r2/chain-b.sh B1a:B1b:B2:B3

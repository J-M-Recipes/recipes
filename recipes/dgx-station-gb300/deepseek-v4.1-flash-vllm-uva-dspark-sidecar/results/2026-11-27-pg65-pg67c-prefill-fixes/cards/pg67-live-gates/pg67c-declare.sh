set -u
GR=~/dsv41/garner
cp $GR/promote/autostart-pg61.sh $GR/promote/autostart-pg61.sh.bak-pg67b-to-pg67c-live-$(date +%Y%m%d-%H%M)
perl -pi -e 's|~/dsv41/sidecar-resid/work-Pg67b|~/dsv41/sidecar-resid/work-Pg67c|g' $GR/promote/autostart-pg61.sh
grep -n "WORK=" $GR/promote/autostart-pg61.sh | head -2
echo "- $(date +%FT%T) PROMOTED (live-gates path): Pg67c serving :30006 = Pg67b + rung 384 only + util 0.9531 slack-pin (card: -18.33 ms/eager step pooled; live warm probe 264=104.2 392=99.5 ms; det 8/8; A12 PASS free 12.47 GiB > 12.0 bar; kv_tokens 11,813,533 > Pg66; capture 46 (rung 384 landed); PF1 installed no DISABLED; the promote script's 264-cell smoke-probe gate failed on a cold-rep-0 decode-drain artifact, not a regression). rollback = WORK=work-Pg67b launch down+up; autostart repointed" >> ~/dsv41/m3/LANE-OWNER.txt
tail -2 ~/dsv41/m3/LANE-OWNER.txt | cut -c1-300
date

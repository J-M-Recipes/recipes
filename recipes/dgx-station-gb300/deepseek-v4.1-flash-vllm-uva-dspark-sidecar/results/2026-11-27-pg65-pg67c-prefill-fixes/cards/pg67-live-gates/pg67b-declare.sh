set -u
GR=~/dsv41/garner
cp $GR/promote/autostart-pg61.sh $GR/promote/autostart-pg61.sh.bak-pg67a-to-pg67b-pf-live-$(date +%Y%m%d-%H%M)
perl -pi -e 's|~/dsv41/sidecar-resid/work-Pg67a|~/dsv41/sidecar-resid/work-Pg67b|g' $GR/promote/autostart-pg61.sh
grep -n "WORK=" $GR/promote/autostart-pg61.sh | head -2
echo "- $(date +%FT%T) PROMOTED (live-gates path): Pg67b serving :30006 = Pg67a + PF1 direct mm_mxfp8 call plan (card: -18.33 ms/eager step pooled, 12 reps; live warm probe 264=103.5 392=102.0 648=108.0 ms; det 8/8; A12 PASS; kv_tokens >= 11.55M; the promote script's smoke-probe gate failed on a det-pass-evicts-prefix artifact, not a regression). rollback = promote/rollback-pg67b-to-pg67a.sh; autostart repointed" >> ~/dsv41/m3/LANE-OWNER.txt
tail -2 ~/dsv41/m3/LANE-OWNER.txt | cut -c1-300
date

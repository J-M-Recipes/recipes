#!/bin/bash
LOG=~/restore-ckpt.log; D=~/models/DeepSeek-V4.1-Flash-1017
echo "[$(date)] START restore from spark5" >> $LOG
rsync -a -W --delete --partial --inplace -e "ssh -c aes128-gcm@openssh.com -o BatchMode=yes" milo@10.0.0.5:$D/ $D/ >> $LOG 2>&1; echo "[$(date)] rsync rc=$?" >> $LOG
cd $D && sha256sum -c ~/sha256-1017.txt --quiet >> $LOG 2>&1 && echo "[$(date)] CKPT_VERIFIED $(ls model-*.safetensors | wc -l)/48" >> $LOG || echo "[$(date)] FATAL hash mismatch" >> $LOG

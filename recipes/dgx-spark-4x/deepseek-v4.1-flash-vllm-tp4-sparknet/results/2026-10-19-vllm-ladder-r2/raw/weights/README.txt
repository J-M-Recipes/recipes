sha256-1017.txt: sha256 list of the 1017 checkpoint (48 shards + small files). hub-lfs-etags-spark5.txt: the LFS sha256
etags huggingface_hub recorded on spark5 at download (.cache/huggingface/download/*.metadata). The 48 shard hashes in
both files are identical (diffed 2026-10-20). spark5's shards were rsync'd to spark1 on 2026-10-19 and `sha256sum -c`
passed 48/48 there (spark1:~/restore-ckpt.log, CKPT_VERIFIED 48/48). Sparks 5-8 each hold 48 shards totalling
510,296,708,312 bytes (du -sb, 2026-10-20); sparks 6-8 were not re-hashed in place.

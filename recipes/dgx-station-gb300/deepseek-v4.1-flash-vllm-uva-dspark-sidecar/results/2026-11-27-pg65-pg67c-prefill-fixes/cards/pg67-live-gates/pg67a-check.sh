set -u
L=$(ls -t ~/dsv41/garner/promote/promote-pg67a-*.log | head -1)
echo "LOG=$L"
docker logs ds41 > /tmp/pg67a-boot2.log 2>&1
n1=$(grep -ac 'verify OK: gumbel 6eac3e134833 rsu edc8f8f4c425 libw6mk b54168223847560b PEER3_SPINS=1000000000000' "$L")
echo "hash-check (launcher log): $n1"
for x in 0 20 21 39; do
  printf 'selftest-%s: %s\n' "$x" "$(grep -ac "MEGA_PEER FUSE1 layer $x: selftest OK" /tmp/pg67a-boot2.log)"
done
echo "capture-lines: $(grep -ac 'Capturing CUDA graph' /tmp/pg67a-boot2.log)"
echo "kv-tokens: $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' /tmp/pg67a-boot2.log | tail -1)"
echo "a12: $(grep -a 'A12 status=' /dev/null || python3 -c 'import json,re;d={};[d.update(json.loads(m.group(1))) for m in re.finditer(r"A12_INSTALL_MEMORY (\{.*)",open("/tmp/pg67a-boot2.log",errors="ignore").read())];print(d.get("status"),(d.get("terms") or {}).get("kv_bytes"),(d.get("terms") or {}).get("post_capture_free_bytes"))' 2>/dev/null)"

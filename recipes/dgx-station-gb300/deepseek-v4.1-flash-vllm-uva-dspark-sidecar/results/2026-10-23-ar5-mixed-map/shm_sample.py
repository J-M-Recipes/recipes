import mmap, os, struct, time, json, sys
f = open("/dev/shm/vllm_peer_tier2", "rb")
m = mmap.mmap(f.fileno(), 4096, access=mmap.ACCESS_READ)
out = open(sys.argv[1], "a")
end = time.time() + float(sys.argv[2])
while time.time() < end:
    w = struct.unpack("<8q", m[:64])
    out.write(json.dumps({"ts": time.time(), "w": list(w)}) + "\n"); out.flush()
    time.sleep(5)

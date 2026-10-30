#!/usr/bin/env python3
"""Unit tests for sidecar_watchdog_p3 (no docker, no GPU). Run: python3 test_watchdog_p3.py"""
import os, struct, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sidecar_watchdog_p3 as wd


def write_v2(p, pub, done, to):
    with open(p, "wb") as f:
        f.write(struct.pack("<3q", pub, done, to) + b"\0" * 64)


def write_v3(p, done, pub, calls=0, version=2):
    b = bytearray(wd.OFF_REC + 4096)
    struct.pack_into("<Q", b, wd.OFF_VERSION, version)
    struct.pack_into("<Q", b, wd.OFF_DONE, done)
    struct.pack_into("<Q", b, wd.OFF_RTXSTATS, calls)
    struct.pack_into("<Q", b, wd.OFF_REC, (pub << 48) | 0xFFFFFFFFFFFF)
    open(p, "wb").write(bytes(b))


def run():
    d = tempfile.mkdtemp()
    p2, p3 = os.path.join(d, "t2"), os.path.join(d, "t3")
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        fails += 0 if cond else 1

    # 1. idle: published == done everywhere -> never trips
    st = wd.State(2.0); write_v2(p2, 10, 10, 0); write_v3(p3, 0x1234, 0x1234, 5)
    r = [st.step(t, wd.words2(p2), wd.seg3(p3)) for t in range(0, 10)]
    check("idle never trips", all(x is None for x in r))
    # 2. moving traffic: tags change every tick, done lags by one tick -> never trips
    st = wd.State(2.0); ok = True
    for t in range(20):
        write_v3(p3, done=0x1000 + t, pub=0x1000 + t + 1, calls=t); write_v2(p2, 100 + t, 100 + t, 0)
        ok &= st.step(t, wd.words2(p2), wd.seg3(p3)) is None
    check("moving v3/v2 traffic never trips", ok)
    # 3. v3 stall: published != done, frozen -> trips after > stall
    st = wd.State(2.0); write_v3(p3, done=0x2001, pub=0x3001, calls=7); write_v2(p2, 5, 5, 0)
    res = {t: st.step(t, wd.words2(p2), wd.seg3(p3)) for t in (0, 1, 2, 3, 4)}
    check("v3 stall: no trip at <= stall", res[0] is None and res[1] is None and res[2] is None)
    check("v3 stall: trips after > stall", res[4] is not None and "v3 stalled" in res[4])
    # 4. v2 timeout counter increase -> immediate trip
    st = wd.State(2.0); write_v3(p3, 1, 1); write_v2(p2, 5, 5, 0); st.step(0, wd.words2(p2), wd.seg3(p3))
    write_v2(p2, 6, 6, 1)
    r = st.step(1, wd.words2(p2), wd.seg3(p3))
    check("v2 timeout trips immediately", r is not None and "TIMEOUTS" in r)
    # 5. v2 stall
    st = wd.State(2.0); write_v2(p2, 9, 8, 0); write_v3(p3, 1, 1)
    res = [st.step(t, wd.words2(p2), wd.seg3(p3)) for t in range(5)]
    check("v2 stall trips after > stall", res[-1] is not None and "v2 stalled" in res[-1] and res[1] is None)
    # 6. v3 published tag 0 (fresh segment, nothing sent) never counts as a stall
    st = wd.State(2.0); write_v3(p3, done=0, pub=0); write_v2(p2, 0, 0, 0)
    check("fresh segment never trips", all(st.step(t, wd.words2(p2), wd.seg3(p3)) is None for t in range(6)))
    # 7. a stall that clears before the bar resets the timer
    st = wd.State(2.0); write_v2(p2, 0, 0, 0)
    write_v3(p3, done=0x10, pub=0x11); a0 = st.step(0, wd.words2(p2), wd.seg3(p3)); a1 = st.step(1, wd.words2(p2), wd.seg3(p3))
    write_v3(p3, done=0x11, pub=0x11); a2 = st.step(2, wd.words2(p2), wd.seg3(p3))
    write_v3(p3, done=0x11, pub=0x12); a3 = st.step(3, wd.words2(p2), wd.seg3(p3)); a4 = st.step(4, wd.words2(p2), wd.seg3(p3))
    check("cleared stall resets the timer", all(x is None for x in (a0, a1, a2, a3, a4)))
    # 8. spin preconditions
    check("env: PEER3_SPINS=1e12 digits -> arm", wd.spin_preconditions({"PEER3_SPINS": "1000000000000"}) == [])
    check("env: unset (his default) -> refuse", wd.spin_preconditions({}) != [])
    check("env: 20M -> refuse", wd.spin_preconditions({"PEER3_SPINS": "20000000"}) != [])
    check("env: '1e15' (int() would crash boot) -> refuse", wd.spin_preconditions({"PEER3_SPINS": "1e15"}) != [])
    check("env: unreadable -> refuse", wd.spin_preconditions(None) != [])
    print(f"{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if run() else 0)

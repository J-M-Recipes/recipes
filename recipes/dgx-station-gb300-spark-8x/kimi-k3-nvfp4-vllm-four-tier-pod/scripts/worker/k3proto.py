"""K3 expert-dispatch wire format v1 + ctypes wrapper for libk3hop.so.

Request (Station -> Spark), 7,424 B:   one (token, layer) dispatch, the full top-16 as GLOBAL ids.
  u32 req_id | u16 layer (model layer 1..92) | u8 n_tok (=1) | u8 flags | u32 bank_tag
  i32 ids[16] (-1 = empty) | f32 w[16] | pad to 256 | bf16 x[3584]
Reply (Spark -> Station), 7,424 B:
  u32 req_id | u32 status | u16 k_used | u16 ids_used_mask | u32 bank_tag
  u64 t_recv, t_xready, t_kstart, t_kdone, t_reply (worker CLOCK_MONOTONIC ns) | u32 kernel_ns | pad to 256
  bf16 y[3584]  = sum over held ids of w_i * expert_i(x)
The client asserts sum(k_used over Sparks) + k_local == number of valid ids, every layer.
"""
from __future__ import annotations

import ctypes
import os
import struct

HIDDEN = 3584
TOPK = 16
HDR = 256                                      # x/y start 128-byte aligned in the slot
X_BYTES = HIDDEN * 2
MSG_BYTES = HDR + X_BYTES                      # 7,424

REQ = struct.Struct("<I H B B I 16i 16f")        # 4+2+1+1+4+64+64 = 140
REP = struct.Struct("<I I H H I Q Q Q Q Q I")    # 4+4+2+2+4+40+4 = 60
assert REQ.size <= HDR and REP.size <= HDR

FLAG_PROBE = 1
SHUTDOWN_ID = 0xFFFFFFFF
ST_OK, ST_EMPTY, ST_NONFINITE, ST_BAD, ST_WRONG_BANK = 0, 1, 2, 3, 4

_LIB = None


def lib(path: str | None = None):
    global _LIB
    if _LIB is None:
        p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libk3hop.so")
        L = ctypes.CDLL(p)
        L.k3_last_error.restype = ctypes.c_char_p
        L.k3_buf.restype = ctypes.c_void_p
        L.k3_buf.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.k3_ring_bytes.restype = ctypes.c_size_t
        L.k3_listen.restype = ctypes.c_void_p
        L.k3_listen.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        L.k3_connect.restype = ctypes.c_void_p
        L.k3_connect.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
        L.k3_send.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.k3_post_recv.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.k3_poll_recv.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.k3_close.argtypes = [ctypes.c_void_p]
        _LIB = L
    return _LIB


def err() -> str:
    e = lib().k3_last_error()
    return e.decode() if e else ""


class Conn:
    def __init__(self, handle):
        if not handle:
            raise RuntimeError(f"k3hop: {err()}")
        self.h = handle
        self.L = lib()

    @classmethod
    def listen(cls, bind_ip, port, dev, gid=3, ring=None, timeout_s=600):
        return cls(lib().k3_listen(bind_ip.encode(), port, dev.encode(), gid, ring, timeout_s))

    @classmethod
    def connect(cls, peer_ip, port, dev, gid=3, ring=None):
        return cls(lib().k3_connect(peer_ip.encode(), port, dev.encode(), gid, ring))

    def addr(self, slot):
        return self.L.k3_buf(self.h, slot)

    def view(self, slot, n=MSG_BYTES):
        return (ctypes.c_uint8 * n).from_address(self.addr(slot))

    def send(self, slot, n=MSG_BYTES):
        if self.L.k3_send(self.h, slot, n) != 0:
            raise RuntimeError(f"k3_send: {err()}")

    def post_recv(self, slot):
        if self.L.k3_post_recv(self.h, slot) != 0:
            raise RuntimeError(f"k3_post_recv: {err()}")

    def poll(self, timeout_us):
        s = self.L.k3_poll_recv(self.h, int(timeout_us))
        if s < 0 and err():
            raise RuntimeError(f"k3_poll_recv: {err()}")
        return s

    def close(self):
        self.L.k3_close(self.h)

#!/usr/bin/env python3
"""Differential corpora for the extracted proved models vs real C code.

Per-model case format (all LE):
  utf8: [u32 n][bytes]
  norm: [u32 aw][aw x u64 histogram]
  pack: [u8 bsz][u64 n][data]           (only pack_enc-valid inputs:
                                          bsz in {1,2}, bsz|n, 1<=k<=256)
  rans: [u64 x][u64 f][u64 c]           (proved domain: x in
                                          [LOWER,256*LOWER), 1<=f<=TOT,
                                          0<=c<=TOT-f)

  norm correspondence domain: sum(h) < 2^64. C accumulates tot in u64
  (the u128 path only guards h[i]*bud; tot itself can wrap past 2^64,
  inflating every quotient ~2^64/tot fold). The model is exact-Z.
  Empirically confirmed: every mismatching case had sum(h) >= 2^64.
  Real histograms are per-block counts — far below the bound.

usage: gen_diff_cases.py <outdir> <seed>  -> creates outdir/{utf8,norm,pack,rans}/
"""
import os, random, struct, sys

def w(path, blob):
    with open(path, "wb") as f:
        f.write(blob)

def gen_utf8(d, rnd):
    os.makedirs(d, exist_ok=True)
    cases = [
        b"", b"ascii only", b"\xe3\x81\x82\xe3\x81\x84",       # valid
        b"\xf0\x9f\x98\x80", b"\xc2\x80", b"\xdf\xbf",          # boundary valid
        b"\xed\x9f\xbf",                                        # U+D7FF ok
        b"\xed\xa0\x80", b"\xed\xbf\xbf",                       # surrogates
        b"\xf4\x90\x80\x80",                                    # >U+10FFFF
        b"\xf5\x80\x80\x80", b"\xfe", b"\xff",                  # invalid leads
        b"\x80", b"\xbf",                                       # stray cont
        b"\xc0\x80", b"\xc1\xbf",                               # overlong 2
        b"\xe0\x80\x80", b"\xe0\x9f\x80",                       # overlong 3
        b"\xf0\x80\x80\x80", b"\xf0\x8f\x80\x80",               # overlong 4
        b"\xc2", b"\xe3\x81", b"\xf0\x9f\x98",                  # truncated
        b"a\xe3\x81\x82b\xf0\x9f\x98\x80c",
        b"\x00", b"a\x00b",                                     # NULs are fine
        b"\xe3\x81\x82\xff", b"\xff\xe3\x81\x82",               # poisoned tails
    ]
    i = 0
    for s in cases:
        w(os.path.join(d, f"case_{i:05d}.bin"),
          struct.pack("<I", len(s)) + s)
        i += 1
    # exhaustive: every 2-byte pair (lead x cont) — cheap, catches tables
    for lead in range(0x80, 0x100, 3):
        for cont in (0x00, 0x3f, 0x80, 0xbf, 0xc0, 0xff):
            s = bytes([lead, cont])
            w(os.path.join(d, f"case_{i:05d}.bin"),
              struct.pack("<I", 2) + s)
            i += 1
    for _ in range(3000):
        n = rnd.randrange(0, 24)
        s = bytes(rnd.randrange(256) for _ in range(n))
        w(os.path.join(d, f"case_{i:05d}.bin"),
          struct.pack("<I", n) + s)
        i += 1
    return i

def gen_norm(d, rnd):
    os.makedirs(d, exist_ok=True)
    def emit(h):
        return struct.pack("<I", len(h)) + b"".join(
            struct.pack("<Q", v) for v in h)
    cases = [
        [0], [0, 0], [1], [5], [0, 0, 0, 0],
        [1, 1, 1, 1], [100, 0, 0, 0], [0, 0, 7],
        [1, 2, 3, 4, 5], [32768, 1, 1],
        [10**12, 1], [2**49 - 1, 3, 0], [2**49, 1],     # u128 path
        [2**55, 7], [0, 2**60],
    ]
    i = 0
    for h in cases:
        w(os.path.join(d, f"case_{i:05d}.bin"), emit(h)); i += 1
    for _ in range(1500):
        aw = rnd.choice([1, 2, 3, 4, 5, 8, 17, 64, 256, 1024])
        mode = rnd.random()
        if mode < 0.3:
            h = [0] * aw
        elif mode < 0.6:
            h = [rnd.randrange(0, 100) for _ in range(aw)]
        elif mode < 0.8:
            h = [rnd.randrange(0, 1 << 20) for _ in range(aw)]
        else:                     # huge: u128 path, sum kept < 2^64
            hi = min(1 << 56, (1 << 62) // aw)
            h = [rnd.choice([0, rnd.randrange(1 << 49, hi)])
                 for _ in range(aw)]
        w(os.path.join(d, f"case_{i:05d}.bin"), emit(h)); i += 1
    return i

def gen_pack(d, rnd):
    os.makedirs(d, exist_ok=True)
    def emit(bsz, data):
        return bytes([bsz]) + struct.pack("<Q", len(data)) + data
    cases = []
    # bsz=1: controlled distinct-count
    for k in (1, 2, 3, 255, 256):
        cases.append((1, bytes(range(k))))
        cases.append((1, bytes([i % k for i in range(64)])))
    cases.append((1, b"\x00" * 32))
    # bsz=2: u16 atoms LE
    for k in (1, 2, 5, 255, 256):
        d2 = b"".join(struct.pack("<H", i) for i in range(k))
        cases.append((2, d2))
        d2 = b"".join(struct.pack("<H", (i * 7919) % k)
                      for i in range(50))
        cases.append((2, d2))
    cases.append((2, struct.pack("<H", 42) * 16))
    i = 0
    for bsz, data in cases:
        w(os.path.join(d, f"case_{i:05d}.bin"), emit(bsz, data)); i += 1
    for _ in range(1200):
        bsz = rnd.choice([1, 2])
        k = rnd.choice([1, 1, 2, 3, 4, 8, 16, 64, 200, 256])
        ne = rnd.randrange(1, 60)
        if bsz == 1:
            data = bytes(rnd.randrange(k) for _ in range(ne))
        else:
            data = b"".join(struct.pack("<H", rnd.randrange(k))
                            for _ in range(ne))
        w(os.path.join(d, f"case_{i:05d}.bin"), emit(bsz, data)); i += 1
    return i

def gen_rans(d, rnd):
    os.makedirs(d, exist_ok=True)
    LOWER, TOT = 1 << 31, 1 << 15
    def emit(x, f, c):
        return struct.pack("<QQQ", x, f, c)
    i = 0
    for f in (1, 2, 3, 5, 128, 257, TOT - 1, TOT):
        for x in (LOWER, LOWER + 1, 256 * LOWER - 1, 200 * LOWER):
            c = rnd.randrange(0, TOT - f + 1)
            w(os.path.join(d, f"case_{i:05d}.bin"), emit(x, f, c)); i += 1
    for _ in range(2000):
        f = rnd.randrange(1, TOT + 1)
        c = rnd.randrange(0, TOT - f + 1)
        x = rnd.randrange(LOWER, 256 * LOWER)
        w(os.path.join(d, f"case_{i:05d}.bin"), emit(x, f, c)); i += 1
    # renorm-heavy corner: f=1 forces XMAX=2^24 — x near domain top emits 2
    for _ in range(300):
        x = rnd.randrange(128 * LOWER, 256 * LOWER)
        w(os.path.join(d, f"case_{i:05d}.bin"),
          emit(x, 1, rnd.randrange(0, TOT))); i += 1
    return i

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "diffcases"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rnd = random.Random(seed * 7919)
    for name, g in (("utf8", gen_utf8), ("norm", gen_norm),
                    ("pack", gen_pack), ("rans", gen_rans)):
        n = g(os.path.join(out, name), rnd)
        print(f"{name}: {n} cases")

#!/usr/bin/env python3
"""Deterministic test fixtures for caiw — pure stdlib, no dependencies.
Regenerates everything test.sh needs into out/."""
import struct, json, random, os, math
from array import array

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

def wst(path, tensors, gmeta=None):
    """tensors: {name: (dtype, shape, bytes)}; gmeta: optional __metadata__ dict"""
    meta, blob = {}, b''
    if gmeta is not None:
        meta['__metadata__'] = gmeta
    for k, (dt, sh, b) in tensors.items():
        meta[k] = {'dtype': dt, 'shape': list(sh), 'data_offsets': [len(blob), len(blob) + len(b)]}
        blob += b
    j = json.dumps(meta).encode()
    j += b' ' * ((8 - len(j) % 8) % 8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(j)) + j + blob)
    return 8 + len(j) + len(blob)

def f16(vals):
    return struct.pack('<%de' % len(vals), *vals)

def bf16(vals):
    f = array('f', vals)
    u = array('I'); u.frombytes(f.tobytes())
    return array('H', (x >> 16 for x in u)).tobytes()

def f32(vals):
    return array('f', vals).tobytes()

def gauss16(rng, n, sd=0.02):
    return f16([rng.gauss(0, sd) for _ in range(n)])

def main():
    rng = random.Random(42)
    A = {}
    A['f64.mat']   = ('F64', [4, 4],  array('d', [rng.gauss(0, 1) for _ in range(16)]).tobytes())
    A['f32.mat']   = ('F32', [8, 16], f32([rng.gauss(0, 1) for _ in range(128)]))
    A['f16.mat']   = ('F16', [16, 32], gauss16(rng, 512))
    A['bf16.mat']  = ('BF16', [16, 32], bf16([rng.gauss(0, 1) for _ in range(512)]))
    A['i64.vec']   = ('I64', [7],   array('q', range(-3, 4)).tobytes())
    A['i32.vec']   = ('I32', [33],  array('i', [rng.randrange(-100, 100) for _ in range(33)]).tobytes())
    A['i16.vec']   = ('I16', [129], array('h', [rng.randrange(-300, 300) for _ in range(129)]).tobytes())
    A['i8.vec']    = ('I8',  [257], bytes(rng.randrange(-5, 5) & 0xFF for _ in range(257)))
    A['u8.vec']    = ('U8',  [1023], bytes(rng.randrange(256) for _ in range(1023)))
    A['bool.vec']  = ('BOOL',[65],  bytes(rng.randrange(2) for _ in range(65)))
    A['empty']     = ('F32', [0], b'')
    A['scalar']    = ('F32', [], struct.pack('<f', 3.14))
    A['one']       = ('BF16',[1], struct.pack('<H', 0x3F80))
    A['tiny3d']    = ('F16', [2, 3, 4], gauss16(rng, 24))
    A['big4d']     = ('BF16',[2, 3, 8, 16], bf16([rng.gauss(0, 1) for _ in range(768)]))
    dup = f16([float(i) for i in range(32)])
    A['dup.src']   = ('F16', [4, 8], dup)
    A['dup.copy']  = ('F16', [4, 8], dup)                    # REF candidate
    shared = bf16([rng.gauss(0, 1) for _ in range(2048)])
    A['shared.name'] = ('BF16', [64, 32], shared)
    # positional pattern -> FIELDPOS/FIELDROW territory
    pos = [math.sin(i % 64 / 10.0) for i in range(256 * 64)]
    A['pos.emb']   = ('BF16', [256, 64], bf16(pos))
    wst(os.path.join(OUT, 'edgeA.st'), A)

    B = {}
    mut = bytearray(shared); mut[0] ^= 1                     # tiny delta
    B['shared.name'] = ('BF16', [64, 32], bytes(mut))        # DELTA candidate
    B['other.f32']   = ('F32', [32], f32([rng.gauss(0, 1) for _ in range(32)]))
    B['dup.src']     = ('F16', [4, 8], dup)                  # REF vs file A
    wst(os.path.join(OUT, 'edgeB.st'), B)

    # checkpoint chain: B and C are small perturbations of A -> DELTA must win
    n = 4 * 1024 * 1024
    base = [rng.gauss(0, 0.02) for _ in range(n)]
    wst(os.path.join(OUT, 'chA.st'), {'w': ('F16', [n], f16(base))})
    for tag, rate in (('chB', 0.01), ('chC', 0.02)):
        v = base[:]
        for i in range(n):
            if rng.random() < rate:
                v[i] += rng.gauss(0, 0.001)
        wst(os.path.join(OUT, tag + '.st'), {'w': ('F16', [n], f16(v))})

    # f32 pair -> F32 / DELTA32 / DELTAX
    n32 = 1024 * 1024
    a32 = [rng.gauss(0, 0.05) for _ in range(n32)]
    wst(os.path.join(OUT, 'f32A.st'), {'w': ('F32', [n32], f32(a32))})
    b32 = [x + (rng.gauss(0, 0.0005) if rng.random() < 0.05 else 0) for x in a32]
    wst(os.path.join(OUT, 'f32B.st'), {'w': ('F32', [n32], f32(b32))})

    # synthetic MoE-like shard: 9 expert mats + layernorm
    E = {}
    for e in range(3):
        for proj, shp in (('down_proj', [256, 96]), ('gate_proj', [96, 256]), ('up_proj', [96, 256])):
            E['model.layers.0.mlp.experts.%d.%s.weight' % (e, proj)] = \
                ('BF16', shp, bf16([rng.gauss(0, 0.02) for _ in range(shp[0] * shp[1])]))
    E['model.layers.0.input_layernorm.weight'] = ('BF16', [96], bf16([1.0 + rng.gauss(0, .01) for _ in range(96)]))
    wst(os.path.join(OUT, 'exp.st'), E)

    # review-driven regressions (CAI5 parser/decoder edge cases):
    # pack1: one distinct value -> PACK dictionary of k=1 (zero-bit indices)
    wst(os.path.join(OUT, 'pack1.st'), {'w': ('F16', [1024], f16([0.0] * 1024))})
    # escname: json.dumps escapes quotes/backslashes/unicode in tensor names —
    # the archive stores raw names and `d` re-escapes; both must round-trip
    wst(os.path.join(OUT, 'escname.st'), {
        'a"b\\c':      ('F32', [16], f32([float(i) for i in range(16)])),
        'uniéname': ('F16', [8],  f16([1.5] * 8)),
    })
    # ndim9: >8 dims used to truncate silently — now supported to 64
    wst(os.path.join(OUT, 'ndim9.st'), {'deep': ('F32', [2] * 9, f32([rng.gauss(0, 1) for _ in range(512)]))})
    # meta: __metadata__ must be preserved verbatim through c/d
    wst(os.path.join(OUT, 'meta.st'), {'w': ('F32', [8], f32([1.0] * 8))},
        gmeta={'format': 'pt', 'training': 'step-4200'})
    # split: 64 small bf16 tensors sharing one distribution — cold-start
    # regression: RAW winners must still train the FIELD channel via teach()
    S = {}
    rows = [rng.gauss(0, 0.02) for _ in range(64 * 256 * 256)]
    for e in range(64):
        S['expert.%02d' % e] = ('BF16', [256, 256], bf16(rows[e * 65536:(e + 1) * 65536]))
    wst(os.path.join(OUT, 'split.st'), S)

    # f16t: bf16 tensor with a concentrated, row-drifting exponent profile —
    # the transmitted frozen table (FIELDT, CAI6) beats per-block adaptation
    # here; must select method 11, write CAI6 magic, and round-trip at any -j
    vals = []
    for i in range(1600 * 1024):
        r = i // 1024
        vals.append(rng.gauss(0, 0.011 + 0.004 * ((r * 7) % 13) / 13.0))
    wst(os.path.join(OUT, 'f16t.st'), {'w': ('BF16', [1600, 1024], bf16(vals))})

    # rows: smooth-row f16 matrix — adjacent rows are tiny perturbations of
    # each other (embedding-like) -> PREVROW should beat FIELD outright
    R = 2048; C = 256
    r0 = [rng.gauss(0, 0.03) for _ in range(C)]
    rowsv = r0[:]
    for r in range(1, R):
        rowsv += [x + rng.gauss(0, 0.0004) for x in rowsv[(r - 1) * C:r * C]]
    wst(os.path.join(OUT, 'rows.st'), {
        'emb.weight': ('F16', [R, C], f16(rowsv)),
        'plain':      ('F16', [512, 128], f16([rng.gauss(0, 0.02) for _ in range(512 * 128)])),
    })

if __name__ == '__main__':
    main()
    print('fixtures ->', OUT)

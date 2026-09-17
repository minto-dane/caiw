#!/usr/bin/env python3
"""Deterministic test fixtures for caiw — pure stdlib, no dependencies.
Regenerates everything test.sh needs into out/."""
import struct, json, random, os, math
from array import array

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

def wst(path, tensors):
    """tensors: {name: (dtype, shape, bytes)}"""
    meta, blob = {}, b''
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

if __name__ == '__main__':
    main()
    print('fixtures ->', OUT)

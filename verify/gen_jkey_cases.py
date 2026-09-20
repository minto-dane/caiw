#!/usr/bin/env python3
"""Generate differential-test cases for jkey (C vs extracted Coq model).

Each case file: [u32 olen][u32 rlen][u32 klen][obj][rest][key] (LE u32).
The C driver NUL-terminates obj+rest; the model sees (obj, rest) lists.
Keys must be NUL-free (C side compares via strcmp).
"""
import os, random, struct, sys

def emit(path, obj, rest, key):
    with open(path, "wb") as f:
        f.write(struct.pack("<III", len(obj), len(rest), len(key)))
        f.write(obj); f.write(rest); f.write(key)

def gen(outdir, n_seed):
    os.makedirs(outdir, exist_ok=True)
    cases = []
    K = [b"dtype", b"shape", b"data_offsets", b"a", b"x", b"", b"ky",
         b"\xe3\x81\x82", b"A"]
    fixed = []
    # --- well-formed objects, key present/absent ---
    fixed += [
        (b'{"dtype":"F16","shape":[1,2],"data_offsets":[0,4]}', b"dtype"),
        (b'{"shape":[1,2],"dtype":"F16"}', b"dtype"),
        (b'{"shape":[1,2],"dtype":"F16"}', b"shape"),
        (b'{"shape":[1,2],"dtype":"F16"}', b"missing"),
        (b'{"a":1}', b"a"), (b'{"a":1}', b"b"),
        (b'{}', b"a"), (b'{"a":1}', b""),
        (b'{  "dtype"  :  "F16"  }', b"dtype"),
        (b'{\t"dtype"\r\n:\t\n "F16"}', b"dtype"),
        # values: nested object / array / scalar / string-with-tricks
        (b'{"a":{"nested":[1,{"k":2}],"s":"x}"},"dtype":1}', b"dtype"),
        (b'{"a":{"nested":[1,{"k":2}],"s":"x}"},"dtype":1}', b"a"),
        (b'{"a":[{"x":1},[2,3]],"dtype":4}', b"dtype"),
        (b'{"a":"str,with}braces[and]quotes\\\"","dtype":5}', b"dtype"),
        (b'{"a":true,"b":false,"c":null,"dtype":6}', b"dtype"),
        (b'{"a":-12.5e+10,"dtype":7}', b"dtype"),
        (b'{"a":"", "dtype":8}', b"dtype"),
        # escaped key names
        (b'{"d\\u0074ype":"X","z":1}', b"dtype"),      # \u0074 = 't'
        (b'{"a\\"b":1,"dtype":2}', b'a"b'),
        (b'{"x\\\\y":1,"dtype":2}', b"x\\y"),
        (b'{"\\uD83D\\uDE00":1,"dtype":2}',           # surrogate pair
            "\U0001F600".encode()),
        (b'{"\\uD800x":1,"dtype":2}', b"\xed\xa0\x80x"),  # lone hi
        (b'{"\\uDC00":1,"dtype":2}', b"\xed\xb0\x80"),    # lone lo
        (b'{"\\uD800\\u0041":1,"dtype":2}', b"\xed\xa0\x80A"),  # hi+\u0041
        (b'{"\\u0041":1,"dtype":2}', b"A"),
        # missing colon / truncated pieces
        (b'{"a" 1,"dtype":2}', b"dtype"),
        (b'{"a"', b"a"), (b'{"a":', b"a"), (b'{"a": ', b"a"),
        (b'{"a":"unterminated', b"a"),
        (b'{"a":{"x"', b"a"), (b'{"a":[1,2', b"a"),
        (b'{"a":"x\\', b"a"), (b'{"a":"x\\u004', b"a"),
        (b'{"a"::1}', b"a"),
        (b'{"a" ,"b":1}', b"b"),
        (b'{"a";"b":1}', b"b"),
        (b'{"a"\"b":1}', b"b"),
        # NUL bytes inside the object
        (b'{"a\x00":1,"dtype":2}', b"dtype"),
        (b'{"a":\x001,"dtype":2}', b"dtype"),
        (b'{"a\x00b":1}', b"a"),
        (b'\x00{"a":1}', b"a"),
        # comma-colon ordering edge
        (b'{"a",:1,"dtype":2}', b"dtype"),
        (b'{"a":1,"a":2}', b"a"),   # duplicate key -> first wins
        (b'{"a":1,"a":2}', b"x"),
        # scalar value ends exactly at '}' (no comma)
        (b'{"a":1,"dtype":9}', b"dtype"),
        (b'{"a":123}', b"a"),
        (b'{"a":123}', b"dtype"),
        (b'{"a":"v","b":"w","dtype":"q"}', b"dtype"),
    ]
    i = 0
    for obj, key in fixed:
        emit(os.path.join(outdir, f"case_{i:05d}.bin"), obj, b"", key)
        i += 1

    rnd = random.Random(n_seed)
    alpha = (b'{}[]",:0123456789truefalsn.eE+- \t\n\r\\u' +
             b'abdypseh' + bytes([0, 1, 0x7f, 0x80, 0xc0, 0xff]))
    # random JSON-ish fuzz: arbitrary obj bytes + rest tail + key
    for _ in range(1200):
        olen = rnd.randrange(0, 48)
        rlen = rnd.randrange(0, 24)
        obj = bytes(rnd.choice(alpha) for _ in range(olen))
        rest = bytes(rnd.choice(alpha) for _ in range(rlen))
        key = rnd.choice(K)
        emit(os.path.join(outdir, f"case_{i:05d}.bin"), obj, rest, key)
        i += 1
    # structured fuzz: build a random object then mutate/cut it
    keys = [b'"a"', b'"dtype"', b'"shape"', b'"k\\\\"', b'"q\\"r"',
            b'"\\u0041"', b'"\\uD83D\\uDE00"', b'"\\uD800"']
    vals = [b'1', b'-2.5e3', b'true', b'null', b'"s"', b'"s\\"t"',
            b'{"n":[1,{"m":2}]}', b'[3,{"x":"}"},4]', b'"unterm',
            b'{"a":', b'[1,2', b'', b'"\\u0041"']
    for _ in range(1200):
        n = rnd.randrange(0, 5)
        parts = [keys[rnd.randrange(len(keys))] + b':' +
                 vals[rnd.randrange(len(vals))] for _ in range(n)]
        obj = b'{' + b','.join(parts)
        if rnd.random() < 0.7:
            obj += b'}'
        if rnd.random() < 0.4 and len(obj) > 1:    # random truncation
            obj = obj[:rnd.randrange(1, len(obj))]
        rlen = rnd.randrange(0, 16)
        rest = bytes(rnd.choice(alpha) for _ in range(rlen))
        key = rnd.choice(K)
        emit(os.path.join(outdir, f"case_{i:05d}.bin"), obj, rest, key)
        i += 1
    # split-boundary stress: take valid objects, split obj/rest mid-token
    protos = [
        b'{"dtype":"F16","shape":[1,{"x":[2]}],"z":"w"}',
        b'{"a":"v\\"w","dtype":{"n":1}}',
        b'{"\\u0041BC":1,"dtype":2}',
        b'{"a":{"b":{"c":{"d":[1]}}},"dtype":3}',
    ]
    for obj0 in protos:
        for cut in range(len(obj0) + 1):
            obj, rest = obj0[:cut], obj0[cut:]
            for key in (b"dtype", b"a", b"z"):
                emit(os.path.join(outdir, f"case_{i:05d}.bin"),
                     obj, rest, key)
                i += 1
    # cap boundary: decoded key right at the oend-obj+1 limit
    for klen in range(0, 12):
        kn = b"k" * klen
        obj = b'{"' + kn + b'":1}'
        emit(os.path.join(outdir, f"case_{i:05d}.bin"), obj, b"", kn)
        i += 1
        # key longer than span allows to decode
        emit(os.path.join(outdir, f"case_{i:05d}.bin"), obj, b"", kn + b"Q")
        i += 1
    print(f"{i} cases in {outdir}")

if __name__ == "__main__":
    gen(sys.argv[1] if len(sys.argv) > 1 else "jkeycases",
        int(sys.argv[2]) if len(sys.argv) > 2 else 1)

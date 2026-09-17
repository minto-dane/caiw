#!/bin/bash
# regression: generate fixtures if missing, round-trip -j1/-j4,
# DELTAX external-ref path, then tests/corpus/ must die cleanly (never crash).
BIN=${1:-./caiw}
cd "$(dirname "$0")"
set -u; fail=0
[ -f out/edgeA.st ] || python3 gen_fixtures.py
for t in "edgeA.st edgeB.st" "chA.st chB.st chC.st" "f32A.st f32B.st" "exp.st" \
         "pack1.st" "escname.st" "ndim9.st" "meta.st" "split.st" "rows.st"; do
    for j in 1 4; do
        ok=1; args=""
        set -- $t; for a in "$@"; do args="$args out/$a"; done
        "$BIN" c /tmp/tst.caiw $args -j$j >/dev/null 2>&1 || { echo "ENC FAIL: $t -j$j"; fail=1; ok=0; }
        [ $ok = 1 ] && { "$BIN" v /tmp/tst.caiw $args -j$j 2>&1 | grep -q "0 bad" || { echo "VERIFY FAIL: $t -j$j"; fail=1; }; }
    done
done
"$BIN" c /tmp/tst.caiw --ref out/f32A.st out/f32B.st -j4 >/dev/null 2>&1 &&
"$BIN" v /tmp/tst.caiw --ref out/f32A.st out/f32B.st -j4 2>&1 | grep -q "0 bad" || { echo "DELTAX FAIL"; fail=1; }
# cold-start: split tensors must compress near single-tensor ratio (teach)
sz=$("$BIN" c /tmp/split.caiw out/split.st -j1 2>&1 | grep -o 'ratio=[0-9.]*' | cut -d= -f2)
awk "BEGIN{exit !($sz < 0.75)}" || { echo "COLD-START FAIL: split ratio $sz"; fail=1; }
# PREVROW: smooth-row embedding must trigger the PRW method and round-trip
"$BIN" c /tmp/rows.caiw out/rows.st -j1 2>&1 | grep -q "PRW" || { echo "PRW NOT SELECTED"; fail=1; }
rm -rf /tmp/rowsd && "$BIN" d /tmp/rows.caiw /tmp/rowsd -j4 >/dev/null 2>&1 || { echo "PRW DECODE FAIL"; fail=1; }
python3 - <<'PYEOF'
import struct, json
def data_region(p):
    d = open(p, 'rb').read(); hl = struct.unpack('<Q', d[:8])[0]
    return d[8 + hl:]
assert data_region('out/rows.st') == data_region('/tmp/rowsd/rows.st'), "PRW data mismatch"
PYEOF
[ $? -eq 0 ] || { echo "PRW DATA MISMATCH"; fail=1; }
# v completeness: dropping a source file must fail (MISS-SRC / FILECOUNT)
"$BIN" c /tmp/tst2.caiw out/pack1.st out/meta.st -j1 >/dev/null 2>&1
"$BIN" v /tmp/tst2.caiw out/pack1.st 2>&1 | grep -q "0 bad" && { echo "V-COMPLETENESS FAIL"; fail=1; }
# metadata must survive a d round-trip
rm -rf /tmp/tstd && "$BIN" d /tmp/tst2.caiw /tmp/tstd -j1 >/dev/null 2>&1 &&
grep -q '__metadata__' /tmp/tstd/meta.st || { echo "META FAIL"; fail=1; }
# hardening: malformed inputs must die(1), never pass silently
python3 - <<'PYEOF'
import struct
# dup tensor name inside one file
hdr = b'{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"a":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}'
open('/tmp/h_dup.st','wb').write(struct.pack('<Q', len(hdr)) + hdr + b'ABCD')
# malformed scalar __metadata__
hdr = b'{"__metadata__":12x,"t":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}'
open('/tmp/h_meta.st','wb').write(struct.pack('<Q', len(hdr)) + hdr + b'ABCD')
# archive with NUL inside filename field
a = bytearray(open('/tmp/tst2.caiw','rb').read())
nl = struct.unpack('<I', a[8:12])[0]
a[12] = 0
open('/tmp/h_nul.caiw','wb').write(a)
# archive with trailing garbage
open('/tmp/h_trail.caiw','wb').write(open('/tmp/tst2.caiw','rb').read() + b'GARBAGE')
def w32(v): return struct.pack('<I', v)
def w64(v): return struct.pack('<q', v)
# file table ["a","a.caiwtmp"]: tmp name would collide with a final name
a = b'CAI5' + w32(2)
for nm in (b'a', b'a.caiwtmp'): a += w32(len(nm)) + nm + w32(0)
a += w32(0)
open('/tmp/h_col.caiw','wb').write(a)
# FIELD method on I16 dtype: encoder never emits this combo -> must die
a = b'CAI5' + w32(1) + w32(1) + b'x' + w32(0) + w32(1)
a += struct.pack('<H',1)+b't'+struct.pack('<H',3)+b'I16'+b'\x01'+w64(32)
a += struct.pack('<H',0)+b'\x03'+w64(64)+w64(0)+w32(0)
open('/tmp/h_dtype.caiw','wb').write(a)
# duplicate filenames in file table
a = b'CAI5' + w32(2)
for nm in (b'same.st', b'same.st'): a += w32(len(nm)) + nm + w32(0)
a += w32(0)
open('/tmp/h_dupf.caiw','wb').write(a)
# member whose output path is the archive itself
a = b'CAI5' + w32(1) + w32(11) + b'h_self.caiw' + w32(0) + w32(0)
open('/tmp/h_self.caiw','wb').write(a)
PYEOF
"$BIN" c /tmp/hz.caiw /tmp/h_dup.st >/dev/null 2>&1 && { echo "DUPNAME ACCEPTED"; fail=1; }
"$BIN" c /tmp/hz.caiw /tmp/h_meta.st >/dev/null 2>&1 && { echo "BADMETA ACCEPTED"; fail=1; }
"$BIN" d /tmp/h_nul.caiw /tmp/hzout >/dev/null 2>&1 && { echo "NULNAME ACCEPTED"; fail=1; }
"$BIN" v /tmp/h_trail.caiw out/pack1.st out/meta.st >/dev/null 2>&1 && { echo "TRAILING ACCEPTED"; fail=1; }
"$BIN" c /tmp/hz.caiw -j 1x out/pack1.st >/dev/null 2>&1 && { echo "BAD-J ACCEPTED"; fail=1; }
"$BIN" d /tmp/h_col.caiw /tmp/hzout >/dev/null 2>&1 && { echo "TMPCOLLIDE ACCEPTED"; fail=1; }
"$BIN" d /tmp/h_dtype.caiw /tmp/hzout >/dev/null 2>&1 && { echo "BADDTYPE ACCEPTED"; fail=1; }
"$BIN" v /tmp/h_dtype.caiw out/pack1.st >/dev/null 2>&1 && { echo "BADDTYPE-V ACCEPTED"; fail=1; }
"$BIN" d /tmp/h_dupf.caiw /tmp/hzout >/dev/null 2>&1 && { echo "DUPFNAME ACCEPTED"; fail=1; }
"$BIN" d /tmp/h_self.caiw /tmp >/dev/null 2>&1 && { echo "SELFD ACCEPTED"; fail=1; }
[ -f /tmp/h_self.caiw ] || { echo "ARCHIVE CLOBBERED"; fail=1; }
cp out/pack1.st /tmp/tcself.st
"$BIN" c /tmp/tcself.st /tmp/tcself.st >/dev/null 2>&1 && { echo "SELF-C ACCEPTED"; fail=1; }
"$BIN" c /tmp/tcself.st --ref out/f32A.st out/f32B.st >/dev/null 2>&1 || { echo "LEGIT-REF FAILED"; fail=1; }
cp out/f32A.st /tmp/tcref.st
"$BIN" c /tmp/tcref.st --ref /tmp/tcref.st out/f32B.st >/dev/null 2>&1 && { echo "SELF-REF ACCEPTED"; fail=1; }
cp out/pack1.st /tmp/tcA; cp out/pack1.st /tmp/tcA.caiwtmp
"$BIN" c /tmp/tcc.caiw /tmp/tcA /tmp/tcA.caiwtmp >/dev/null 2>&1 && { echo "ENCTMP ACCEPTED"; fail=1; }
for f in tests/corpus/*; do
    [ -e "$f" ] || continue
    timeout 10 "$BIN" c /tmp/tz.caiw "$f" -j2 >/dev/null 2>&1; rc=$?
    [ $rc -ge 128 ] && { echo "CRASH on $f rc=$rc"; fail=1; }
done
[ $fail -eq 0 ] && echo "ALL PASS" || echo "FAILURES"
exit $fail

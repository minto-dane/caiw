#!/bin/bash
# regression: generate fixtures if missing, round-trip -j1/-j4,
# DELTAX external-ref path, then tests/corpus/ must die cleanly (never crash).
BIN=${1:-./caiw}
export BIN
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
# d-mode with --ref: DELTAX records must resolve through the ref files
rm -rf /tmp/rfxout && "$BIN" d /tmp/tst.caiw /tmp/rfxout --ref out/f32A.st -j4 >/dev/null 2>&1 || { echo "DELTAX-D FAIL"; fail=1; }
python3 - <<'PYEOF' || { echo "DELTAX-D DIFF"; fail=1; }
import struct, sys
a = open('/tmp/rfxout/f32B.st','rb').read(); b = open('out/f32B.st','rb').read()
la, lb = struct.unpack('<Q', a[:8])[0], struct.unpack('<Q', b[:8])[0]
sys.exit(0 if a[8+la:] == b[8+lb:] else 1)
PYEOF
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
# a single-tensor archive whose record payload has 4 garbage bytes appended:
# plen is bumped so the record walk stays in-frame — the codec must notice
# the stream does not consume the whole payload
import subprocess, os
subprocess.run([os.environ.get('BIN', './caiw'),'c','/tmp/h_pad.caiw','out/pack1.st','-j1'],
               check=True, capture_output=True)
d = bytearray(open('/tmp/h_pad.caiw','rb').read())
p = 4
nf = struct.unpack('<I', d[p:p+4])[0]; p += 4
for _ in range(nf):
    nl = struct.unpack('<I', d[p:p+4])[0]; p += 4 + nl
    ml = struct.unpack('<I', d[p:p+4])[0]; p += 4 + ml
p += 4  # NT == 1
p += 2 + struct.unpack('<H', d[p:p+2])[0]   # name
p += 2 + struct.unpack('<H', d[p:p+2])[0]   # dtype
p += 1 + d[p] * 8                          # nd + shape
p += 2                                     # file
m = d[p]; p += 1                           # method
p += 8                                     # len
if (m & 0x3f) in (2, 5, 9): p += 4         # REF/DELTA/DELTAX ref field
plen_at = p
plen = struct.unpack('<Q', d[p:p+8])[0]
p += 8 + 4                                 # plen + crc
p += plen
d2 = bytearray(d[:plen_at]) + struct.pack('<Q', plen + 4) + d[plen_at+8:p] + b'JUNK' + d[p:]
open('/tmp/h_pad.caiw','wb').write(d2)
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
"$BIN" v /tmp/h_pad.caiw out/pack1.st >/dev/null 2>&1 && { echo "PADPAY ACCEPTED"; fail=1; }
"$BIN" d /tmp/h_pad.caiw /tmp/hzout >/dev/null 2>&1 && { echo "PADPAY-D ACCEPTED"; fail=1; }
cp out/pack1.st /tmp/tcself.st
"$BIN" c /tmp/tcself.st /tmp/tcself.st >/dev/null 2>&1 && { echo "SELF-C ACCEPTED"; fail=1; }
# symlink alias: output path resolves to the input — must be caught too
cp out/pack1.st /tmp/tcreal.st && ln -sf /tmp/tcreal.st /tmp/tcalias.st
"$BIN" c /tmp/tcalias.st /tmp/tcreal.st >/dev/null 2>&1 && { echo "ALIAS-C ACCEPTED"; fail=1; }
[ -s /tmp/tcreal.st ] || { echo "ALIAS CLOBBERED SOURCE"; fail=1; }
# d-side: member output resolving to the archive via a symlinked outdir
mkdir -p /tmp/tcreald
python3 - <<'PYEOF'
import struct
def w32(v): return struct.pack('<I', v)
# member name "x.caiw"; archive itself placed at reald/x.caiw
a = b'CAI5' + w32(1) + w32(6) + b'x.caiw' + w32(0) + w32(0)
open('/tmp/tcreald/x.caiw','wb').write(a)
PYEOF
ln -sfn /tmp/tcreald /tmp/tcdlink
"$BIN" d /tmp/tcreald/x.caiw /tmp/tcdlink >/dev/null 2>&1 && { echo "ALIAS-D ACCEPTED"; fail=1; }
[ -f /tmp/tcreald/x.caiw ] || { echo "ALIAS-D CLOBBERED"; fail=1; }
"$BIN" c /tmp/tcself.st --ref out/f32A.st out/f32B.st >/dev/null 2>&1 || { echo "LEGIT-REF FAILED"; fail=1; }
cp out/f32A.st /tmp/tcref.st
"$BIN" c /tmp/tcref.st --ref /tmp/tcref.st out/f32B.st >/dev/null 2>&1 && { echo "SELF-REF ACCEPTED"; fail=1; }
cp out/pack1.st /tmp/tcA; cp out/pack1.st /tmp/tcA.caiwtmp
"$BIN" c /tmp/tcc.caiw /tmp/tcA /tmp/tcA.caiwtmp >/dev/null 2>&1 && { echo "ENCTMP ACCEPTED"; fail=1; }
# input named exactly <out>.caiwtmp — the tmp fopen must not truncate it
cp out/pack1.st /tmp/tself.caiw.caiwtmp
"$BIN" c /tmp/tself.caiw /tmp/tself.caiw.caiwtmp >/dev/null 2>&1 && { echo "TMPSELF ACCEPTED"; fail=1; }
[ -s /tmp/tself.caiw.caiwtmp ] || { echo "TMPSELF CLOBBERED"; fail=1; }
# planted symlink at the tmp path — O_NOFOLLOW must refuse it
rm -f /tmp/symt.caiw.caiwtmp && echo precious > /tmp/precious.txt
ln -sf /tmp/precious.txt /tmp/symt.caiw.caiwtmp
"$BIN" c /tmp/symt.caiw out/pack1.st >/dev/null 2>&1 && { echo "SYMTMP ACCEPTED"; fail=1; }
[ "$(cat /tmp/precious.txt)" = precious ] || { echo "SYMTMP CLOBBERED"; fail=1; }
# d-side member output landing on a --ref source must be refused
cp out/f32A.st /tmp/rfout/base.st 2>/dev/null || { mkdir -p /tmp/rfout; cp out/f32A.st /tmp/rfout/base.st; }
"$BIN" c /tmp/rf.caiw --ref /tmp/rfout/base.st out/f32A.st >/dev/null 2>&1
cp out/f32A.st /tmp/base.st && "$BIN" c /tmp/rf2.caiw --ref /tmp/rfout/base.st /tmp/base.st >/dev/null 2>&1
"$BIN" d /tmp/rf2.caiw /tmp/rfout --ref /tmp/rfout/base.st >/dev/null 2>&1 && { echo "D-REF ACCEPTED"; fail=1; }
[ -s /tmp/rfout/base.st ] || { echo "D-REF CLOBBERED"; fail=1; }
# strict-JSON metadata: brace-balanced but ungrammatical values must die
python3 - <<'PYEOF'
import struct
def w32(v): return struct.pack('<I', v)
for tag, meta in (('nogram', b'{"a" 1}'), ('badesc', b'{"k":"\\q"}'),
                  ('deep', b'[' * 200 + b']' * 200), ('junk', b'{"a":1}x'),
                  ('escnul', b'"\\'), ('tinyf', b'f'), ('tinysp', b' ')):
    a = b'CAI5' + w32(1) + w32(1) + b'x' + w32(len(meta)) + meta + w32(0)
    open('/tmp/mg_%s.caiw' % tag, 'wb').write(a)
# non-utf8 tensor name in a record (raw 0x80 byte) — regenerated JSON
# would be invalid UTF-8; the walk must reject it
a = b'CAI5' + w32(1) + w32(1) + b'x' + w32(0) + w32(1)
a += struct.pack('<H', 3) + b'A\x80B'
open('/tmp/mg_utf.caiw', 'wb').write(a)
# non-utf8 tensor name in a source safetensors header
h = b'{"a\x80b":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}'
open('/tmp/mg_utf.st', 'wb').write(struct.pack('<Q', len(h)) + h + b'AB')
PYEOF
for t in nogram badesc deep junk escnul tinyf tinysp; do
    "$BIN" d /tmp/mg_$t.caiw /tmp/hzout >/dev/null 2>&1 && { echo "META-GRAMMAR-$t ACCEPTED"; fail=1; }
    "$BIN" d /tmp/mg_$t.caiw /tmp/hzout >/dev/null 2>&1; [ $? -ge 128 ] && { echo "META-GRAMMAR-$t CRASH"; fail=1; }
done
# tmp-path self-destruction: archive IS <outdir>/<member>.caiwtmp —
# must die, not O_TRUNC its own mmap'd input mid-decode
mkdir -p /tmp/tatk
python3 - <<'PYEOF'
import struct
def w32(v): return struct.pack('<I', v)
a = b'CAI5' + w32(1) + w32(1) + b'x' + w32(0) + w32(0)
open('/tmp/tatk/x.caiwtmp', 'wb').write(a)
PYEOF
"$BIN" d /tmp/tatk/x.caiwtmp /tmp/tatk >/dev/null 2>&1 && { echo "TMPSELF-D ACCEPTED"; fail=1; }
[ -s /tmp/tatk/x.caiwtmp ] || { echo "TMPSELF-D CLOBBERED ARCHIVE"; fail=1; }
"$BIN" d /tmp/mg_utf.caiw /tmp/hzout >/dev/null 2>&1 && { echo "UTF8-NAME ACCEPTED"; fail=1; }
"$BIN" c /tmp/hz.caiw /tmp/mg_utf.st >/dev/null 2>&1 && { echo "UTF8-SRC ACCEPTED"; fail=1; }
# unterminated __metadata__ string at a page boundary — must die, not run
# off the mmap end (the ck_meta buffer is the terminated copy now)
python3 - <<'PYEOF'
import struct
meta = b'"' + b'A' * 4000          # '"' then no closer -> unbounded jstr_skip on raw buf
rec = struct.pack('<I', 1) + b'a' + struct.pack('<I', len(meta)) + meta
body = b'CAI5' + struct.pack('<I', 1) + rec + struct.pack('<I', 0)
body += b'B' * (4096 - len(body) % 4096)   # page-align fsz: scan hits unmapped page
open('/tmp/metabad.caiw', 'wb').write(body)
PYEOF
"$BIN" d /tmp/metabad.caiw /tmp/hzout >/dev/null 2>&1; [ $? -ge 128 ] && { echo "META-SCAN CRASH"; fail=1; }
"$BIN" v /tmp/metabad.caiw out/pack1.st >/dev/null 2>&1; [ $? -ge 128 ] && { echo "META-SCAN-V CRASH"; fail=1; }
# sub-byte dtype (F4, non-multiple-of-8 product) — spec floor-div must accept
python3 - <<'PYEOF'
import struct, json
h = json.dumps({'w': {'dtype': 'F4', 'shape': [3], 'data_offsets': [0, 1]}}).encode()
open('/tmp/f4x.st', 'wb').write(struct.pack('<Q', len(h)) + h + b'\xAB')
PYEOF
"$BIN" c /tmp/f4x.caiw /tmp/f4x.st >/dev/null 2>&1 && "$BIN" v /tmp/f4x.caiw /tmp/f4x.st 2>&1 | grep -q "0 bad" || { echo "F4 SUBBYTE FAILED"; fail=1; }
# DELTA on a zero-copy RAW referent at an ODD archive offset — the referent
# pointer is unaligned; decode must alview-copy it (UBSan alignment check)
python3 - <<'PYEOF'
import json, struct, random
random.seed(7)
n = 4096
w1 = [random.randrange(65536) for _ in range(n)]
w2 = list(w1)
for i in range(0, n, 64): w2[i] ^= 0x0100
for pth, vals in (('/tmp/mA.st', w1), ('/tmp/mB.st', w2)):
    h = json.dumps({'ww': {'dtype':'F16','shape':[n],'data_offsets':[0,2*n]}}).encode()
    open(pth,'wb').write(struct.pack('<Q', len(h)) + h + struct.pack(f'<{n}H', *vals))
PYEOF
"$BIN" c /tmp/mAB.caiw /tmp/mA.st /tmp/mB.st -j1 >/dev/null 2>&1 && \
"$BIN" v /tmp/mAB.caiw /tmp/mA.st /tmp/mB.st -j1 2>&1 | grep -q "0 bad" || { echo "UNALIGNED-DELTA FAILED"; fail=1; }
python3 - <<'PYEOF'
# the test is only meaningful if rec0's RAW payload sits at an odd offset —
# fail loudly if the format layout ever shifts parity
import struct, sys
d = open('/tmp/mAB.caiw','rb').read()
q = 4; nf = struct.unpack('<I', d[q:q+4])[0]; q += 4
for _ in range(nf):
    l = struct.unpack('<I', d[q:q+4])[0]; q += 4 + l
    ml = struct.unpack('<I', d[q:q+4])[0]; q += 4 + ml
nt = struct.unpack('<I', d[q:q+4])[0]; q += 4
nl = struct.unpack('<H', d[q:q+2])[0]; q += 2 + nl
dl = struct.unpack('<H', d[q:q+2])[0]; q += 2 + dl
nd = d[q]; q += 1 + 8*nd + 2 + 1 + 8
q += 8 + 4                      # plen + crc
sys.exit(0 if (q & 1) else 1)   # q = rec0 payload offset; must be odd
PYEOF
[ $? -eq 0 ] || { echo "UNALIGNED-DELTA: payload offset is even (test vacuous)"; fail=1; }
rm -rf /tmp/mdout && "$BIN" d /tmp/mAB.caiw /tmp/mdout -j2 >/dev/null 2>&1 || { echo "UNALIGNED-DELTA-D FAILED"; fail=1; }
# zero-element tensor with nonzero last dim — pos_gain(0)/flog2(0) was UB
python3 - <<'PYEOF'
import json, struct
h = json.dumps({'z': {'dtype':'F16','shape':[0,5],'data_offsets':[0,0]},
                'w': {'dtype':'F16','shape':[4],'data_offsets':[0,8]}}).encode()
open('/tmp/zerodim.st','wb').write(struct.pack('<Q', len(h)) + h + struct.pack('<4H',1,2,3,4))
PYEOF
"$BIN" c /tmp/zd.caiw /tmp/zerodim.st -j1 >/dev/null 2>&1 && \
"$BIN" v /tmp/zd.caiw /tmp/zerodim.st -j1 2>&1 | grep -q "0 bad" || { echo "ZERODIM FAILED"; fail=1; }
for f in tests/corpus/*; do
    [ -e "$f" ] || continue
    timeout 10 "$BIN" c /tmp/tz.caiw "$f" -j2 >/dev/null 2>&1; rc=$?
    [ $rc -ge 128 ] && { echo "CRASH on $f rc=$rc"; fail=1; }
    case "$f" in *.caiw)
        rm -rf /tmp/cdo; mkdir -p /tmp/cdo
        timeout 10 "$BIN" d "$f" /tmp/cdo -j2 >/dev/null 2>&1; rc=$?
        [ $rc -ge 128 ] && { echo "D-CRASH on $f rc=$rc"; fail=1; }
        timeout 10 "$BIN" v "$f" out/edgeA.st -j2 >/dev/null 2>&1; rc=$?
        [ $rc -ge 128 ] && { echo "V-CRASH on $f rc=$rc"; fail=1; }
    ;; esac
done
[ $fail -eq 0 ] && echo "ALL PASS" || echo "FAILURES"
exit $fail

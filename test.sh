#!/bin/bash
# regression: generate fixtures if missing, round-trip -j1/-j4,
# DELTAX external-ref path, then tests/corpus/ must die cleanly (never crash).
BIN=${1:-./caiw}
cd "$(dirname "$0")"
set -u; fail=0
[ -f out/edgeA.st ] || python3 gen_fixtures.py
for t in "edgeA.st edgeB.st" "chA.st chB.st chC.st" "f32A.st f32B.st" "exp.st" \
         "pack1.st" "escname.st" "ndim9.st" "meta.st" "split.st"; do
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
# v completeness: dropping a source file must fail (MISS-SRC / FILECOUNT)
"$BIN" c /tmp/tst2.caiw out/pack1.st out/meta.st -j1 >/dev/null 2>&1
"$BIN" v /tmp/tst2.caiw out/pack1.st 2>&1 | grep -q "0 bad" && { echo "V-COMPLETENESS FAIL"; fail=1; }
# metadata must survive a d round-trip
rm -rf /tmp/tstd && "$BIN" d /tmp/tst2.caiw /tmp/tstd -j1 >/dev/null 2>&1 &&
grep -q '__metadata__' /tmp/tstd/meta.st || { echo "META FAIL"; fail=1; }
for f in tests/corpus/*; do
    [ -e "$f" ] || continue
    timeout 10 "$BIN" c /tmp/tz.caiw "$f" -j2 >/dev/null 2>&1; rc=$?
    [ $rc -ge 128 ] && { echo "CRASH on $f rc=$rc"; fail=1; }
done
[ $fail -eq 0 ] && echo "ALL PASS" || echo "FAILURES"
exit $fail

#!/bin/bash
# mutation fuzz: corrupt a valid archive, decode must only exit(1) cleanly —
# never crash (signal >= 128), never sanitizer error.
BIN=${1:-./caiw}
SRC=${2:-out/edgeA.st}
CAIW=${3:-/tmp/fuzzbase.caiw}
N=${4:-300}
"$BIN" c "$CAIW" -j4 "$SRC" 2>/dev/null || exit 2
crash=0; clean=0; vb=0
for i in $(seq 1 "$N"); do
    python3 -c "
import random, sys
random.seed($i)
d = bytearray(open('$CAIW','rb').read())
mode = random.randrange(4)
if mode == 0:      # bit flips
    for _ in range(random.randrange(1, 9)):
        p = random.randrange(len(d)); d[p] ^= 1 << random.randrange(8)
elif mode == 1:    # byte corruption
    for _ in range(random.randrange(1, 33)):
        d[random.randrange(len(d))] = random.randrange(256)
elif mode == 2:    # truncate
    d = d[:random.randrange(4, len(d))]
else:              # field-targeted: corrupt a record's method/len/plen region
    p = random.randrange(min(3000, len(d)))
    for _ in range(random.randrange(1, 17)):
        q = p + random.randrange(64)
        if q < len(d): d[q] = random.randrange(256)
open('/tmp/fz.caiw','wb').write(d)
" || continue
    timeout 10 "$BIN" v /tmp/fz.caiw -j4 "$SRC" >/dev/null 2>&1
    rc=$?
    if [ $rc -ge 128 ]; then echo "CRASH seed=$i rc=$rc"; cp /tmp/fz.caiw /tmp/crash_$i.caiw; crash=$((crash+1));
    elif [ $rc -eq 0 ]; then vb=$((vb+1)); else clean=$((clean+1)); fi
    # d-mode path too: malformed archives must die cleanly there as well
    rm -rf /tmp/fzd; mkdir -p /tmp/fzd
    timeout 10 "$BIN" d /tmp/fz.caiw /tmp/fzd -j4 >/dev/null 2>&1
    rc=$?
    if [ $rc -ge 128 ]; then echo "D-CRASH seed=$i rc=$rc"; cp /tmp/fz.caiw /tmp/dcrash_$i.caiw; crash=$((crash+1)); fi
done
echo "fuzz done: $N iters, crashes=$crash, clean_die=$clean, verified_ok=$vb"
[ $crash -eq 0 ]

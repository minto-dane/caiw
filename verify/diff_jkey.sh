#!/bin/sh
# diff_jkey.sh — differential test: proved Coq jkey model (extracted to
# OCaml) vs the real C jkey(), on a generated corpus.
#
# This is EMPIRICAL model<->C correspondence evidence, not a refinement
# proof: Jkey.v proves jkey_bound/jkey_fuel/jkey_sem for the abstract
# dec contract and discharges them on the concrete jstr_dec model; the
# diff run then shows the extracted model and the C binary agree on the
# corpus.  What remains unbridged (pointer/NUL memory semantics, the
# strcmp key comparison, xm allocation) is documented in design.md.
#
# usage: diff_jkey.sh [casedir]   (default: verify/jkeycases)
set -e
cd "$(dirname "$0")"
COQC=${COQC:-coqc}
OCAMLFIND=${OCAMLFIND:-ocamlfind}
CC=${CC:-cc}

CASES=${1:-jkeycases}
if [ ! -d "$CASES" ]; then
    python3 gen_jkey_cases.py "$CASES" 1
fi

# 1. extract + build the OCaml oracle (Extract.v now also emits the
#    other models — they are diff_models.sh's business; Jkey.v alone is
#    enough for this oracle but Extract.v needs all five .vo)
for f in Utf8 Norm Pack Rans Jkey; do $COQC -q "$f.v"; done
$COQC -q Extract.v
$OCAMLFIND ocamlopt -package zarith -linkpkg -c jkeym.mli
$OCAMLFIND ocamlopt -package zarith -linkpkg -c jkeym.ml
$OCAMLFIND ocamlopt -package zarith -linkpkg -c diff_jkey.ml
$OCAMLFIND ocamlopt -package zarith -linkpkg jkeym.cmx diff_jkey.cmx \
    -o diff_jkey_ml

# 2. build the C oracle (ASan+UBSan: any OOB inside jkey aborts loudly)
$CC -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    -o diff_jkey_c diff_jkey_c.c -lm -pthread

# 3. run both on every case, compare
./diff_jkey_c "$CASES"/case_*.bin > .diff_c.out
./diff_jkey_ml "$CASES"/case_*.bin > .diff_ml.out
n=$(wc -l < .diff_c.out | tr -d ' ')
hits=$(grep -vc '^-1$' .diff_c.out || true)
if diff -q .diff_c.out .diff_ml.out > /dev/null; then
    echo "diff_jkey: $n cases all agree ($hits hits)"
else
    echo "diff_jkey: MISMATCHES:" >&2
    diff .diff_c.out .diff_ml.out | head -20 >&2
    exit 1
fi

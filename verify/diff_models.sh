#!/bin/sh
# diff_models.sh — differential tests: proved Coq models (extracted to
# OCaml) vs the real C implementations, on generated corpora.
#
# EMPIRICAL model<->C correspondence evidence, not refinement proofs.
# Per-model contract domains (where the proofs and the C code agree by
# construction — inputs outside them are out of scope, not bugs):
#   utf8 : all byte strings
#   norm : histograms with sum(h) < 2^64 (C tot is u64; the u128 path
#          guards h[i]*bud only — past 2^64 tot wraps and every quotient
#          inflates; real per-block histograms never approach this)
#   pack : pack_enc-encodable inputs (bsz in {1,2}, bsz|n, 1<=k<=256)
#   rans : the proved domain x in [LOWER, 256*LOWER), 1<=f<=TOT,
#          0<=c<=TOT-f  (C wraps at u64 outside it)
#
# usage: diff_models.sh [casedir]   (default: verify/diffcases)
set -e
cd "$(dirname "$0")"
COQC=${COQC:-coqc}
OCAMLFIND=${OCAMLFIND:-ocamlfind}
CC=${CC:-cc}

CASES=${1:-diffcases}
if [ ! -d "$CASES" ]; then
    python3 gen_diff_cases.py "$CASES" 1
fi

# 1. compile + extract the proved models (ZBigInt: unary of_nat makes
#    the aw=65536 dict scan / u64 histograms otherwise infeasible)
for f in Utf8 Norm Pack Rans Jkey; do $COQC -q "$f.v"; done
$COQC -q Extract.v
for m in utf8m normm packm ransm; do
    $OCAMLFIND ocamlopt -package zarith -linkpkg -c "$m.mli"
    $OCAMLFIND ocamlopt -package zarith -linkpkg -c "$m.ml"
done
for m in utf8 norm pack rans; do
    $OCAMLFIND ocamlopt -package zarith -linkpkg -c "diff_$m.ml"
    $OCAMLFIND ocamlopt -package zarith -linkpkg "${m}m.cmx" \
        "diff_$m.cmx" -o "diff_${m}_ml"
done

# 2. build the C oracles (ASan+UBSan: memory errors abort loudly)
for m in utf8 norm pack rans; do
    $CC -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
        -o "diff_${m}_c" "diff_${m}_c.c" -lm -pthread
done

# 3. run both on every case, compare
rc=0
for m in utf8 norm pack rans; do
    ./diff_${m}_c "$CASES/$m"/case_*.bin > ".diff_${m}_c.out"
    ./diff_${m}_ml "$CASES/$m"/case_*.bin > ".diff_${m}_ml.out"
    n=$(ls "$CASES/$m"/case_*.bin | wc -l | tr -d ' ')
    if diff -q ".diff_${m}_c.out" ".diff_${m}_ml.out" > /dev/null; then
        echo "diff_$m: $n cases all agree"
    else
        echo "diff_$m: MISMATCHES:" >&2
        diff ".diff_${m}_c.out" ".diff_${m}_ml.out" | head -20 >&2
        rc=1
    fi
done
[ "$rc" -eq 0 ]

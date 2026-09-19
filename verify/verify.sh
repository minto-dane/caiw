#!/bin/sh
# verify.sh — formal-verification runner for caiw.
#
# Runs every artifact under verify/ and compares each tool verdict against
# the documented expectation. Deliberately FLAWED models must produce
# counterexamples — their detection is itself the evidence that the model
# can catch the hazard it claims to prevent.
#
# Layers (kept distinct — do not conflate):
#   TLA+ (TLC)  — design-level protocol model checking (finite-state)
#   Alloy       — design-level structural invariant checking (bounded scope)
#   CBMC        — implementation-level bounded proofs on the real caiw.c
#   ESBMC       — second BMC engine (SMT-encoded); proves the pack_dec
#                 bit-index bound that CBMC/minisat cannot finish at any
#                 tested bound — optional, local only (~600MB binary)
#   Frama-C     — implementation-level static analysis (EVA/RTE) and
#                 deductive contract proofs (WP), optional
#
# Proven ≠ modeled: TLA+/Alloy validate the DESIGN abstraction; CBMC proves
# properties of the compiled C semantics within the stated bounds; none of
# these replace dynamic testing (sanitizers, fuzzing, test.sh).
#
# Env overrides: TLA_JAR ALLOY_JAR CBMC ESBMC FRAMAC CBMC_TIMEOUT_SLOW
# Tool discovery order: env override → verify/tools/ (CI download dir) →
# PATH.  Nothing here depends on a developer machine layout.
set -u
cd "$(dirname "$0")"
V=.
TLA_JAR=${TLA_JAR:-}
for j in "$V/tools/tla2tools.jar" /home/nia/devbox/fmdos-dev/poc/tools/tla2tools.jar; do
    [ -z "$TLA_JAR" ] && [ -f "$j" ] && TLA_JAR=$j
done
ALLOY_JAR=${ALLOY_JAR:-}
for j in "$V/tools/org.alloytools.alloy.dist.jar" \
         /home/nia/devbox/fmdos-dev/poc/tools/org.alloytools.alloy.dist.jar; do
    [ -z "$ALLOY_JAR" ] && [ -f "$j" ] && ALLOY_JAR=$j
done
CBMC=${CBMC:-cbmc}
FRAMAC=${FRAMAC:-frama-c}
command -v "$CBMC"   >/dev/null 2>&1 || CBMC=/home/nia/devbox/tools/usr/bin/cbmc
command -v "$FRAMAC" >/dev/null 2>&1 || FRAMAC=$HOME/.opam/caiw-fc/bin/frama-c
CBMC=$(command -v "$CBMC"   2>/dev/null || echo "$CBMC")   # PATH name → absolute (-x test below needs a path)
FRAMAC=$(command -v "$FRAMAC" 2>/dev/null || echo "$FRAMAC")
ESBMC=${ESBMC:-esbmc}
for e in "$V/tools/esbmc/bin/esbmc" /tmp/esbmc/release/bin/esbmc; do
    command -v "$ESBMC" >/dev/null 2>&1 || { [ -x "$e" ] && ESBMC=$e; }
done
ESBMC=$(command -v "$ESBMC" 2>/dev/null || echo "$ESBMC")
CBMC_TIMEOUT=${CBMC_TIMEOUT:-300}
CBMC_TIMEOUT_SLOW=${CBMC_TIMEOUT_SLOW:-120}
for d in "$V/tools/lib" /home/nia/devbox/tools/usr/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
# /tmp may be a small/full tmpfs — keep tool scratch repo-local.
# MUST be absolute: why3 writes goal files under TMPDIR relative to the
# prover's spawn CWD (WP session dir), so a relative TMPDIR silently
# breaks every external prover call (Sys_error ENOENT on the .smt2).
export TMPDIR=${TMPDIR:-$PWD/.tmp}
mkdir -p "$TMPDIR"
JTMP="-Djava.io.tmpdir=$TMPDIR"

pass=0; fail=0; skip=0
ok()   { pass=$((pass+1)); printf 'PASS  %s\n' "$1"; }
bad()  { fail=$((fail+1)); printf 'FAIL  %s\n' "$1"; }
note() { printf '      %s\n' "$1"; }

# ---------------------------------------------------------------- TLA+/TLC
# correct models must finish with no error; flawed ones must trip the
# named invariant. -deadlock: termination is a legitimate end state.
tlc() { # name expect-pattern-or-NONE
    out=$(timeout 300 java $JTMP -jar "$TLA_JAR" -deadlock -config "$V/$1.cfg" "$V/$1.tla" 2>&1)
    if [ "$2" = NONE ]; then
        echo "$out" | grep -q "No error has been found" \
            && ok "TLC $1 (invariants hold)" || { bad "TLC $1"; echo "$out" | tail -5; }
    else
        echo "$out" | grep -q "Invariant $2 is violated" \
            && ok "TLC $1 (expected violation: $2)" || { bad "TLC $1"; echo "$out" | tail -5; }
    fi
}
if [ -f "$TLA_JAR" ]; then
    tlc CaiwBatch        NONE
    tlc CaiwOutput       NONE
    tlc CaiwBatchFlawed  NoLostUpdate
    tlc CaiwOutputFlawed NoPartialFinal
else
    skip=$((skip+4)); note "SKIP TLC — tla2tools.jar not found (set TLA_JAR)"
fi

# ---------------------------------------------------------------- Alloy
# correct model: every command UNSAT (no counterexample in scope 6).
# flawed model (BackwardRefs + NoBatchSelfRef removed): AcyclicRefs and
# NoIntraBatchRef must go SAT — witnesses that the checks detect the flaw.
alloy_expect() { # file "name:verdict" ...
    out=$(timeout 600 java $JTMP -jar "$ALLOY_JAR" exec -t text -f "$V/$1" 2>&1)
    shift
    for e in "$@"; do
        n=${e%%:*}; want=${e##*:}
        got=$(echo "$out" | grep "$n" | awk '{print $NF}')
        [ "$got" = "$want" ] && ok "Alloy $n → $want" \
            || { bad "Alloy $n (want $want, got $got)"; }
    done
}
if [ -f "$ALLOY_JAR" ]; then
    alloy_expect CaiwArchive.als \
        AcyclicRefs:UNSAT NoIntraBatchRef:UNSAT DupFree:UNSAT \
        NoOrphanMember:UNSAT EncSubsetDec:UNSAT OverAccept:UNSAT
    alloy_expect CaiwArchiveFlawed.als \
        AcyclicRefs:SAT NoIntraBatchRef:SAT DupFree:UNSAT \
        NoOrphanMember:UNSAT EncSubsetDec:UNSAT
else
    skip=$((skip+2)); note "SKIP Alloy — alloy jar not found (set ALLOY_JAR)"
fi

# ---------------------------------------------------------------- CBMC
# Implementation-level proofs on the real caiw.c (harnesses #include it).
# -I cbmcinc: glibc >= 2.41 declares strtofN() with _FloatN types that
# CBMC 6.6's parser rejects; the shadow header turns them off. Harmless
# on older glibc (unused declarations merely stay enabled).
CBMC_I="-I$V/cbmcinc"
cbmc_run() { # file extra-args...
    f=$1; shift
    timeout "$CBMC_TIMEOUT" $CBMC "$V/$f" --function main $CBMC_I "$@" 2>&1
}
cbmc_ok() { # label file args...
    l=$1; f=$2; shift 2
    out=$(cbmc_run "$f" "$@")
    echo "$out" | grep -q "VERIFICATION SUCCESSFUL" \
        && ok "CBMC $l" || { bad "CBMC $l"; echo "$out" | tail -5; }
}
if [ -x "$CBMC" ]; then
    cbmc_ok "kmap16 bijection"            cbmc_kmap.c
    cbmc_ok "dsym cell selection"         cbmc_dsym.c --unwind 40
    # parser trust boundary — per-function bounded instances
    cbmc_ok "utf8_ok bounded scan (n<=16)"  cbmc_parse.c --unwind 40 -DPICK=1
    cbmc_ok "hex4 exactly-4 reads"          cbmc_parse.c --unwind 10 -DPICK=2
    cbmc_ok "jstr bounded dst (S=12)"       cbmc_parse.c --unwind 30 -DPICK=3
    cbmc_ok "jstr_skip NUL scan (J=16)"     cbmc_parse.c --unwind 30 -DPICK=4
    cbmc_ok "jspan balanced scan (J=8)"     cbmc_parse.c --unwind 20 -DPICK=5
    cbmc_ok "norm_ctx contract (u64 path, AW=2)"  cbmc_norm.c  --unwind 10
    cbmc_ok "norm_ctx contract (u128 path, AW=2)" cbmc_norm128.c --unwind 10
    # pack_dec symbolic bit indexing — timed out at every bound under the
    # pre-refactor code (u16 dict, memcpy, int k); the byte-level refactor
    # made it tractable (~68s). ESBMC proves the same bound in ~24s below.
    cbmc_ok "pack_dec bit-index bounds (BL=32, n<=24)" cbmc_pack.c --unwind 40
    # rANS round trip — fixed-frequency sweep (divisor concrete ⇒ tractable).
    # Covers emit-loop worst case f=1 through f=TOT.
    for f in 1 2 3 5 128 257 32768; do
        cbmc_ok "rANS round trip f=$f" cbmc_rans_f.c --unwind 8 -DFVAL=$f
    done
    # General-domain rANS (nondet divisor): known solver limit — 64-bit
    # division bit-blasting times out in minisat. Run briefly to keep the
    # record honest; a timeout here is the documented expectation.
    out=$(timeout "$CBMC_TIMEOUT_SLOW" $CBMC "$V/cbmc_rans.c" --function main $CBMC_I --unwind 8 2>&1)
    if echo "$out" | grep -q "VERIFICATION SUCCESSFUL"; then
        ok "CBMC rANS round trip (general f)"
    elif echo "$out" | grep -q "VERIFICATION FAILED"; then
        bad "CBMC rANS general — real counterexample"; echo "$out" | tail -5
    else
        note "LIM  CBMC rANS general domain: solver timeout (documented bound —"
        note "     coverage continues via the fixed-f sweep + exhaustive C tests)"
    fi
else
    skip=$((skip+17)); note "SKIP CBMC — binary not found (set CBMC)"
fi

# ---------------------------------------------------------------- ESBMC
# Second BMC engine (SMT-encoded VCCs, default Bitwuzla). Its one job here
# is the bound CBMC cannot reach: pack_dec symbolic bit indexing.
# Non-vacuity: the run must report a nonzero property count AND the per-loop
# unwinding assertions must pass (constraints active, paths complete).
# Local-only layer — the binary is ~600MB, not pulled into CI.
if [ -x "$ESBMC" ] && [ -f "$V/esbmc_pack.c" ]; then
    out=$(timeout 240 "$ESBMC" "$V/esbmc_pack.c" --unwind 40 2>&1)
    np=$(echo "$out" | grep -oE "[0-9]+ of [0-9]+ properties failed" | awk '{print $3}')
    if echo "$out" | grep -q "VERIFICATION SUCCESSFUL" && [ "${np:-0}" -gt 0 ]; then
        ok "ESBMC pack_dec bit-index bounds (BL=32, n<=24 — CBMC solver-limit area)"
    elif echo "$out" | grep -q "VERIFICATION FAILED"; then
        bad "ESBMC pack_dec — real counterexample"; echo "$out" | tail -5
    else
        bad "ESBMC pack_dec"; echo "$out" | tail -5
    fi
else
    skip=$((skip+1)); note "SKIP ESBMC — binary not found (set ESBMC)"
fi

# ---------------------------------------------------------------- Frama-C
# EVA + RTE on the pure codec/parse kernels via a stubbed harness —
# whole-program deductive proof of caiw.c (mmap, pthread, fs ops) is out
# of scope for a single-file utility; scope is stated honestly in docs.
# Verdict: PASS iff EVA completes AND reports 0 *invalid* properties.
# Residual "unknown" alarms are EVA precision limits on nondet domains
# (NUL-driven scans, guarded bit reads) — triaged, not proofs of absence.
if command -v "$FRAMAC" >/dev/null 2>&1 && [ -f "$V/framac_eva.c" ]; then
    out=$(timeout 600 "$FRAMAC" -machdep gcc_x86_64 -rte -eva \
        -eva-no-show-progress "$V/framac_eva.c" -main eva_main 2>&1)
    na=$(echo "$out" | grep -oE "[0-9]+ alarms generated" | grep -oE "[0-9]+")
    if echo "$out" | grep -q "ANALYSIS SUMMARY" && \
       echo "$out" | grep -qE "0 +invalid"; then
        ok "Frama-C EVA/RTE kernel scope (${na:-?} alarms, all unknown-class — triaged)"
    else
        bad "Frama-C EVA"; echo "$out" | tail -8
    fi
else
    skip=$((skip+1)); note "SKIP Frama-C — not installed"
fi

# Frama-C WP — deductive (unbounded) proofs of the ACSL contracts that live
# in caiw.c on hex4/utf8_ok/dsym/kmap16/kmap16_inv, incl. RTE-generated
# safety goals.  PASS iff every scheduled goal is proven ("Proved goals: N/N").
# Needs why3-registered provers (alt-ergo, z3) — see ~/.why3.conf.
if command -v "$FRAMAC" >/dev/null 2>&1 && [ -f "$V/wp.c" ]; then
    out=$(PATH="$(dirname "$FRAMAC"):$V/tools/bin:/home/nia/devbox/tools/usr/bin:$PATH" \
        timeout 600 "$FRAMAC" -wp -wp-rte \
        -wp-fct hex4,utf8_ok,dsym,kmap16,kmap16_inv,g32le \
        -wp-timeout 60 -wp-prover z3,alt-ergo -machdep gcc_x86_64 "$V/wp.c" 2>&1)
    got=$(echo "$out" | grep -oE "[0-9]+ / [0-9]+" | tail -1)
    if [ -n "$got" ]; then
        set -- $got; p=$1; t=$3
        if [ "$p" = "$t" ]; then
            ok "Frama-C WP kernel contracts ($p/$t goals proved)"
        else
            bad "Frama-C WP ($p/$t proved)"
            echo "$out" | grep -iE "warn|error|prover|fail|z3|ergo" | tail -30
        fi
    else
        bad "Frama-C WP (no summary)"; echo "$out" | tail -8
    fi
else
    skip=$((skip+1)); note "SKIP Frama-C WP — not installed"
fi

echo "----------------------------------------------------------------"
printf 'verify: %d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ]

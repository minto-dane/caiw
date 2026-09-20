/* CBMC harness: parser-layer memory safety — the functions that touch
 * attacker-controlled bytes first.  Properties are the instrumented
 * runtime checks (bounds, init, overflow, shift, div-by-zero) plus
 * explicit post conditions where they exist.
 *
 * Each block is a separate SAT instance — select with -DPICK=n:
 *   1 utf8_ok(s,n)   bounded scan; proven at N=16 (N=24 exceeds solver
 *                    limits — every byte branches on 4 UTF-8 classes)
 *   2 hex4(p,&out)   exactly-4-byte buffer proves it never reads p[4]
 *   3 jstr(p,buf,bs) JSON string parse; writes stay in buf[0..bs), S=12
 *   4 jstr_skip      NUL-driven skip inside a bounded buffer, J=16
 *   5 jspan          balanced {..}/[..] scan, J=8
 *   6 jkey           nested key scan — smallest useful bound (O=6,R=4);
 *                    larger bounds exceed minisat limits — Jkey.v covers
 *                    the algorithm at arbitrary length
 * pack_dec (bit-level index arithmetic) has its own harness in
 * cbmc_pack.c / esbmc_pack.c. */
#define main caiw_main_
#include "../caiw.c"
#undef main
uint64_t nondet_u64(void);
#ifndef PICK
#define PICK 1
#endif

int main() {
#if PICK == 1
    enum { N = 16 };
    char s[N];
    for (int i = 0; i < N; i++) s[i] = (char)nondet_u64();
    size_t n = nondet_u64(); __CPROVER_assume(n <= N);
    utf8_ok(s, n);
#elif PICK == 2
    char p[4]; unsigned out;
    for (int i = 0; i < 4; i++) p[i] = (char)nondet_u64();
    if (hex4(p, &out)) __CPROVER_assert(out < 65536, "hex4 range");
#elif PICK == 3
    enum { S = 12, B = 8 };
    char src[S], buf[B];
    for (int i = 0; i < S - 1; i++) src[i] = (char)nondet_u64();
    src[S - 1] = 0;
    size_t bs = nondet_u64(); __CPROVER_assume(bs <= B);
    jstr(src, buf, bs);
#elif PICK == 4
    enum { J = 16 };
    char j[J];
    for (int i = 0; i < J - 1; i++) j[i] = (char)nondet_u64();
    j[J - 1] = 0;
    jstr_skip(j);
#elif PICK == 5
    enum { J = 8 };
    char j[J];
    for (int i = 0; i < J - 1; i++) j[i] = (char)nondet_u64();
    j[J - 1] = 0;
    if (*j == '{' || *j == '[') jspan(j);
#elif PICK == 6
#ifndef OB
#define OB 6
#endif
#ifndef RB
#define RB 4
#endif
    enum { O = OB, R = RB, K = 4 };
    char buf[O + R], key[K];
    for (int i = 0; i < O + R - 1; i++) buf[i] = (char)nondet_u64();
    buf[O + R - 1] = 0;
    for (int i = 0; i < K - 1; i++) key[i] = (char)nondet_u64();
    key[K - 1] = 0;
    size_t olen = nondet_u64(); __CPROVER_assume(olen <= O);
    const char *r = jkey(buf, buf + olen, key);
    if (r) __CPROVER_assert(r >= buf && r < buf + olen, "in span");
#endif
    return 0;
}

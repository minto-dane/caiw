/* CBMC harness: rANS round trip, REAL constants, divisor fixed per run.
 * -DF=<freq> picks f concretely: division-by-constant bit-blasts far
 * easier than nondet divisor while x,c stay fully nondet.
 * Sweep f over {1, 3, 257, 27053, TOT} — includes the emit-loop worst
 * case (f=1) and a power-of-two (TOT). */
#define main caiw_main_
#include "../caiw.c"
#undef main
#ifndef FVAL
#define FVAL 1
#endif

int main() {
    uint64_t x; __CPROVER_assume(x >= LOWER && x < (LOWER << 8));
    const uint32_t f = FVAL;
    uint32_t c; __CPROVER_assume(c <= TOT - f);
    uint8_t buf[64]; uint8_t *pp = buf + sizeof buf;
    uint64_t x1 = enc(x, f, c, &pp);
    __CPROVER_assert(pp >= buf, "enc stays in scratch");
    const uint8_t *rp = pp;
    uint64_t x2 = dec(x1, f, c, &rp, buf + sizeof buf);
    __CPROVER_assert(x2 == x, "rans round trip");
    return 0;
}

/* CBMC harness: rANS single-symbol round trip with REAL constants.
 * x ∈ [LOWER, 256·LOWER), f ∈ [1,TOT], c ∈ [0, TOT-f].
 * The emit loop iterates at most ceil((39-log2(XMAX(1)))/8)+1 = 3 times
 * (f=1 worst case: x<2^39, XMAX(1)=2^24).  --unwind 5 covers it.
 * dec's renorm loop reads at most as many bytes as enc emitted. */
#define main caiw_main_
#include "../caiw.c"
#undef main

int main() {
    uint64_t x; __CPROVER_assume(x >= LOWER && x < (LOWER << 8));
    uint32_t f; __CPROVER_assume(f >= 1 && f <= TOT);
    /* NB: "c + f <= TOT" alone wraps in u32 and admits c≈4G — constrain
       with subtraction form so the domain is exactly c ∈ [0, TOT-f] */
    uint32_t c; __CPROVER_assume(c <= TOT - f);
    uint8_t buf[64]; uint8_t *pp = buf + sizeof buf;
    uint64_t x1 = enc(x, f, c, &pp);
    __CPROVER_assert(pp >= buf, "enc stays in scratch");
    const uint8_t *rp = pp;
    uint64_t x2 = dec(x1, f, c, &rp, buf + sizeof buf);
    __CPROVER_assert(x2 == x, "rans round trip");
    return 0;
}

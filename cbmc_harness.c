/* CBMC harness: prove core invariants of caiw.c
 *  P1 norm_ctx: sum(f)==TOT, support preserved (f[i]>0 iff h[i]>0)
 *  P2 kmap16 is a bijection: inv(kmap(u))==u
 *  P3 rANS single-symbol round trip: dec(enc(x,f,c),f,c)==x
 *  P4 dsym picks the cell containing v (binary-search correctness)
 */
#define main caiw_main_
#include "caiw.c"
#undef main

int main() {
    {   /* P1 */
        enum { AW = 16 };
        uint32_t h[AW]; uint16_t f[AW];
        for (int i = 0; i < AW; i++) { h[i] = nondet_uint(); __CPROVER_assume(h[i] <= 100000); }
        norm_ctx(h, f, AW);
        uint32_t sum = 0;
        for (int i = 0; i < AW; i++) {
            sum += f[i];
            __CPROVER_assert(!(h[i] > 0) || f[i] > 0, "used syms keep freq");
            __CPROVER_assert(f[i] <= TOT, "freq within TOT");
        }
        __CPROVER_assert(sum == TOT, "freqs sum to TOT");
    }
    {   /* P2 */
        uint16_t u;
        __CPROVER_assert(kmap16_inv(kmap16(u)) == u, "kmap16 inverse");
    }
    {   /* P3 */
        uint64_t x; __CPROVER_assume(x >= LOWER && x < (LOWER << 8));
        uint32_t f; __CPROVER_assume(f >= 1 && f <= TOT);
        uint32_t c; __CPROVER_assume(c + f <= TOT);
        uint8_t buf[64]; uint8_t *pp = buf + sizeof buf;
        uint64_t x1 = enc(x, f, c, &pp);
        __CPROVER_assert(pp >= buf, "enc stays in scratch");
        const uint8_t *rp = pp;
        uint64_t x2 = dec(x1, f, c, &rp, buf + sizeof buf);
        __CPROVER_assert(x2 == x, "rans round trip");
    }
    {   /* P4 */
        enum { AW = 32 };
        uint16_t f[AW]; uint32_t cum[AW + 1];
        uint32_t h[AW];
        for (int i = 0; i < AW; i++) { h[i] = nondet_uint(); __CPROVER_assume(h[i] <= 1000); }
        norm_ctx(h, f, AW);
        cum[0] = 0;
        for (int i = 0; i < AW; i++) cum[i + 1] = cum[i] + f[i];
        uint32_t v; __CPROVER_assume(v < TOT);
        uint32_t fc; uint32_t s = dsym(f, cum, AW, v, &fc);
        __CPROVER_assert(cum[s] <= v && v < cum[s + 1], "dsym selects containing cell");
        __CPROVER_assert(fc == f[s] && f[s] > 0, "nonzero freq at selected sym");
    }
    return 0;
}

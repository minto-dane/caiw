/* CBMC harness: dsym cell-selection correctness — isolated from norm_ctx.
 * dsym is pure binary search over a monotone cum[]; validity needs only:
 *   cum[0]==0, cum monotone nondecreasing, cum[aw]==TOT, f[i]>0 iff cum grows.
 * Generating f/cum via norm_ctx would drag 64-bit division into the SAT
 * instance for no benefit — constrain the invariant directly instead. */
#define main caiw_main_
#include "../caiw.c"
#undef main

int main() {
    enum { AW = 32 };
    uint16_t f[AW]; uint32_t cum[AW + 1];
    /* nondet normalized table: each f[i] in [0,TOT], exactly summing TOT,
       nonzero-support guaranteed by requiring f[i]>0 where cum increases */
    cum[0] = 0;
    for (int i = 0; i < AW; i++) {
        uint32_t d; __CPROVER_assume(d <= (uint32_t)(TOT - cum[i]));
        /* forbid leaving a deficit no remainder can fill: if this is the
           last cell, force exact completion */
        if (i == AW - 1) __CPROVER_assume(d == TOT - cum[i]);
        /* at least one nonzero cell overall is guaranteed by i==AW-1 forcing
           completion AND by requiring d>0 when cum[i]==0 is impossible only
           if a later cell picks up — simpler: require support consistency */
        f[i] = (uint16_t)d;
        cum[i + 1] = cum[i] + d;
        __CPROVER_assume(!(d > 0) || f[i] > 0);   /* support consistency */
    }
    __CPROVER_assume(cum[AW] == TOT);

    uint32_t v; __CPROVER_assume(v < TOT);
    uint32_t fc; uint32_t s = dsym(f, cum, AW, v, &fc);
    __CPROVER_assert(cum[s] <= v && v < cum[s + 1], "dsym selects containing cell");
    __CPROVER_assert(fc == f[s] && f[s] > 0, "nonzero freq at selected sym");
    return 0;
}

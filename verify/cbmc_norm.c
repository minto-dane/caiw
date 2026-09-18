/* CBMC harness: norm_ctx output contract, u64-multiply path (tot < 2^49).
 *   sum(f)==TOT, support preserved (f[i]>0 iff h[i]>0, nz>0 case),
 *   f[i]<=TOT, plus all runtime checks.
 * AW=2 is the largest alphabet that fits minisat in this environment
 * (AW=4 and AW=8 instances time out at 240s — the h[i]*bud/tot division
 * bit-blast dominates). h[i]<=100000 exercises the general u64 path:
 * rounding, +1 bias, deficit redistribution into the dominant cell, and
 * the all-empty uniform fallback (nz==0).  cbmc_norm128.c covers the
 * u128 path; larger alphabets are covered by exhaustive C sweeps
 * (see docs/design.md verification notes). */
#define main caiw_main_
#include "../caiw.c"
#undef main
uint64_t nondet_u64(void);   /* undeclared-type nondet is 32-bit; declare for u64 */

int main() {
    enum { AW = 2 };
    uint64_t h[AW]; uint16_t f[AW];
    for (int i = 0; i < AW; i++) { h[i] = nondet_u64(); __CPROVER_assume(h[i] <= 100000); }
    norm_ctx(h, f, AW);
    uint32_t sum = 0; int nz = 0;
    for (int i = 0; i < AW; i++) { sum += f[i]; if (h[i]) nz++; }
    for (int i = 0; i < AW; i++) {
        __CPROVER_assert(!(h[i] > 0) || f[i] > 0, "used syms keep freq");
        /* unused cells are zeroed only on the nz>0 path — the all-empty
           fallback deliberately assigns a uniform table (never encoded) */
        __CPROVER_assert(!(nz > 0 && h[i] == 0) || f[i] == 0, "unused syms zero freq");
        __CPROVER_assert(f[i] <= TOT, "freq within TOT");
    }
    __CPROVER_assert(sum == TOT, "freqs sum to TOT");
    return 0;
}

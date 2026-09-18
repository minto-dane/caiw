/* CBMC harness: norm_ctx output contract, u128-multiply path.
 * Same properties as cbmc_norm.c; h[0] >= 2^49 forces tot >= 2^49 so the
 * (unsigned __int128) branch is the one exercised.  AW=2 for the same
 * solver-limit reason documented there. */
#define main caiw_main_
#include "../caiw.c"
#undef main
uint64_t nondet_u64(void);   /* nondet_uint() tops out at 2^32-1 — never reaches 2^49 */

int main() {
    enum { AW = 2 };
    uint64_t h[AW]; uint16_t f[AW];
    for (int i = 0; i < AW; i++) {
        h[i] = nondet_u64();
        __CPROVER_assume(h[i] <= (((uint64_t)1 << 56) - 1));   /* keep h*bud < 2^128 trivially */
    }
    __CPROVER_assume(h[0] >= ((uint64_t)1 << 49));   /* force the u128 branch */
    norm_ctx(h, f, AW);
    uint32_t sum = 0; int nz = 0;
    for (int i = 0; i < AW; i++) { sum += f[i]; if (h[i]) nz++; }
    for (int i = 0; i < AW; i++) {
        __CPROVER_assert(!(h[i] > 0) || f[i] > 0, "used syms keep freq");
        __CPROVER_assert(!(nz > 0 && h[i] == 0) || f[i] == 0, "unused syms zero freq");
        __CPROVER_assert(f[i] <= TOT, "freq within TOT");
    }
    __CPROVER_assert(sum == TOT, "freqs sum to TOT");
    return 0;
}

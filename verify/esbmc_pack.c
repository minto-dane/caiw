/* ESBMC harness: pack_dec bit-level bounds on an arbitrary blob.
 * Same property set as cbmc_pack.c, which CBMC/minisat cannot finish at
 * ANY tested bound.  ESBMC 8.5 + Bitwuzla 0.9.0 proves all 27 safety
 * properties at BL=32, n<=24, bsz in {1,2} in ~16s — symbolic bit
 * indexing (in[bit>>3] >> (7-(bit&7))) is hard for SAT bit-blasting but
 * tractable for SMT bit-vector+array encoding.
 *
 * Assumptions use early return, not solver intrinsics, so the file also
 * parses under CBMC for cross-checking (it still times out there).
 * realpath is declared explicitly because ESBMC's bundled libc model
 * lacks the prototype (identical to the glibc signature — harmless
 * duplicate elsewhere).  `data` is 64B >= n*bsz worst case (48B); the
 * earlier cbmc_pack.c used 32B — undersized, fixed there too.
 *
 * Larger bounds (BL=64/n<=48/bsz<=4) exceed solver time — the residual
 * documented limit.  Run: esbmc esbmc_pack.c --unwind 40 */
char *realpath(const char *restrict, char *restrict);
#define main caiw_main_
#include "../caiw.c"
#undef main
uint64_t nondet_u64(void);

int main() {
    enum { BL = 32 };
    uint8_t blob[BL];
    for (int i = 0; i < BL; i++) blob[i] = (uint8_t)nondet_u64();
    uint64_t n = nondet_u64();
    if (n > 24) return 0;                    /* portable assume */
    int bsz = 1 + (int)(nondet_u64() & 1);   /* {1,2} */
    uint8_t data[64];
    pack_dec(blob, blob + BL, n, bsz, data);
    return 0;
}

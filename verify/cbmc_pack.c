/* CBMC harness: pack_dec bit-level bounds on an arbitrary blob.
 * Every guard (k range, dict bytes, idx<k, bit position vs lim, exact
 * tail consumption) is exercised by nondet input; properties are the
 * instrumented bounds/init checks.
 *
 * KNOWN SOLVER LIMIT: times out under minisat at every tested bound
 * (BL=16/n<=8 upward) — symbolic bit indexing (in[bit>>3] >> (7-(bit&7)))
 * is SAT-hard.  Retained so a faster solver (or CBMC upgrade) can run it;
 * verify.sh does NOT run this file.  Dynamic coverage:
 * ASan + the crash corpus exercise these guards continuously. */
#define main caiw_main_
#include "../caiw.c"
#undef main
uint64_t nondet_u64(void);

int main() {
    enum { BL = 32 };
    uint8_t blob[BL];
    for (int i = 0; i < BL; i++) blob[i] = (uint8_t)nondet_u64();
    uint64_t n = nondet_u64(); __CPROVER_assume(n <= 24);
    int bsz = 1 + (int)(nondet_u64() & 1);
    uint8_t data[64];   /* >= n*bsz worst case (48B) — was undersized at 32B */
    pack_dec(blob, blob + BL, n, bsz, data);
    return 0;
}

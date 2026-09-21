/* Frama-C WP deductive-verification target: functional contracts on caiw's
 * pure kernels, discharged by WP (Qed + external provers) — an UNBOUNDED
 * proof per contract, unlike CBMC's bounded model checking.
 *
 * Contracts live in caiw.c itself (ACSL annotations — invisible to
 * gcc, verified here).  Scope (26 functions): byte-order codecs
 * (p16le/p32le/p64le, g32le/g64le), fnv, crc_setup/crc32_of, hex4,
 * utf8_ok, dsym, kmap16(+inv)/kmap32, the dtype predicate family
 * (is_bf/is_f16/is_f32/is_flt16/mbits_of/dbits — strcmp specs come
 * from Frama-C's bundled libc), ck_shape_len (bounded shape loop,
 * u128 product, die() paths), the rANS state transitions enc/dec
 * (state band + cursor bounds), the block framing pair
 * emit_blk/read_blk (header+payload bounds, exits/terminates for
 * the die() paths), and norm_ctx (memory safety of both branches,
 * dominant-cell index bounds, zero-support preservation
 * h[i]==0 ==> f[i]==0, signed-remainder bounds — under an explicit
 * h[i] <= 2^48-1 per-cell precondition that keeps tot = Σh exact
 * in u64; real histograms count block elements, orders of magnitude
 * below that bound, but callers are outside WP scope so the bound
 * is an assumption, not a discharged obligation).
 * Still outside WP: norm_ctx's full invariants Σf==TOT and
 * h[i]>0 ==> f[i]>0 need \sum/quotient-sum reasoning WP 33 does
 * not discharge; NUL-driven scans (jstr/jws/jkey/jget) need the
 * strlen axioms (strlen_not_zero & co.) whose quantifier
 * instantiation times out in z3/alt-ergo — attempted, reverted,
 * documented boundary (docs/design.md).  CBMC + the exhaustive
 * sweeps own those functions. */
#ifdef __FRAMAC__
#ifndef MADV_SEQUENTIAL
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_HUGEPAGE 14
#endif
int madvise(void *addr, unsigned long length, int advice);
#endif
#define main caiw_main_
#include "../caiw.c"
#undef main

/* kmap16's uint16 bijection (inv∘map = id) is already FULLY proven by CBMC —
 * the domain is finite (all 65,536 inputs) so the bounded check is complete.
 * Composing the bitwise contracts through WP needs bitwise→arith bridging
 * lemmas Qed does not carry — documented boundary, not a gap in evidence. */

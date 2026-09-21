/* Frama-C WP deductive-verification target: functional contracts on caiw's
 * pure kernels, discharged by WP (Qed + external provers) — an UNBOUNDED
 * proof per contract, unlike CBMC's bounded model checking.
 *
 * Contracts live in caiw.c itself (ACSL annotations — invisible to
 * gcc, verified here).  Scope (30 functions): byte-order codecs
 * (p16le/p32le/p64le, g32le/g64le), fnv, crc_setup/crc32_of, hex4,
 * utf8_ok, dsym, kmap16(+inv)/kmap32, the dtype predicate family
 * (is_bf/is_f16/is_f32/is_flt16/mbits_of/dbits/dtb — strcmp specs
 * come from Frama-C's bundled libc), ck_shape_len (bounded shape
 * loop,
 * u128 product, die() paths), the rANS state transitions enc/dec
 * (state band + cursor bounds), the block framing pair
 * emit_blk/read_blk (header+payload bounds, exits/terminates for
 * the die() paths), and norm_ctx — the block-table normalizer:
 * both division branches memory-safe, dominant-cell bounds,
 * zero-support preservation (h[i]==0 ==> f[i]==0), POSITIVE
 * support (h[i]>0 ==> f[i]>=1 — the property decoders rely on for
 * div-safety), uniform cell bound f[i]<=TOT, and the all-zero
 * branch's uniform positivity — under explicit requires
 * 1<=aw<=32768 (bud>=0) and h[i]<=2^48-1 (tot = Σh exact in u64;
 * real callers pass aw<=1024 and histograms count block elements,
 * but callers are outside WP scope so these stay assumptions).
 * The final dominant-cell update is gated by (rem>0) (identical
 * on every reachable state since rem>=0 always — Norm.v proves it
 * at model level; CBMC is shown the unconditional add via
 * __CPROVER__ so its sum==TOT check stays structural rather than
 * requiring the solver to discharge rem>=0 itself) so every
 * ensures holds WITHOUT discharging the
 * quotient-sum bound Σq<=TOT in WP: that bound needs partial-sum
 * reasoning (Σ_{j<i} h_j <= tot) whose quantifier instantiations
 * drown Z3 — attempted via ghost prefix arrays and an axiomatic
 * recursive sum, reverted, documented.  Σf==TOT and rem>=0 stay
 * with Norm.v (all histograms, axiom-free) + the sweeps.
 * joinable (bounded candidate scan, 0/1 result), ebound
 * (non-wrapping capacity formula — true bound recorded as the
 * terminates condition) and dec_aux (assigns \nothing plus an
 * honest non-wrap requires; a semantic ensures drowned in the
 * strcmp-derived context — reverted) round out the scope.
 * Still outside WP: NUL-driven scans (jstr/jws/jkey/jget) need
 * the strlen axioms whose quantifier instantiation times out in
 * z3/alt-ergo — attempted, reverted, documented boundary
 * (docs/design.md).  The u8_enc/u8_dec/xm composite layer was
 * likewise attempted and reverted: *rp returned through assigns
 * boundaries loses syntactic base identity so the callers'
 * \valid_read subrange requires don't match (the \base_addr
 * ensures now on dec/read_blk plus pointer re-anchoring fixed
 * part of it), and 256-cell loop-assigns forall-preservation
 * goals drown Z3 in large-function contexts — every surviving
 * goal was a true proposition; zero code defects were found.
 * The u8 layer keeps the runtime cum[256]==TOT guard that the
 * effort added.  CBMC + the exhaustive sweeps own those. */
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

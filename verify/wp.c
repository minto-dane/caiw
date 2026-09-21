/* Frama-C WP deductive-verification target: functional contracts on caiw's
 * pure kernels, discharged by WP (Qed + external provers) — an UNBOUNDED
 * proof per contract, unlike CBMC's bounded model checking.
 *
 * Contracts live in caiw.c itself (ACSL annotations — invisible to
 * gcc, verified here).  Scope (36 functions): byte-order codecs
 * (p16le/p32le/p64le, g32le/g64le), the little-endian archive
 * readers r64/r32/r16/r8 (pointer advance + in-range reads on the
 * untrusted record-walk path), fnv, crc_setup/crc32_of, hex4,
 * utf8_ok, dsym (incl. *fc>0 — the caller-visible form of its
 * exit asserts), kmap16(+inv)/kmap32, the dtype predicate family
 * (is_bf/is_f16/is_f32/is_flt16/mbits_of/dbits/dtb — strcmp specs
 * come from Frama-C's bundled libc), ck_shape_len (bounded shape
 * loop,
 * u128 product, die() paths), the rANS state transitions enc/dec
 * (state band + cursor bounds), the block framing pair
 * emit_blk/read_blk (header+payload bounds, exits/terminates for
 * the die() paths), norm_ctx — the block-table normalizer:
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
 * strcmp-derived context — reverted), and the U8 DECODE layer —
 * closed by decomposition: the per-block inner loop is u8_blk
 * (a small void function taking the stream cursor by value, so
 * *rp is never assigned across a call boundary), contracted
 * with the normalized-table facts (cum monotone, cum[256]==TOT,
 * cum[i]<cum[i+1] ==> ft[i]>0), the h-increment bound, and the
 * stream-validity invariant; u8_dec itself then proves the
 * whole block loop — memory safety, per-cell h growth <= n, and
 * exact-consumption \result==lim — with dsym's *fc>0 feeding
 * dec()'s f>=1 domain.  The earlier monolithic attempt failed
 * for solver, not semantic, reasons: *rp returned through
 * assigns boundaries loses syntactic base identity and 256-cell
 * loop-assigns forall-preservation drowned Z3 — decomposition
 * shrinks each goal's context enough that the same facts close.
 * Still outside WP: NUL-driven scans (jstr/jws/jkey/jget) need
 * the strlen axioms whose quantifier instantiation times out in
 * z3/alt-ergo — attempted, reverted, documented boundary
 * (docs/design.md).  The u8_enc side resists the analogous
 * decomposition: its backward-moving write cursor needs a
 * range-assigns frame whose Qed encoding drowns the same
 * preservation goals — encoder input is trusted, so CBMC + the
 * exhaustive sweeps + Norm.v continue to own that side; the
 * runtime cum[256]==TOT guard stays on both paths. */
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

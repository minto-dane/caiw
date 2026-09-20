/* Frama-C WP deductive-verification target: functional contracts on caiw's
 * pure kernels, discharged by WP (Qed + external provers) — an UNBOUNDED
 * proof per contract, unlike CBMC's bounded model checking.
 *
 * Contracts live in caiw.c itself (ACSL annotations — invisible to
 * gcc, verified here).  Scope: byte-order codecs (p16le/p32le/p64le,
 * g32le/g64le), fnv, hex4, utf8_ok, dsym, kmap16(+inv), the rANS state
 * transitions enc/dec (state bounds, read/write cursor bounds), and the
 * block framing pair emit_blk/read_blk (header+payload bounds, including
 * read_blk's exits/terminates contract for the die() paths).
 * norm_ctx's sum-of-frequencies invariants need \sum/lambda constructs
 * WP 33 does not implement, and pointer-scan parsers (jstr/jkey/jget)
 * sit behind Qed's missing strlen/valid_read_string model — attempted,
 * reverted, documented boundary (docs/design.md).  CBMC + the exhaustive
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

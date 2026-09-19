/* Frama-C WP deductive-verification target: functional contracts on caiw's
 * pure kernels, discharged by WP (Qed + external provers) — an UNBOUNDED
 * proof per contract, unlike CBMC's bounded model checking.
 *
 * Contracts live in caiw.c itself (ACSL annotations — invisible to
 * gcc, verified here).  Scope: g32le / hex4 / utf8_ok / dsym / kmap16(+inv).
 * Division-heavy kernels (enc/dec, norm_ctx) and pointer-scan parsers
 * (jstr/jstr_skip/jspan/jkey) are outside WP's practical reach — Qed carries
 * no strlen/valid_read_string model and its bit-index arithmetic defeats the
 * solvers (pack_dec: 54/64 non-terminates goals proved, the bit-position
 * loop invariants time out — attempted, then reverted to keep every in-tree
 * annotation machine-checked).  CBMC/exhaustive sweeps own those
 * (honest boundary, docs/design.md). */
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

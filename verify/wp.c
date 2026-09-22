/* Frama-C WP deductive-verification target: functional contracts on caiw's
 * pure kernels, discharged by WP (Qed + external provers) — an UNBOUNDED
 * proof per contract, unlike CBMC's bounded model checking.
 *
 * Contracts live in caiw.c itself (ACSL annotations — invisible to
 * gcc, verified here).  Scope (65 functions): byte-order codecs
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
 * The F16/FIELD decoder layer is closed the same way, one level
 * deeper: f16_elem (one element: S/E/M symbol decodes via
 * dsym_raw — a memory-safety-only binary-search variant whose
 * contract needs no table semantics — with every dec() call
 * guarded by a runtime fc check so div-safety is a path fact,
 * not a quantified hypothesis), f16_blk (per-block loop, cursor
 * by value), fill_ctx (per-context prefix-sum build, u32-safe by
 * the linear 65535*i bound — f entries are u16 so no table
 * invariant is needed at all), norm_fM (the 2ew-context
 * norm_ctx sweep with an int64 index check per context), and
 * f16_tab (whole ft/cum construction); f16_dec then proves the
 * block loop, exact consumption (rp==lim), memory safety, and
 * the h-growth bound.  ew/mw enter as literals per mb∈{7,10}
 * branch — semantically identical (mb is contract-bound) but
 * keeps 1<<(15-mb) shift evaluation out of every goal.
 * Quantifier-instantiation notes: ctx-index products like
 * c*(mw+1) in forall hypotheses defeat e-matching (Qed's let-CSE
 * renames the products), so the layer carries NO semantic cum
 * invariants — the runtime cum[aw]==TOT and fc guards (unreachable
 * on encoder-produced tables, die() otherwise) plus Norm.v own
 * those facts instead.  The FIELDPOS/FIELDROW decoder closes
 * identically: pos_dec is a thin alloc+call+free shell and the
 * real work lives in pos_dec_ws (heap workspace passed as
 * pointers — xm/xc cannot be contracted since \fresh cannot
 * re-export malloc's bundled spec; callers are out of scope
 * anyway), pos_tab walks contexts with accumulator indices
 * (off/fo/k2 — zero product-index goals), pos_blk bounds the
 * position ctx via a po>=P die (unreachable when P|n) and a
 * __int128 po*K/P product, and pos_elem decodes SE/M through
 * dsym_raw with int64 checks feeding every bound as a path
 * fact.  The F32 decoder reuses the identical template at five
 * decode stages (S/E/m1/m2/m3, se<=511 ctx, fixed 328194-cell
 * ft / 329730-cell cum workspaces): f32_elem (5 dsym_raw +
 * 5 runtime fc guards + int64 product checks), f32_blk,
 * f32_tab (norm_ctx/fill_ctx sweeps over accumulator indices
 * ko/fo carrying the equalities ko==c*(aw+1)/fo==c*aw — the
 * equalities, not bounds alone, are what let call-site
 * requires instantiate), f32_dec_ws (block loop + rp==lim).
 * pack_dec proves bit-extraction safety and exact consumption
 * (lim-in == ceil(ne*ib/8)): k∈[1,256]/idx<k stay runtime
 * checks that themselves supply every downstream bound, bit
 * reads use /8 and %8 arithmetic forms, and the ib width loop
 * is a doubling p2 accumulator with an unreachable ib>=8
 * die-guard (k<=256 makes it dead — the guard, not lsl
 * reasoning, supplies the bound).  The DELTA16 layer is fully
 * decomposed: dlt_val (kmap16/kmap16_inv value mapping pulled
 * non-inline — the inlined bit_test chain poisoned every
 * enclosing goal), dlt_elem (one symbol; ctx = ref>>9 written
 * /512, the u16-load upper bound supplied by an explicit
 * assert since the model does not infer C integer ranges, a
 * (uint32_t) cast ahead of the shift keeps every stage
 * nonneg), dlt_blk, dlt_tab (per-ctx norm_ctx + fill_ctx over
 * accumulator offsets; the earlier fo<=520319 invariant was a
 * genuine bound bug — the true max at c=128 is 524416),
 * dlt_dec_ws (workspace form; the escape-position list that
 * used to realloc inside the element loop is now an escbits
 * bitmap — the escape count is derived as its popcount, so no
 * pointer-count invariant is needed), dlt_scatter (bitmap-
 * driven escape collection) and dlt_tail (escape values run
 * through f16_dec, inheriting its own rp==lim check);
 * zero-escape payloads close with rp==lim.  The DELTA32 layer
 * repeats the template over four byte planes: d32_elem uses a
 * saturating histogram increment (if (hp[i] != UINT64_MAX)
 * hp[i]++ — unreachable for counts, removes every u64-overflow
 * precondition at once), d32_blk/d32_tab mirror DELTA16,
 * d32_plane lifts the whole per-plane pass to a function
 * boundary (its ensures carry the per-cell +n growth bound —
 * a mixed-plane h[j]<=Pre+c0 invariant would be false from
 * pass 2 on since earlier planes' cells already grew), and
 * dlt32_dec_ws unrolls the four plane calls with literal
 * offsets h+0/65536/131072/196608 — literal indices make every
 * slice instantiation, separation fact and assigns-disjointness
 * decidable, whereas the same goals with a symbolic p*65536
 * index drown e-matching (verified by minimal repro).  The PRW
 * decoder (prw_dec_ws behind a thin alloc shell) proves the
 * row framing with a linear pos cursor in place of r*cols,
 * per-row [u32 len] bounds l1/lr, exact in==lim consumption,
 * and workspace-slice separation stated as asserts.
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

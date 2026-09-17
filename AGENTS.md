# caiw — Compressor for AI Weights

Lossless (byte-exact) archiver for safetensors model weights. Single dependency-free C file (pthread only).

## Build

```
gcc -O3 -march=native -Wall -Wextra -pthread -o caiw caiw.c -lm
```

## Usage

```
./caiw c out.caiw [-j N] [--ref base.st ...] in1.st [in2.st ...]
./caiw d out.caiw outdir/ [-j N] [--ref base.st ...]   # same --ref set+order as c
./caiw v out.caiw [-j N] [--ref base.st ...] in1.st [...]
```

- `c` compress, `d` decompress to safetensors, `v` verify against sources — BIDIRECTIONAL: every archive tensor must exist in the source with equal dtype+shape+bytes, every source tensor must appear in the archive (MISS-SRC), no duplicates (DUP), file set and `__metadata__` compared positionally (FILE/META).
- `-j N` threads (`-jN` also accepted; `-j` must be followed by digits). Default = online cores capped at 8, clamp 1..64. `-j 1` = fully sequential. Archives are thread-count independent on the decode side: an archive written at any -j decodes to identical bytes at any -j. Encoded bytes themselves may differ across -j (batch boundaries change which model state each tensor sees).
- `--ref` repeatable; best external reference is chosen per tensor by sampled escape rate and its file index is stored in the archive. Decode requires the same `--ref` files in the same order (fails loudly otherwise).

## Format

`CAI5` magic (decoder/verifier also accept `CAI3`/`CAI4`). File table: `[u32 basename_len][basename][u32 meta_len][meta]` — `meta` is the source `__metadata__` object stored verbatim (absent in CAI3/4). Per-tensor record: name (raw bytes), dtype, shape (nd<=64), file idx, method byte, len, ref (intra idx for REF/DELTA, ref-file idx for DELTAX), payload len, CRC32, payload. Methods (low 6 bits): RAW PACK REF FIELD FIELDPOS FIELDROW DELTA U8 F32 DELTAX.

`safetensors` input parsing is real JSON, not substring matching: `jstr` decodes all escapes incl. `\uXXXX` (surrogate pairs -> UTF-8), `jspan` skips balanced objects/arrays string-aware, `jget` walks key:value pairs with proper value skipping. Tensor names are stored UNescaped in the archive and re-escaped (`jesc`) when `d` regenerates headers — escaped names like `a\"b` round-trip through the official loader. Enforced at load AND at archive-decode: `product(shape) * dtype_size == len` (rejects >64 dims, shape/len mismatch, negative dims); duplicate tensor names within a file are rejected.

Method byte flags (CAI4+):
- `0x80` MF_BAT: tensor belongs to a parallel batch — decoded against the batch-start entropy snapshot.
- `0x40` MF_BH: first member (head) of a batch. Head = 0x80|0x40, members = 0x80, solo = no flags.
- Batches: encoder groups consecutive tensors whose every intra-archive ref (dup target + all DELTA candidates) predates the batch (`joinable()`); workers encode against a snapshot of committed hist state, then hist deltas merge (`worker += worker - snapshot`, counts are additive/order-free). Decoder rebuilds identical snapshots at MF_BH boundaries. Orphan member / ref into open batch = hard error.

## Test / regression

- `./test.sh` — one-shot regression: all fixtures `-j1`/`-j4` round-trip + DELTAX + `out/fuzz/` crash corpus must die cleanly. `./fuzz.sh` — archive-mutation fuzz.
- `./test.sh` — self-contained regression: generates fixtures via `gen_fixtures.py` (pure Python, no numpy), exercises `-j1`/`-j4`, checks `0 bad`, metadata round-trip, incomplete-source `v` failure, corpus crashes.
- `./fuzz.sh` — archive mutation fuzzer (expect clean `die`, no signals).
- Synthetic fixtures in `out/` (generated): `edgeA/edgeB.st` (all dtypes, 0-dim, empty, dups, delta pairs), `ch*.st` (3-checkpoint f16 chain — DELTA must win), `f32A/B.st` (f32 pair), `exp.st` (10-tensor MoE-like), `pack1.st` (k=1 PACK — ib=0 edge), `escname.st` (quote/backslash/unicode names), `ndim9.st` (rank-9), `meta.st` (`__metadata__`), `split.st` (64 small bf16 tensors — teach/cold-start), `dx*.st` (DELTAX), `olmoA/B.st` (956MB real F32 pair, generated locally only).
- Models in `models/` (HF safetensors, local only). Verified: pythia pair (332t), MoE shard (1063t), qwen3-4b (398t), OLMo F32 pair (173t, DELTAX 170).

## Notes for contributors

- All methods compete by *actual encoded size*; never gate on estimates alone. Pre-screening exists (pos_gain>0.06, sampled DELTAX escape rate) so it is not an exhaustive minimum over all candidates — document any addition.
- DELTA16 residual coding is conditioned on `ref>>9` (128 ctxs); DELTA32 XOR planes conditioned on `ref>>23` (256 ctxs). Context tables normalize only contexts used in the block (used-mask), decode side reconstructs the same set.
- Parallel candidate trials: each candidate encodes once against a clone of the same model state; smallest real size wins; only winner's hist delta commits — output identical to sequential competition. Trial buffers fold into a rolling best (`nsp` concurrent scratch + 1 retained), so TRYBUD is a real concurrent bound.
- Cold start: RAW/PACK winners emit no channel stream, so `teach()` counts their elements into the natural channel (CBF/CFP/C32/C8) on BOTH sides — encode (post-commit) and decode (post-payload). Any future streamless method must join `teach()` or document why not.
- Model histograms are `uint64_t` (no periodic rescale — rescale would desync the batch delta-merge). In-block cum/sym frequency tables stay u32/u16 (bounded by TOT=2^15).
- mmap everywhere: keep offsets page-aligned; drop consumed pages with `drop_pages`.
- DELTA16 escape values are coded through the shared FIELD channel (measured better than a dedicated channel).
- Cold start: RAW/PACK winners carry no rANS streams, so without help a run of small tensors would keep the float channels at their all-ones init forever. `teach()` counts every RAW/PACK winner's elements into its natural channel (CBF/CFP/C32/C8) on BOTH sides — the decode replay is identical, so the model warms up even when RAW keeps winning. (Split-tensor fixt: 64x256x256 bf16 went from ~100% RAW to ratio 0.660, beating the single-tensor 0.666.)
- Candidate trials keep only the rolling best (buffer + hist clones), so peak try memory is ~(nsp+1) x ebound — `TRYBUD` now actually bounds concurrent scratch. Note ebound is VIRTUAL reservation; resident use tracks real encoded size.
- Histogram cells are `uint64_t` — no rescale needed; `hmerge` deltas stay exact for any archive size. `cum` prefix tables stay u32 (sums <= TOT=2^15).
- Batching costs ~0.8% ratio on pythia pair (cold-start model state per batch) — inherent trade for parallelism. Tensors with promising in-batch refs are kept solo (`joinable`) so no DELTA/REF win is ever dropped.
- CAI5 vs CAI4: new archives preserve `__metadata__`; CAI3/CAI4 archives still decode but old `v` reports META when the source has metadata — that archive genuinely lost it.
- Host has ~8GB RAM — avoid loading multi-GB tensors into anonymous memory; numpy probes must use memmap + bounded samples.

## Hardening / verification status (2026-09)

Malformed-input contract: every archive/loader failure exits(1) with a `die()` message — never crashes, never OOB, never silent corruption (CRC32 verified on every tensor; partial `d` outputs are unlinked via `atexit` cleanup).

Decoder trust boundary checks (all present — do not remove):
- Record walk: `BND` before every field; `nd<=64`, `file<nf`, `ref<i`, `method<=M_DELTAX`, `plen` within buffer; `prod(shape)*dtb==len` enforced on records (decode AND verify); `MF_BAT` orphan-member rejection; REF/DELTA member may not ref into its own batch (`ref >= bstart` → die) — parallel decode would otherwise read a not-yet-decoded (NULL) referent.
- `dec_tensor`: method/dtype coherence — FIELD* need bsz==2, F32 needs bsz==4, DELTA/DELTAX need bsz∈{2,4}, PACK needs bsz<=2, RAW needs plen==len (prevents typed-write heap overflow on crafted dtype+method pairs, and RAW over-reads).
- Payload decoders take `lim` (payload end): `read_blk` validates 12-byte block header + block len; `pack_dec` validates k∈[1,256], dict bytes, `idx<k`, and exact bit-position bounds (`in + ((bit+ib-1)>>3) < lim` — a looser check permits a 1-byte over-read at payload end).
- `st_load`: real JSON string/object handling (escapes, nested braces, surrogate pairs); header len ≤ 2^30; name ≤ ~600B, dtype ≤ 64B, shape ≤ ~4KB/64 dims; `b>=a`, `off+len` non-wrapping, `prod(shape)*dtb(dtype)==len`, dup names rejected, and `dsz > filesize - 8 - hl` via fstat — subtraction form, because `8+hl+dsz` itself wraps on near-UINT64_MAX data_offsets (AFL-found real SIGBUS).
- `pos_enc`/`pos_dec`/`pos_gain`: `P<=0` rejected AND `D=n/P` clamped to >=1 — a degenerate shape with P>n gave D=0 → `gi/D` SIGFPE (AFL-found).
- Encode output buffer: `ebound(len)=4*len+len/1024+64MB` is a proven upper bound — ≤2 emitted bytes per enc() call; FIELD does 3 calls/elem (3*len); DELTA16 escapes recode through f16_enc (len+3*len=4*len — the max); DELTA32/F32 ≤2.5*len. Trial encodes write the full stream regardless of winner, so under-sizing overflows on adversarial inputs even when the method loses.
- Alignment: `alview()` supplies aligned copies for odd-offset safetensors data; DELTAX ref tensor is alview'd on the decode side too; sampling paths use memcpy loads.
- rANS state invariant: x ∈ [LOWER, 256·LOWER) at every enc/dec boundary (blocks init x=LOWER; enc preserves it — verified exhaustively). Encoder scratch is SCRSZ=BLK*16+64 (worst case 3 emitted bytes/enc call, ≤5 calls/elem).
- `out/fuzz/` keeps AFL-discovered crash inputs as a permanent regression corpus; `test.sh` runs the full suite (round-trips -j1/-j4 + DELTAX + crash corpus must-die-clean).

Verification performed:
- Sanitizers: ASan+UBSan clean (only ~KB exit-time leaks), MSan clean, TSan clean incl. Pythia pair (332t, all batch/DELTA/REF paths, byte-identical archive).
- Fuzzing: AFL++ (coverage-guided) found 5 real crashes on the *st_load/compete* side — 3× SIGFPE (`pos_enc` D=0), 2× SIGBUS (fstat wrap) — all fixed and kept in `out/fuzz/`. Post-fix re-run: ~130k execs, 0 crashes. Plus 600 archive-mutation + 400 sanitizer-mutation iters — zero crashes; all corrupt inputs die cleanly.
- CBMC 6.6: kmap16 bijection PROVEN. norm_ctx/dsym/rANS properties time out on 64-bit division bit-blasting → replaced by exhaustive C tests: 524,537-case boundary sweep for enc/dec round-trip (every f × every 256-power boundary — complete coverage since byte counts change only there; also asserts rp==end exact consumption) and 3,264 histograms × all 32,768 v for dsym — all pass.
- Static analyzers: gcc -fanalyzer triaged (real bugs fixed), scan-build-19 (post-fix run: only exit-path leak notes + escbuf size-0 FP — alloc is ≥2KB always), clang-19 --analyze (remaining items are analyzer FP — untracked invariants), cppcheck 2.17.1 clean (user-space patched binary; std.cfg via /tmp/cfg symlink).
- Valgrind 3.26 (user-space build): static-linked caiw runs clean — all reports are the known static-glibc startup FP class (`__strcmp_avx2` reading vector-width chunks incl. uninit tail bytes of argv/execfn buffers, which valgrind's dynamic-only interceptors can't suppress). MSan cross-confirms zero uninit flow in caiw code.

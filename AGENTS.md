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

- `c` compress, `d` decompress to safetensors, `v` verify against sources (byte-exact + CRC).
- `-j N` threads (`-jN` also accepted; `-j` must be followed by digits). Default = online cores capped at 8, clamp 1..64. `-j 1` = fully sequential. Archives are thread-count independent: any -j decodes any archive.
- `--ref` repeatable; best external reference is chosen per tensor by sampled escape rate and its file index is stored in the archive. Decode requires the same `--ref` files in the same order (fails loudly otherwise).

## Format

`CAI4` magic (decoder/verifier also accept `CAI3`). Per-tensor record: name, dtype, shape, file idx, method byte, len, ref (intra idx for REF/DELTA, ref-file idx for DELTAX), payload len, CRC32, payload. Methods (low 6 bits): RAW PACK REF FIELD FIELDPOS FIELDROW DELTA U8 F32 DELTAX.

Method byte flags (CAI4 only):
- `0x80` MF_BAT: tensor belongs to a parallel batch — decoded against the batch-start entropy snapshot.
- `0x40` MF_BH: first member (head) of a batch. Head = 0x80|0x40, members = 0x80, solo = no flags.
- Batches: encoder groups consecutive tensors whose every intra-archive ref (dup target + all DELTA candidates) predates the batch (`joinable()`); workers encode against a snapshot of committed hist state, then hist deltas merge (`worker += worker - snapshot`, counts are additive/order-free). Decoder rebuilds identical snapshots at MF_BH boundaries. Orphan member / ref into open batch = hard error.

## Test / regression

- `./test.sh` — one-shot regression: all fixtures `-j1`/`-j4` round-trip + DELTAX + `out/fuzz/` crash corpus must die cleanly. `./fuzz.sh` — archive-mutation fuzz.
- `./caiw c out/x.caiw <files> && ./caiw v out/x.caiw <files>` — expect `0 bad`.
- Exercise both `-j 1` and `-j 8`; also run `d` mode (not just `v`) and compare tensor data regions (regenerated JSON headers differ — full-file cmp is invalid).
- Synthetic fixtures in `out/`: `edgeA/edgeB.st` (all dtypes, 0-dim, empty, dups, delta pairs), `ch*.st` (3-checkpoint f16 chain — DELTA must win), `f32A/B.st` (f32 pair), `exp.st` (10-tensor MoE-like), `olmoA/B.st` (956MB real F32 pair).
- Models in `models/` (HF safetensors). Verified CAI4+threads: pythia pair (332t), MoE shard (1063t), qwen3-4b (398t), OLMo F32 pair. CAI3 archives verified readable: moe3.caiw, q4b.caiw.

## Notes for contributors

- All methods compete by *actual encoded size*; never gate on estimates alone.
- DELTA16 residual coding is conditioned on `ref>>9` (128 ctxs); DELTA32 XOR planes conditioned on `ref>>23` (256 ctxs). Context tables normalize only contexts used in the block (used-mask), decode side reconstructs the same set.
- Parallel candidate trials: each candidate encodes once against a clone of the same model state; smallest real size wins; only winner's hist delta commits — output identical to sequential competition.
- mmap everywhere: keep offsets page-aligned; drop consumed pages with `drop_pages`.
- DELTA16 escape values are coded through the shared FIELD channel (measured better than a dedicated channel).
- Batching costs ~0.8% ratio on pythia pair (cold-start model state per batch) — inherent trade for parallelism. Tensors with promising in-batch refs are kept solo (`joinable`) so no DELTA/REF win is ever dropped.
- Host has ~8GB RAM — avoid loading multi-GB tensors into anonymous memory; numpy probes must use memmap + bounded samples.

## Hardening / verification status (2026-09)

Malformed-input contract: every archive/loader failure exits(1) with a `die()` message — never crashes, never OOB, never silent corruption (CRC32 verified on every tensor; partial `d` outputs are unlinked via `atexit` cleanup).

Decoder trust boundary checks (all present — do not remove):
- Record walk: `BND` before every field; `nd<=8`, `file<nf`, `ref<i`, `method<=M_DELTAX`, `plen` within buffer; `MF_BAT` orphan-member rejection; REF/DELTA member may not ref into its own batch (`ref >= bstart` → die) — parallel decode would otherwise read a not-yet-decoded (NULL) referent.
- `dec_tensor`: method/dtype coherence — FIELD* need bsz==2, F32 needs bsz==4, DELTA/DELTAX need bsz∈{2,4}, PACK needs bsz<=2, RAW needs plen==len (prevents typed-write heap overflow on crafted dtype+method pairs, and RAW over-reads).
- Payload decoders take `lim` (payload end): `read_blk` validates 12-byte block header + block len; `pack_dec` validates k∈[1,256], dict bytes, `idx<k`, and exact bit-position bounds (`in + ((bit+ib-1)>>3) < lim` — a looser check permits a 1-byte over-read at payload end).
- `st_load`: header len ≤ 2^30, `b>=a`, `off+len` non-wrapping, and `dsz > filesize - 8 - hl` via fstat — subtraction form, because `8+hl+dsz` itself wraps on near-UINT64_MAX data_offsets (AFL-found real SIGBUS).
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

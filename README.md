# caiw — Compressor for AI Weights

**Lossless, byte-exact archiver for safetensors model weights.**
One C file. No dependencies beyond libc + pthread. Multithreaded.

Generic compressors see model weights as opaque bytes. caiw understands
the bit structure of `F16`/`BF16`/`F32` tensors — sign, exponent and
mantissa fields — plus cross-tensor structure: duplicates, checkpoint
deltas, external reference weights. Every tensor is compressed with
whichever method *actually produces the fewest bytes*, and every archive
reconstructs its inputs bit-for-bit (CRC32-verified).

## Results

Qwen3-4B shard 3 (BF16, 99.6 MB), `-j8`, this machine:

| tool | size | ratio | wall time |
|---|---|---|---|
| **caiw** | **65.78 MB** | **0.660** | **3.2 s** |
| xz -9e | 70.15 MB | 0.704 | 85.9 s |
| zstd -19 -T8 | 76.52 MB | 0.768 | 26.4 s |
| gzip -9 | 79.22 MB | 0.795 | 7.8 s |

Full Qwen3-4B (8.0 GB, 3 shards, 398 tensors): **5.32 GB (0.662)**,
encoded in 4m40s, verified `0 bad`.

MoE shard (4.0 GB, 1063 tensors): **2.65 GB (0.663)**, ~2 min encode.
OLMo-2 F32 pair (5.0 GB) against its base snapshot via `--ref`:
**2.79 GB (0.560)** — 170 of 173 tensors resolved as `DELTAX`.

## Quick start

```console
$ make                      # cc -O3 -march=native, pthread + libm only
$ ./caiw c model.caiw model.st -j8        # compress
$ ./caiw v model.caiw model.st -j8        # verify byte-exact
$ ./caiw d model.caiw outdir/ -j8         # decompress to safetensors
```

Checkpoint deltas (e.g. store a fine-tune against its base):

```console
$ ./caiw c tune.caiw --ref base.st tune.st -j8
$ ./caiw d tune.caiw outdir/ --ref base.st   # same --ref set, same order
```

`make test` generates deterministic fixtures and runs the full
regression (round-trips at -j1/-j4, metadata, and a checked-in
crash corpus that must die cleanly).

## How it works

**Competition, not heuristics.** Every candidate encoder runs against a
clone of the same entropy-model state; the smallest *real* output wins.
Cheap integer estimates may *prune* a hopeless candidate before it runs —
they never pick the winner — and if nothing helps, RAW wins.

| method | exploits |
|---|---|
| `FIELD` | sign/exponent/mantissa of f16-family, context-conditioned rANS |
| `FIELDPOS`/`FIELDROW` | + column/row-position context (embeddings, rotary tables) |
| `F32` | 5-plane field decomposition for f32 |
| `DELTA` | residual vs an earlier same-name/family tensor (checkpoint chains) |
| `DELTAX` | residual vs an external `--ref` checkpoint |
| `PRW` | residual vs the previous row of the same tensor (smooth rows) |
| `REF` | content-hash exact dedup |
| `PACK` | restricted alphabets (≤256 distinct atoms → bit-packed indices) |
| `U8` | per-byte entropy fallback |
| `RAW` | store — the honest floor |

**Parallelism without nondeterminism.** Candidate trials and consecutive
tensors encode in parallel (batches, `CAI5`): each worker sees a snapshot
of the committed histogram state, histogram deltas merge commutatively,
and the archive records batch boundaries so an archive written at *any*
`-j` decodes to identical bytes under *any* `-j`.

**Rolling context models.** Each channel keeps a cumulative histogram;
per-block tables are normalized only over contexts the block actually
uses (used-mask), which the decoder reconstructs identically from the
already-decoded reference tensor.

## Correctness and robustness

- **Byte-exact by construction** — `v` re-derives every tensor and
  compares bytes + CRC32 in both directions (archive vs source, incl.
  dtype/shape/`__metadata__`); `d` restores every tensor and the
  verbatim metadata object.
- **Hostile-input safe** — every malformed archive or truncated
  safetensors file exits(1) through `die()`; the decoder validates every
  field, method/dtype pair, payload bound, and batch reference before
  touching memory.
- **Verified** — ASan/UBSan/MSan/TSan clean; a checked-in crash
  corpus (`tests/corpus/`) replays inputs that once crashed. CI
  (`ci.yml`) rebuilds from a clean Ubuntu image and reruns `test.sh`
  under -O3/-Werror/ASan+UBSan and on real aarch64, plus every
  `verify.sh` layer — TLA+/Alloy/CBMC in the `verify` job, Frama-C
  EVA+WP + Rocq proofs in `verify-framac` (pinned opam switch +
  pinned z3). Formal layer under `verify/`
  (`verify/verify.sh`): TLA+ model-checks the batch-merge and
  tmp→rename protocols (flawed variants produce the expected
  counterexamples), Alloy checks archive structural invariants at
  scope 8, CBMC proves kmap16/dsym/norm_ctx/rANS (fixed-frequency)
  plus the parser boundary (utf8_ok/hex4/jstr/jstr_skip/jspan),
  Frama-C EVA+RTE screens the codec kernels (0 invalid), and Frama-C
  WP discharges the ACSL contracts on those kernels — 116/116 goals,
  an unbounded proof per contract. Five Rocq/Coq proofs (zero-axiom,
  coqchk-rechecked) cover the regions every BMC backend timed out on:
  rANS round trip over the full general domain, utf8_ok correctness
  at any length, norm_ctx's output contract over all histograms,
  PACK bitstream round trip at any length, and jkey's nested scan
  (bounds, termination, key-position correctness). Bounded results
  and solver limits are documented honestly; the rANS core is
  additionally covered by an exhaustive boundary sweep (524,537
  cases) and `dsym` by 3,264 tables × all 32,768 symbol values.
- **Deterministic** — encode decisions are integer-only, so a fixed `-j`
  reproduces bit-identical archives across machines; archives are
  `-j`-independent on the decode side.

See `AGENTS.md` for the format specification, the decoder trust-boundary
checklist, and the full verification log.

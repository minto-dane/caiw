module caiw_archive_flawed

/*
 * Structural invariants of a caiw archive (decoder trust boundary),
 * modeled from caiw.c record-walk checks (~2540-2660, ~2780-2900):
 *
 *   - intra-archive refs (REF/DELTA) point strictly backward (ref < i)
 *   - DELTA referent dtype == member dtype
 *   - DELTAX uses an external ref-file index, never an intra ref
 *   - batch flags canonical: head = BAT|BH, member = BAT, solo = none
 *     (BH without BAT is rejected in the code — "bad method flags")
 *   - batch members may not ref into their own open batch
 *     ("ref into open batch" — referent wouldn't be decoded yet)
 *   - batch members require a preceding head ("orphan batch member")
 *   - (file,name) unique ("dup tensor in archive")
 *   - method/dtype coherence subset: FIELD* need 2-byte dtypes,
 *     F32 needs 4, DELTA/DELTAX need 2 or 4, PACK needs <=2
 *   - reserved name __metadata__ forbidden as a tensor name
 */

open util/ordering[Record] as ord

sig Name {}
one sig MetaName extends Name {}        -- reserved "__metadata__"
sig File {}
sig Width {}                            -- element width abstraction
one sig WSub, W1, W2, W4 extends Width {}
sig Dtype { width: one Width }

abstract sig Method {}
one sig RAW, PACK, REF, FIELD, FIELDPOS, FIELDROW, DELTA, U8, F32, DELTAX
    extends Method {}

abstract sig Bat {} one sig Solo, Head, Member extends Bat {}

sig Record {
    file:  one File,
    name:  one Name,
    dtype: one Dtype,
    meth:  one Method,
    ref:   lone Record,        -- intra-archive referent (REF/DELTA)
    xfile: lone File,          -- external ref-file index (DELTAX)
    bat:   one Bat,
    bhead: lone Record         -- for Member: its batch's head record
}

-- ---------------- decoder acceptance facts (what caiw.c enforces) --------

// FLAW: BackwardRefs fact removed

fact RefPresence {
    all r: Record | (some r.ref) iff (r.meth in REF + DELTA)
}

fact DeltaxPresence {
    all r: Record | (some r.xfile) iff (r.meth = DELTAX)
}

fact DeltaDtypeMatch {
    all r: Record | r.meth = DELTA implies r.ref.dtype = r.dtype
}

fact CanonicalFlags {
    -- BH-without-BAT is impossible by construction (bat is a 3-way enum:
    -- Solo, Head, Member). Members and heads know their batch:
    all r: Record | r.bat = Member implies r.bhead.bat = Head
    all r: Record | r.bat = Member implies ord/lt[r.bhead, r]
    -- batch is contiguous: no Head strictly between a member and its head
    all r: Record | r.bat = Member implies
        (no h: Record | h.bat = Head and ord/lt[r.bhead, h] and ord/lt[h, r])
}

// FLAW: NoBatchSelfRef fact removed

fact SoloRefsAnywhereBefore {
    -- solo/head records may ref anything before them (already BackwardRefs)
    all r: Record | (r.bat = Head and some r.ref) implies ord/lt[r.ref, r]
}

fact UniqueFileName {
    no disj a, b: Record | a.file = b.file and a.name = b.name
}

fact NoMetaName {
    all r: Record | r.name != MetaName
}

fact MethodDtypeCoherence {
    all r: Record | r.meth in (FIELD + FIELDPOS + FIELDROW) implies r.dtype.width = W2
    all r: Record | r.meth = F32 implies r.dtype.width = W4
    all r: Record | r.meth in (DELTA + DELTAX) implies r.dtype.width in (W2 + W4)
    all r: Record | r.meth = PACK implies r.dtype.width in (WSub + W1 + W2)
}

-- ------------------------- assertions ------------------------------------

-- ref chains are acyclic (follows from strict backward ordering)
assert AcyclicRefs { no r: Record | r in r.^ref }

-- batch member never references its own batch
assert NoIntraBatchRef {
    all r: Record | r.bat = Member implies
        (no r.ref or ord/lt[r.ref, r.bhead])
}

-- duplicate (file,name) impossible
assert DupFree {
    no disj a, b: Record | a.file = b.file and a.name = b.name
}

-- member has a head (no orphan members)
assert NoOrphanMember {
    all r: Record | r.bat = Member implies
        (some r.bhead and r.bhead.bat = Head and ord/lt[r.bhead, r])
}

check AcyclicRefs for 8
check NoIntraBatchRef for 8
check DupFree for 8
check NoOrphanMember for 8

-- ------------------------- emittable subset -------------------------------

-- What the encoder can emit is a strict subset of what the decoder accepts.
-- Representative asymmetries (from caiw.c encoder):
--   * RAW/PACK/REF/... payload-bearing methods are all emittable
--   * encoder never emits FIELD on non-W2, never emits U8 on W2-float...
--     model one asymmetry: U8 is emittable on any dtype, but FIELD family
--     only on W2; encoder emits RAW for anything it cannot compress.
pred Emittable[r: Record] {
    r.meth in (FIELD + FIELDPOS + FIELDROW) implies r.dtype.width = W2
    r.meth = F32 implies r.dtype.width = W4
    r.meth in (DELTA + DELTAX) implies r.dtype.width in (W2 + W4)
    r.meth = PACK implies r.dtype.width in (WSub + W1 + W2)
    (some r.ref) iff (r.meth in REF + DELTA)
    (some r.xfile) iff (r.meth = DELTAX)
    r.meth = DELTA implies r.ref.dtype = r.dtype
    (some r.ref) implies ord/lt[r.ref, r]
    r.bat = Member implies (some r.ref implies ord/lt[r.ref, r.bhead])
    r.name != MetaName
}

-- every encoder-emittable record is decoder-acceptable (given global facts)
assert EncSubsetDec {
    all r: Record | Emittable[r] implies
        (r.meth in (FIELD + FIELDPOS + FIELDROW) implies r.dtype.width = W2)
}
check EncSubsetDec for 8

-- find over-acceptance surface: archives the decoder accepts that the
-- encoder would never emit (informs the attack-surface documentation)
run OverAccept {
    some r: Record | not Emittable[r]
} for 4

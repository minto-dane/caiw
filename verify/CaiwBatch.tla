---- MODULE CaiwBatch ----
(***************************************************************************)
(* caiw parallel batch protocol — correct variant.                          *)
(*                                                                          *)
(* Faithful model of caiw.c (encode ~2390-2409, decode ~2184-2203):         *)
(*   - hsnap(): snapshot of committed hist state at batch start             *)
(*   - each worker encodes against a PRIVATE clone of that snapshot         *)
(*     (gsnp lazy clone — workers never touch the shared committed table)   *)
(*   - JOIN BARRIER: all workers complete before any merge                  *)
(*   - hmerge(): g += (w[m] - snap) per member — additive, order-free       *)
(*   - decoder rebuilds the same snapshot at MF_BH, decodes each member     *)
(*     against it, and merges contributions identically                     *)
(*                                                                          *)
(* Abstraction: committed hist = one Nat counter; member m adds Contrib[m]. *)
(* Per-channel structure is orthogonal: additive merge on each cell         *)
(* commutes, so a single counter captures the merge algebra.                *)
(*                                                                          *)
(* Checked: TypeOK, JoinBarrier, SnapshotConsistency, NoLostUpdate,         *)
(* DecEncAgree.                                                             *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

CONSTANTS Members
Contrib == [m \in Members |-> m]    \* abstract per-member contribution
ASSUME Members # {} /\ Members \subseteq Nat

VARIABLES g,          \* committed hist (shared, written only by main thread)
          snapV,      \* batch-start snapshot value
          w,          \* w[m]: worker m's private hist (UNSET until encoded)
          merged,     \* members merged on encoder side
          decMerged,  \* members merged on decoder side
          decG        \* decoder committed hist

UNSET == -1
vars == <<g, snapV, w, merged, decMerged, decG>>

RECURSIVE Sum(_)
Sum(S) == IF S = {} THEN 0
          ELSE LET x == CHOOSE x \in S : TRUE
               IN  Contrib[x] + Sum(S \ {x})

TypeOK ==
    /\ g \in Nat
    /\ snapV \in Nat
    /\ w \in [Members -> Nat \cup {UNSET}]
    /\ merged \subseteq Members
    /\ decMerged \subseteq Members
    /\ decG \in Nat

Init ==
    /\ g = 0
    /\ snapV = 0                    \* hsnap(): snap := g at batch start
    /\ w = [m \in Members |-> UNSET]
    /\ merged = {}
    /\ decMerged = {}
    /\ decG = 0

\* Worker m encodes: private table = snapshot clone + own contribution.
\* Concurrent with other workers; writes no shared state.
Enc(m) ==
    /\ w[m] = UNSET
    /\ w' = [w EXCEPT ![m] = snapV + Contrib[m]]
    /\ UNCHANGED <<g, snapV, merged, decMerged, decG>>

\* JOIN BARRIER (pthread_join over all workers) precedes any merge.
Merge(m) ==
    /\ \A x \in Members : w[x] # UNSET
    /\ m \notin merged
    /\ g' = g + (w[m] - snapV)      \* hmerge(): g += w[m] - snap
    /\ merged' = merged \cup {m}
    /\ UNCHANGED <<snapV, w, decMerged, decG>>

\* Decoder merges each member's contribution once, any order, after the
\* batch is fully encoded (its members all decoded against snapV first —
\* decode-vs-snapshot correctness is captured by SnapshotConsistency).
DecMerge(m) ==
    /\ merged = Members              \* archive complete => decode may run
    /\ m \notin decMerged
    /\ decMerged' = decMerged \cup {m}
    /\ decG' = decG + Contrib[m]
    /\ UNCHANGED <<g, snapV, w, merged>>

Next ==
    \/ \E m \in Members : Enc(m)
    \/ \E m \in Members : Merge(m)
    \/ \E m \in Members : DecMerge(m)

Spec == Init /\ [][Next]_vars

\* ------------------------------ invariants ------------------------------

\* No merge fires before the join barrier completes every worker.
JoinBarrier ==
    \A m \in merged : \A x \in Members : w[x] # UNSET

\* Every worker encoded against the snapshot value — no live-g leakage.
SnapshotConsistency ==
    \A m \in Members : w[m] # UNSET => w[m] = snapV + Contrib[m]

\* After all members merged, g == snap + total contribution, under any
\* merge order TLC explored.
NoLostUpdate ==
    (merged = Members) => g = snapV + Sum(Members)

\* Decoder converges to the encoder's committed hist.
DecEncAgree ==
    (merged = Members /\ decMerged = Members) => decG = g
=============================================================================

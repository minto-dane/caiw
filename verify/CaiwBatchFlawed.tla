---- MODULE CaiwBatchFlawed ----
(***************************************************************************)
(* Flawed variant: NO per-worker snapshot clone — every worker adds its     *)
(* contribution straight into the shared committed hist g with a nonatomic *)
(* read-modify-write. Two workers can read the same g and both write back,  *)
(* losing one update.                                                       *)
(*                                                                          *)
(* This is the hazard the real design prevents: caiw gives each worker a    *)
(* private snapshot clone (gsnp) and merges deltas AFTER the join barrier.  *)
(* TLC should find a trace where g < Sum(Contrib) — a lost update.          *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

CONSTANTS Members
Contrib == [m \in Members |-> m]    \* abstract per-member contribution
ASSUME Members # {} /\ Members \subseteq Nat

VARIABLES g,        \* shared committed hist — workers write it directly
          rd        \* rd[m]: value worker m read from g (UNSET = not started)

UNSET == -1
DONE  == -2
vars == <<g, rd>>

RECURSIVE Sum(_)
Sum(S) == IF S = {} THEN 0
          ELSE LET x == CHOOSE x \in S : TRUE
               IN  Contrib[x] + Sum(S \ {x})

TypeOK ==
    /\ g \in Nat
    /\ rd \in [Members -> Nat \cup {UNSET, DONE}]

Init ==
    /\ g = 0
    /\ rd = [m \in Members |-> UNSET]

\* Step 1 of nonatomic increment: read shared g into a worker-local temp.
Read(m) ==
    /\ rd[m] = UNSET
    /\ rd' = [rd EXCEPT ![m] = g]
    /\ UNCHANGED g

\* Step 2: write back rd[m] + Contrib[m]. Between Read and Write another
\* worker's Write may land — this read's value is then stale.
Write(m) ==
    /\ rd[m] \in Nat                  \* holds a read value, not UNSET/DONE
    /\ g' = rd[m] + Contrib[m]
    /\ rd' = [rd EXCEPT ![m] = DONE]
    
Next ==
    \/ \E m \in Members : Read(m)
    \/ \E m \in Members : Write(m)

Spec == Init /\ [][Next]_vars

\* All workers finished but the total is short: at least one update lost.
NoLostUpdate ==
    (\A m \in Members : rd[m] = DONE) => g = Sum(Members)
=============================================================================

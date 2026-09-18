---- MODULE CaiwOutputFlawed ----
(***************************************************************************)
(* Flawed variant: outputs written DIRECTLY to final paths (no tmp+rename), *)
(* and die() leaves partial files behind (no cleanup).                      *)
(*                                                                          *)
(* TLC should violate both NoPartialFinal (a final path exists while its    *)
(* content is still incomplete) and AllOrNothing (die leaves artifacts).    *)
(* This is the hazard tmp+rename + tracked cleanup prevents.                *)
(***************************************************************************)
EXTENDS FiniteSets

CONSTANTS Outs
ASSUME Outs # {}

VARIABLES st,        \* st[o] \in {"none","wr","final","gone"} — wr = writing FINAL
          died, ok

vars == <<st, died, ok>>
Phases == {"none", "wr", "final", "gone"}

TypeOK ==
    /\ st \in [Outs -> Phases]
    /\ died \in BOOLEAN
    /\ ok \in BOOLEAN

Init ==
    /\ st = [o \in Outs |-> "none"]
    /\ died = FALSE
    /\ ok = FALSE

\* FLAW: open the FINAL path directly — a reader sees partial content.
OpenFinal(o) ==
    /\ ~died /\ ~ok
    /\ st[o] = "none"
    /\ st' = [st EXCEPT ![o] = "wr"]
    /\ UNCHANGED <<died, ok>>

Finish(o) ==
    /\ ~died /\ ~ok
    /\ st[o] = "wr"
    /\ st' = [st EXCEPT ![o] = "final"]
    /\ UNCHANGED <<died, ok>>

Commit ==
    /\ ~died /\ ~ok
    /\ \A o \in Outs : st[o] = "final"
    /\ ok' = TRUE
    /\ UNCHANGED <<st, died>>

\* FLAW: die() cleans nothing — partial/final files stay on disk.
Die ==
    /\ ~died /\ ~ok
    /\ died' = TRUE
    /\ UNCHANGED <<st, ok>>

Next ==
    \/ \E o \in Outs : OpenFinal(o)
    \/ \E o \in Outs : Finish(o)
    \/ Commit
    \/ Die

Spec == Init /\ [][Next]_vars

\* A final-visible path exists only with complete content — "wr" violates.
NoPartialFinal ==
    \A o \in Outs : st[o] \notin {"wr"}

\* After die(), nothing remains — cleanup forgot everything, so any
\* st[o] \notin "gone" violates.
AllOrNothing ==
    died => \A o \in Outs : st[o] = "gone"
=============================================================================

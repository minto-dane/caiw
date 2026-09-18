---- MODULE CaiwOutput ----
(***************************************************************************)
(* caiw d-mode output protocol — correct variant (tmp -> rename).          *)
(*                                                                          *)
(* Faithful model of caiw.c ~2620-2710:                                     *)
(*   - each output is written to <name>.caiwtmp via xfopen_tmp              *)
(*     (O_NOFOLLOW|O_NONBLOCK — symlink/FIFO plants refused)                *)
(*   - g_outp[] tracks every tmp path from open time                        *)
(*   - all writes complete + fclose before ANY rename                       *)
(*   - rename(tmp, final); g_outp[i] is then updated to the FINAL name,     *)
(*     so a later die() unlinks completed outputs too — all-or-nothing      *)
(*   - die() -> atexit(out_cleanup): unlink every tracked path              *)
(*   - success sets g_outok -> cleanup unlinks nothing                      *)
(*                                                                          *)
(* Checked invariants:                                                      *)
(*   NoPartialFinal — a final path exists only with complete content        *)
(*   AllOrNothing   — after die, no output artifact remains                 *)
(* Reachability of a successful commit is witnessed by counterexample:      *)
(* enabling INVARIANT NotCommitted makes TLC produce a committed run.       *)
(* (Unconditional <>ok is NOT claimed — Die is always enabled pre-commit.)  *)
(***************************************************************************)
EXTENDS FiniteSets

CONSTANTS Outs        \* output members, e.g. {1,2}
ASSUME Outs # {}

\* per-output phases
VARIABLES st,         \* st[o] \in {"none","tmpw","tmpd","final","gone"}
          wroteAll,   \* all tmps closed (rename stage may begin)
          died,       \* die() ran — cleanup removed everything tracked
          ok          \* g_outok — run committed successfully

vars == <<st, wroteAll, died, ok>>

Phases == {"none", "tmpw", "tmpd", "final", "gone"}

TypeOK ==
    /\ st \in [Outs -> Phases]
    /\ wroteAll \in BOOLEAN
    /\ died \in BOOLEAN
    /\ ok \in BOOLEAN

Init ==
    /\ st = [o \in Outs |-> "none"]
    /\ wroteAll = FALSE
    /\ died = FALSE
    /\ ok = FALSE

\* Open tmp + begin writing (any order across outputs, before die).
OpenTmp(o) ==
    /\ ~died /\ ~ok
    /\ st[o] = "none"
    /\ st' = [st EXCEPT ![o] = "tmpw"]
    /\ UNCHANGED <<wroteAll, died, ok>>

\* Finish writing + fclose tmp o.
CloseTmp(o) ==
    /\ ~died /\ ~ok
    /\ st[o] = "tmpw"
    /\ st' = [st EXCEPT ![o] = "tmpd"]
    /\ UNCHANGED <<wroteAll, died, ok>>

\* All tmps closed -> rename stage starts (wroteAll latch).
AllClosed ==
    /\ ~died /\ ~ok
    /\ ~wroteAll
    /\ \A o \in Outs : st[o] = "tmpd"
    /\ wroteAll' = TRUE
    /\ UNCHANGED <<st, died, ok>>

\* rename(tmp_o, final_o) — only from the rename stage.
Rename(o) ==
    /\ ~died /\ ~ok
    /\ wroteAll
    /\ st[o] = "tmpd"
    /\ st' = [st EXCEPT ![o] = "final"]
    /\ UNCHANGED <<wroteAll, died, ok>>

\* All renamed -> success commit (g_outok = 1; cleanup unlinks nothing).
Commit ==
    /\ ~died /\ ~ok
    /\ \A o \in Outs : st[o] = "final"
    /\ ok' = TRUE
    /\ UNCHANGED <<st, wroteAll, died>>

\* die() anywhere pre-commit: out_cleanup unlinks EVERY tracked path —
\* open tmps AND already-renamed finals (g_outp tracks final names too).
Die ==
    /\ ~died /\ ~ok
    /\ died' = TRUE
    /\ st' = [o \in Outs |-> "gone"]
    /\ UNCHANGED <<wroteAll, ok>>

Next ==
    \/ \E o \in Outs : OpenTmp(o)
    \/ \E o \in Outs : CloseTmp(o)
    \/ AllClosed
    \/ \E o \in Outs : Rename(o)
    \/ Commit
    \/ Die

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* ------------------------------ properties ------------------------------

\* A final path exists only for a fully written output: rename is reachable
\* solely from "tmpd" (closed tmp), so "final" implies complete content.
NoPartialFinal ==
    \A o \in Outs : st[o] = "final" => wroteAll

\* After die(), nothing remains — neither tmp files nor renamed finals.
AllOrNothing ==
    died => \A o \in Outs : st[o] = "gone"

\* Reachability witness: TLC violates this by exhibiting a successful run
\* (Die is always enabled pre-commit, so <>ok cannot hold under fairness —
\* the counterexample trace to ~ok IS the success path).
NotCommitted == ~ok
=============================================================================

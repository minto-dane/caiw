(* verify/Jkey.v — algorithm-level proof for jkey's nested scan, the last
   documented BMC solver limit: while (p < oend) over symbolic bytes with
   inner memchr / jstr / jstr_skip / jspan unwinds times out every bounded
   engine. Here the scan structure is modeled over lists of Z bytes and
   proved for ARBITRARY length.

   Model (mirrors caiw.c):
     * l    = remaining object bytes [p, oend)   — bounds-checked scans
     * rest = buffer tail past oend              — NUL-driven sub-scanners
              (jstr_skip / jspan) may legally read into it
     * dec  = jstr, ABSTRACT: returns (decoded bytes, consumed count) and
              must satisfy the consumed-count contract below. jkey's
              outer structure is proved for ANY decoder meeting it —
              decode internals stay with cbmc_parse.c (jstr bounded dst,
              hex4 exactly-4 reads) and WP (utf8_ok/hex4 contracts).

   Proved:
     * jstr_scan / jspan bounds — consumed counts stay within the slice.
     * jkey_bound — result is strictly inside the object span.
     * jkey_fuel  — under the contract the scan terminates within
                    length l outer iterations (extra fuel is irrelevant):
                    the exact property the unbounded unwinding could not
                    reach.
     * jkey_sem   — Some v implies v sits immediately after
                    quote<decoded==key>quote colon with optional whitespace:
                    the returned pointer really is the value position of
                    a key occurrence.

   Scope: algorithm-level Z/list model, same tier as Rans/Utf8/Norm/Pack —
   correspondence with the C code is by inspection. *)

From Stdlib Require Import ZArith List Bool Lia Wf_nat.
Import ListNotations.
Open Scope Z_scope.

(* eliminate set()-style local definitions, expose skipn/length structure,
   then close the arithmetic side condition *)
Ltac jsub :=
  repeat match goal with
  | x := _ |- _ => subst x
  end;
  rewrite ?length_skipn; cbn [length]; lia.

Definition is_ws (c : Z) : bool :=
  (c =? 32) || (c =? 9) || (c =? 10) || (c =? 13).

Lemma option_map_inv : forall (A B : Type) (f : A -> B) o y,
  option_map f o = Some y -> exists x, o = Some x /\ f x = y.
Proof.
  intros A B f o y H. destruct o as [x|]; simpl in H; inversion H; eauto.
Qed.

(* memchr for c within l *)
Fixpoint find_c (c : Z) (l : list Z) : option nat :=
  match l with
  | [] => None
  | x :: t => if x =? c then Some 0%nat else option_map S (find_c c t)
  end.

Lemma find_c_bound : forall c l k, find_c c l = Some k -> (k < length l)%nat.
Proof.
  induction l as [|x t IH]; simpl; intros k H; [discriminate|].
  destruct (Z.eqb x c) eqn:E.
  - inversion H. simpl. lia.
  - apply option_map_inv in H. destruct H as [k' [F Hk]].
    pose proof (IH _ F) as Hb. subst. simpl. lia.
Qed.

Lemma find_c_char : forall c l k, find_c c l = Some k -> nth k l 0 = c.
Proof.
  induction l as [|x t IH]; simpl; intros k H; [discriminate|].
  destruct (Z.eqb x c) eqn:E.
  - inversion H; subst. simpl. apply Z.eqb_eq. exact E.
  - apply option_map_inv in H. destruct H as [k' [F Hk]].
    subst. simpl. apply IH. exact F.
Qed.


(* whitespace skip: bytes consumed *)
Fixpoint ws_skip (l : list Z) : nat :=
  match l with
  | [] => 0%nat
  | x :: t => if is_ws x then S (ws_skip t) else 0%nat
  end.

Lemma ws_skip_le : forall l, (ws_skip l <= length l)%nat.
Proof.
  induction l as [|x t IH]; simpl; [lia|].
  destruct (is_ws x); simpl; lia.
Qed.

Lemma ws_skip_ws : forall l j, (j < ws_skip l)%nat -> is_ws (nth j l 0) = true.
Proof.
  induction l as [|x t IH]; simpl; intros j H; [lia|].
  destruct (is_ws x) eqn:E.
  - destruct j as [|j']; [exact E|]. apply IH. lia.
  - lia.
Qed.

Lemma ws_skip_stop : forall l, (ws_skip l < length l)%nat ->
  is_ws (nth (ws_skip l) l 0) = false.
Proof.
  induction l as [|x t IH]; simpl; intros H; [lia|].
  destruct (is_ws x) eqn:E.
  - apply IH. lia.
  - exact E.
Qed.

(* ---------------- jstr_skip model ---------------- *)
(* body scan: l = bytes AFTER the opening quote; returns bytes consumed
   through the closing quote, None on NUL/end (C: *p==0 -> return NULL) *)
Fixpoint jstr_scan (l : list Z) : option nat :=
  match l with
  | [] => None
  | c :: t =>
    if c =? 0 then None
    else if c =? 34 then Some 1%nat
    else if c =? 92 then
      match t with
      | [] => None                          (* '\\' then implicit NUL *)
      | c2 :: t2 =>
        if c2 =? 0 then None                (* '\\' NUL -> fail *)
        else option_map (Nat.add 2) (jstr_scan t2)
      end
    else option_map S (jstr_scan t)
  end.

Lemma jstr_scan_bound : forall n l k,
  (length l <= n)%nat -> jstr_scan l = Some k -> (1 <= k <= length l)%nat.
Proof.
  induction n as [n IH] using lt_wf_ind; intros l k Hln H.
  destruct l as [|c t]; simpl in H; [discriminate|].
  destruct (Z.eqb c 0); [discriminate|].
  destruct (Z.eqb c 34); [inversion H; simpl; lia|].
  destruct (Z.eqb c 92).
  - destruct t as [|c2 t2]; [discriminate|].
    destruct (Z.eqb c2 0); [discriminate|].
    apply option_map_inv in H. destruct H as [k' [F Hk]].
    pose proof (IH (length t2) ltac:(simpl in Hln; lia) t2 k'
                  ltac:(jsub) F) as Hb.
    simpl in *. lia.
  - apply option_map_inv in H. destruct H as [k' [F Hk]].
    pose proof (IH (length t) ltac:(simpl in Hln; lia) t k'
                  ltac:(jsub) F) as Hb.
    simpl in *. lia.
Qed.

(* jstr_skip: l starts AT the quote; returns bytes consumed past close *)
Definition jstr_skip_l (l : list Z) : option nat :=
  match l with
  | [] => None
  | c :: t => if c =? 34 then option_map S (jstr_scan t) else None
  end.

Lemma jstr_skip_bound : forall l k, jstr_skip_l l = Some k ->
  (2 <= k <= length l)%nat.
Proof.
  intros l k H. unfold jstr_skip_l in H.
  destruct l as [|c t]; [discriminate|].
  destruct (Z.eqb c 34) eqn:E; try discriminate.
  apply option_map_inv in H. destruct H as [k' [F Hk]].
  pose proof (jstr_scan_bound (length t) t k' ltac:(jsub) F) as Hb.
  simpl. lia.
Qed.

(* ---------------- jspan model ---------------- *)
(* fused two-mode scan: instr=true scans inside a string literal (after
   its opening quote), instr=false counts depth. l = bytes from current
   position; returns bytes consumed through the depth-0 close bracket.
   depth is Z to match C's int (a stray close at depth 0 goes negative
   and keeps scanning — unreachable for well-formed calls but modeled). *)
Fixpoint jspan_l (open close : Z) (depth : Z) (instr : bool) (l : list Z)
  : option nat :=
  match l with
  | [] => None
  | c :: t =>
    if instr then
      if c =? 0 then None
      else if c =? 92 then
        match t with
        | [] => None
        | c2 :: t2 =>
          if c2 =? 0 then None
          else option_map (Nat.add 2) (jspan_l open close depth true t2)
        end
      else if c =? 34 then option_map S (jspan_l open close depth false t)
      else option_map S (jspan_l open close depth true t)
    else
      if c =? 0 then None
      else if c =? 34 then option_map S (jspan_l open close depth true t)
      else if c =? open then
        option_map S (jspan_l open close (depth + 1) false t)
      else if c =? close then
        if Z.eqb (depth - 1) 0 then Some 1%nat
        else option_map S (jspan_l open close (depth - 1) false t)
      else option_map S (jspan_l open close depth false t)
  end.

Lemma jspan_bound : forall n oc c2 d ins l k,
  (length l <= n)%nat ->
  jspan_l oc c2 d ins l = Some k -> (1 <= k <= length l)%nat.
Proof.
  induction n as [n IH] using lt_wf_ind; intros oc c2 d ins l k Hln H.
  destruct l as [|c t]; simpl in H; [discriminate|].
  destruct ins.
  - destruct (Z.eqb c 0); [discriminate|].
    destruct (Z.eqb c 92).
    + destruct t as [|c2' t2]; [discriminate|].
      destruct (Z.eqb c2' 0); [discriminate|].
      apply option_map_inv in H. destruct H as [k' [F Hk]].
      pose proof (IH (length t2) ltac:(simpl in Hln; lia) oc c2 d true t2 k'
                    ltac:(jsub) F) as Hb. simpl in *. lia.
    + destruct (Z.eqb c 34).
      * apply option_map_inv in H. destruct H as [k' [F Hk]].
        pose proof (IH (length t) ltac:(simpl in Hln; lia) oc c2 d false t k'
                      ltac:(jsub) F) as Hb. simpl in *. lia.
      * apply option_map_inv in H. destruct H as [k' [F Hk]].
        pose proof (IH (length t) ltac:(simpl in Hln; lia) oc c2 d true t k'
                      ltac:(jsub) F) as Hb. simpl in *. lia.
  - destruct (Z.eqb c 0); [discriminate|].
    destruct (Z.eqb c 34).
    + apply option_map_inv in H. destruct H as [k' [F Hk]].
      pose proof (IH (length t) ltac:(simpl in Hln; lia) oc c2 d true t k'
                    ltac:(jsub) F) as Hb. simpl in *. lia.
    + destruct (Z.eqb c oc).
      * apply option_map_inv in H. destruct H as [k' [F Hk]].
        pose proof (IH (length t) ltac:(simpl in Hln; lia) oc c2 (d+1) false
                      t k' ltac:(jsub) F) as Hb. simpl in *. lia.
      * destruct (Z.eqb c c2).
        -- destruct (Z.eqb (d - 1) 0); [inversion H; simpl; lia|].
           apply option_map_inv in H. destruct H as [k' [F Hk]].
           pose proof (IH (length t) ltac:(simpl in Hln; lia) oc c2 (d-1)
                         false t k' ltac:(jsub) F) as Hb. simpl in *. lia.
        -- apply option_map_inv in H. destruct H as [k' [F Hk]].
           pose proof (IH (length t) ltac:(simpl in Hln; lia) oc c2 d false
                         t k' ltac:(jsub) F) as Hb. simpl in *. lia.
Qed.

(* ---------------- list helpers ---------------- *)

(* ---------------- jkey model ---------------- *)

Lemma nth_pos : forall (l : list Z) n,
  nth n l 0 <> 0 -> (n < length l)%nat.
Proof.
  intros l n H. destruct (Nat.lt_ge_cases n (length l)) as [Hlt|Hge];
    [exact Hlt|].
  apply nth_overflow with (d:=0) in Hge. congruence.
Qed.

(* the abstract decoder contract: jstr at a quote consumes k >= 1
   bytes — progress is all the fuel argument needs; the per-iteration
   e <= oend check lives inside jkey_f itself *)
Definition dec_contract (dec : list Z -> option (list Z * nat)) : Prop :=
  forall b d k, dec b = Some (d, k) -> (1 <= k)%nat.

(* fuel-bounded faithful model of
     while (p < oend) { q = memchr(quote); e = jstr(q); ws; colon; ws;
                        match ? return : skip value }
   l = remaining object span [p,oend); rest = buffer tail past oend.
   Returns the value offset within l, or None. *)
Fixpoint jkey_f (fuel : nat) (l rest : list Z)
                (dec : list Z -> option (list Z * nat)) (key : list Z)
  : option nat :=
  match fuel with
  | 0%nat => None
  | S f =>
    match find_c 34 l with
    | None => None
    | Some q =>
      match dec (skipn q l ++ rest) with
      | None => None
      | Some (d, k) =>
        if (q + k <=? length l)%nat then
          let e := (q + k)%nat in
          let after := skipn e l in
          let w1 := ws_skip after in
          if nth w1 after 0 =? 58 then
            let after2 := skipn (S w1) after in
            let w2 := ws_skip after2 in
            if (w2 <? length after2)%nat then
              if list_eq_dec Z.eq_dec d key then Some (e + w1 + 1 + w2)%nat
              else
                let w := (e + w1 + 1 + w2)%nat in
                let npb := nth w2 after2 0 in
                if npb =? 34 then
                  match jstr_skip_l (skipn w2 after2 ++ rest) with
                  | Some vr => option_map (fun x => (w + vr + x)%nat)
                     (jkey_f f (skipn (w + vr)%nat l) rest dec key)
                  | None => None          (* p = oend -> loop exits *)
                  end
                else if (npb =? 123) || (npb =? 91) then
                  match jspan_l npb (if npb =? 123 then 125 else 93) 0 false
                                 (skipn w2 after2 ++ rest) with
                  | Some vr => option_map (fun x => (w + vr + x)%nat)
                     (jkey_f f (skipn (w + vr)%nat l) rest dec key)
                  | None => None
                  end
                else
                  match find_c 44 (skipn w2 after2) with
                  | Some co => option_map (fun x => (w + co + x)%nat)
                     (jkey_f f (skipn (w + co)%nat l) rest dec key)
                  | None => None          (* np reached oend *)
                  end
            else None
          else option_map (fun x => (e + x)%nat)
                 (jkey_f f (skipn e l) rest dec key)   (* no colon: p = e *)
        else None                              (* e > oend -> die *)
      end
    end
  end.

Definition jkey (l rest : list Z)
                (dec : list Z -> option (list Z * nat)) (key : list Z)
  : option nat := jkey_f (S (length l)) l rest dec key.

(* the returned offset is strictly inside the object span *)
Theorem jkey_bound : forall fuel l rest dec key v,
  jkey_f fuel l rest dec key = Some v -> (v < length l)%nat.
Proof.
  induction fuel as [|f IH]; intros l rest dec key v H;
    cbn [jkey_f] in H; [discriminate|].
  destruct (find_c 34 l) as [q|] eqn:Fq; [|discriminate].
  destruct (dec (skipn q l ++ rest)) as [[d k]|] eqn:Fd; [|discriminate].
  destruct (Nat.leb (q + k) (length l)) eqn:Fl; [|discriminate].
  apply Nat.leb_le in Fl.
  set (e := (q + k)%nat) in *.
  set (after := skipn e l) in *.
  set (w1 := ws_skip after) in *.
  destruct (Z.eqb (nth w1 after 0) 58) eqn:Ec.
  - set (after2 := skipn (S w1) after) in *.
    set (w2 := ws_skip after2) in *.
    set (w := (e + w1 + 1 + w2)%nat) in *.
    destruct (Nat.ltb w2 (length after2)) eqn:Ew2; [|discriminate].
    apply Nat.ltb_lt in Ew2.
    destruct (list_eq_dec Z.eq_dec d key); subst.
    + inversion H; subst v.
      apply Z.eqb_eq in Ec.
      assert (Hw1 : (w1 < length after)%nat)
        by (apply nth_pos; congruence).
      subst after2 after w2 w1 e.
      rewrite !length_skipn in *. lia.
    + destruct (Z.eqb (nth w2 after2 0) 34) eqn:E34.
      * destruct (jstr_skip_l (skipn w2 after2 ++ rest)) as [vr|] eqn:Ejs;
          [|discriminate].
        destruct (jkey_f f (skipn (w + vr) l) rest dec key) as [x|] eqn:Erec;
          simpl in H; [|discriminate].
        inversion H; subst v.
        apply IH in Erec. rewrite length_skipn in Erec. lia.
      * destruct (((nth w2 after2 0) =? 123) || ((nth w2 after2 0) =? 91))
          eqn:Ebr.
        -- destruct (jspan_l (nth w2 after2 0)
              (if nth w2 after2 0 =? 123 then 125 else 93) 0 false
              (skipn w2 after2 ++ rest)) as [vr|] eqn:Ejp; [|discriminate].
           destruct (jkey_f f (skipn (w + vr) l) rest dec key) as [x|]
             eqn:Erec; simpl in H; [|discriminate].
           inversion H; subst v.
           apply IH in Erec. rewrite length_skipn in Erec. lia.
        -- destruct (find_c 44 (skipn w2 after2)) as [co|] eqn:Fco;
             [|discriminate].
           destruct (jkey_f f (skipn (w + co) l) rest dec key) as [x|]
             eqn:Erec; simpl in H; [|discriminate].
           inversion H; subst v.
           apply IH in Erec. rewrite length_skipn in Erec. lia.
  - destruct (jkey_f f after rest dec key) as [x|] eqn:Erec;
      simpl in H; [|discriminate].
    inversion H; subst v.
    apply IH in Erec.
    subst after. rewrite length_skipn in Erec. lia.
Qed.

(* under the decoder contract, every outer iteration consumes >= 1 byte of
   the object span, so fuel beyond length l is provably irrelevant *)
Theorem jkey_fuel : forall n l fuel rest dec key,
  dec_contract dec ->
  (length l <= n)%nat -> (n < fuel)%nat ->
  jkey_f fuel l rest dec key = jkey_f (S (length l)) l rest dec key.
Proof.
  induction n as [n IH] using lt_wf_ind;
    intros l fuel rest dec key Hc Hln Hf.
  destruct fuel as [|f]; [lia|].
  destruct l as [|c0 t].
  - reflexivity.
  - cbn [length] in Hln. cbn [jkey_f]. cbn [length].
    destruct (find_c 34 (c0 :: t)) as [q|] eqn:Fq; [|
      destruct (find_c 34 (c0 :: t)) as [q'|] eqn:Fq';
        [congruence|reflexivity]].
    destruct (dec (skipn q (c0 :: t) ++ rest)) as [[d k]|] eqn:Fd;
      [|reflexivity].
    pose proof (Hc _ _ _ Fd) as Hk1.
    destruct (Nat.leb (q + k) (S (length t))) eqn:Fl; [|reflexivity].
    apply Nat.leb_le in Fl.
    set (e := (q + k)%nat).
    set (after := skipn e (c0 :: t)).
    set (w1 := ws_skip after).
    destruct (Z.eqb (nth w1 after 0) 58) eqn:Ec.
    + set (after2 := skipn (S w1) after).
      set (w2 := ws_skip after2).
      destruct (Nat.ltb w2 (length after2)) eqn:Ew2; [|reflexivity].
      destruct (list_eq_dec Z.eq_dec d key) as [Ee|Ee]; [reflexivity|].
      set (w := (e + w1 + 1 + w2)%nat).
      destruct (Z.eqb (nth w2 after2 0) 34) eqn:E34.
      * destruct (jstr_skip_l (skipn w2 after2 ++ rest)) as [vr|] eqn:Ejs;
          [|reflexivity].
        (* recurse at w + vr >= w >= e + 1 >= 1 *)
        rewrite (IH (length (skipn (w + vr)%nat (c0 :: t)))
                  ltac:(jsub)
                  (skipn (w + vr)%nat (c0 :: t)) f rest dec key Hc
                  ltac:(jsub) ltac:(jsub)).
        rewrite (IH (length (skipn (w + vr)%nat (c0 :: t)))
                  ltac:(jsub)
                  (skipn (w + vr)%nat (c0 :: t)) (S (length t)) rest dec key
                  Hc ltac:(jsub) ltac:(jsub)).
        reflexivity.
      * destruct (((nth w2 after2 0) =? 123) || ((nth w2 after2 0) =? 91))
          eqn:Ebr.
        -- destruct (jspan_l (nth w2 after2 0)
              (if nth w2 after2 0 =? 123 then 125 else 93) 0 false
              (skipn w2 after2 ++ rest)) as [vr|] eqn:Ejp; [|reflexivity].
           rewrite (IH (length (skipn (w + vr)%nat (c0 :: t)))
                     ltac:(jsub)
                     (skipn (w + vr)%nat (c0 :: t)) f rest dec key Hc
                     ltac:(jsub) ltac:(jsub)).
           rewrite (IH (length (skipn (w + vr)%nat (c0 :: t)))
                     ltac:(jsub)
                     (skipn (w + vr)%nat (c0 :: t)) (S (length t)) rest
                     dec key Hc ltac:(jsub)
                     ltac:(jsub)).
           reflexivity.
        -- destruct (find_c 44 (skipn w2 after2)) as [co|] eqn:Fco;
             [|reflexivity].
           rewrite (IH (length (skipn (w + co)%nat (c0 :: t)))
                     ltac:(jsub)
                     (skipn (w + co)%nat (c0 :: t)) f rest dec key Hc
                     ltac:(jsub) ltac:(jsub)).
           rewrite (IH (length (skipn (w + co)%nat (c0 :: t)))
                     ltac:(jsub)
                     (skipn (w + co)%nat (c0 :: t)) (S (length t)) rest
                     dec key Hc ltac:(jsub)
                     ltac:(jsub)).
           reflexivity.
    + rewrite (IH (length after)
                ltac:(jsub)
                after f rest dec key Hc
                ltac:(jsub)
                ltac:(jsub)).
      rewrite (IH (length after)
                ltac:(jsub)
                after (S (length t)) rest dec key Hc
                ltac:(jsub)
                ltac:(jsub)).
      reflexivity.
Qed.
(* the semantic bundle: v sits right after  "decoded==key" ws : ws  *)
Definition key_at (l rest : list Z)
                  (dec : list Z -> option (list Z * nat)) (key : list Z)
                  (v : nat) : Prop :=
  exists q k w1 w2,
    nth q l 0 = 34 /\
    dec (skipn q l ++ rest) = Some (key, k) /\
    (q + k <= length l)%nat /\
    (forall j, (j < w1)%nat -> is_ws (nth (q + k + j)%nat l 0) = true) /\
    nth (q + k + w1)%nat l 0 = 58 /\
    (forall j, (j < w2)%nat -> is_ws (nth (q + k + w1 + 1 + j)%nat l 0)
               = true) /\
    v = (q + k + w1 + 1 + w2)%nat /\ (v < length l)%nat.

Lemma jkey_f_nil : forall f rest dec key,
  jkey_f f [] rest dec key = None.
Proof. destruct f; reflexivity. Qed.

(* transport a key_at on a suffix back to the whole list, shifted by W *)
Lemma key_at_shift : forall l rest dec key W x,
  (W <= length l)%nat ->
  key_at (skipn W l) rest dec key x -> key_at l rest dec key (W + x).
Proof.
  intros l rest dec key W x HW H.
  destruct H as [q [k [w1 [w2 Hx]]]].
  destruct Hx as [H34 [Hd [Hlen [Hws1 [H58 [Hws2 [Hvx Hxb]]]]]]].
  exists (W + q)%nat, k, w1, w2.
  rewrite length_skipn in Hlen, Hxb.
  repeat split.
  - rewrite <- nth_skipn. exact H34.
  - replace (W + q)%nat with (q + W)%nat by lia.
    rewrite <- skipn_skipn. exact Hd.
  - lia.
  - intros j Hj. pose proof (Hws1 j Hj) as Hw.
    rewrite nth_skipn in Hw.
    replace (W + (q + k + j))%nat with (W + q + k + j)%nat in Hw by lia.
    exact Hw.
  - rewrite nth_skipn in H58.
    replace (W + (q + k + w1))%nat with (W + q + k + w1)%nat in H58 by lia.
    exact H58.
  - intros j Hj. pose proof (Hws2 j Hj) as Hw.
    rewrite nth_skipn in Hw.
    replace (W + (q + k + w1 + 1 + j))%nat
      with (W + q + k + w1 + 1 + j)%nat in Hw by lia.
    exact Hw.
  - lia.
  - lia.
Qed.

(* semantics: a returned offset sits immediately after
   quote <decoded==key> quote ws colon ws - the result IS the value position of
   a key occurrence *)
Theorem jkey_sem : forall fuel l rest dec key v,
  jkey_f fuel l rest dec key = Some v -> key_at l rest dec key v.
Proof.
  induction fuel as [|f IH]; intros l rest dec key v H;
    cbn [jkey_f] in H; [discriminate|].
  destruct (find_c 34 l) as [q|] eqn:Fq; [|discriminate].
  destruct (dec (skipn q l ++ rest)) as [[d k]|] eqn:Fd; [|discriminate].
  destruct (Nat.leb (q + k) (length l)) eqn:Fl; [|discriminate].
  apply Nat.leb_le in Fl.
  set (e := (q + k)%nat) in *.
  set (after := skipn e l) in *.
  set (w1 := ws_skip after) in *.
  destruct (Z.eqb (nth w1 after 0) 58) eqn:Ec.
  - set (after2 := skipn (S w1) after) in *.
    set (w2 := ws_skip after2) in *.
    set (w := (e + w1 + 1 + w2)%nat) in *.
    destruct (Nat.ltb w2 (length after2)) eqn:Ew2; [|discriminate].
    destruct (list_eq_dec Z.eq_dec d key) as [Edk|Edk]; subst.
    + inversion H; subst v.
      unfold key_at. exists q, k, w1, w2.
      pose proof (find_c_char _ _ _ Fq) as Hq34.
      apply Z.eqb_eq in Ec.
      apply Nat.ltb_lt in Ew2.
      assert (Hw1 : (w1 < length after)%nat) by (apply nth_pos; congruence).
      repeat split.
      * exact Hq34.
      * exact Fd.
      * exact Fl.
      * intros j Hj.
        pose proof (ws_skip_ws after j Hj) as Hw.
        unfold after in Hw. rewrite nth_skipn in Hw.
        replace (e + j)%nat with (q + k + j)%nat in Hw by (subst e; lia).
        exact Hw.
      * unfold after in Ec. rewrite nth_skipn in Ec.
        replace (e + w1)%nat with (q + k + w1)%nat in Ec by (subst e; lia).
        exact Ec.
      * intros j Hj.
        pose proof (ws_skip_ws after2 j Hj) as Hw.
        unfold after2, after in Hw. rewrite !nth_skipn in Hw.
        replace (e + (S w1 + j))%nat with (q + k + w1 + 1 + j)%nat in Hw
          by (subst e; lia).
        exact Hw.
      * subst e after after2.
        rewrite length_skipn in Hw1.
        rewrite !length_skipn in Ew2. lia.
    + destruct (Z.eqb (nth w2 after2 0) 34) eqn:E34.
      * destruct (jstr_skip_l (skipn w2 after2 ++ rest)) as [vr|] eqn:Ejs;
          [|discriminate].
        destruct (jkey_f f (skipn (w + vr) l) rest dec key) as [x|]
          eqn:Erec; simpl in H; [|discriminate].
        inversion H; subst v.
        assert (Hwl : (w + vr <= length l)%nat).
        { destruct (Nat.lt_ge_cases (w + vr) (length l)) as [|Hge]; [lia|].
          rewrite skipn_all2 in Erec by lia.
          rewrite jkey_f_nil in Erec. discriminate. }
        apply IH in Erec. apply key_at_shift; assumption.
      * destruct (((nth w2 after2 0) =? 123) || ((nth w2 after2 0) =? 91))
          eqn:Ebr.
        -- destruct (jspan_l (nth w2 after2 0)
              (if nth w2 after2 0 =? 123 then 125 else 93) 0 false
              (skipn w2 after2 ++ rest)) as [vr|] eqn:Ejp; [|discriminate].
           destruct (jkey_f f (skipn (w + vr) l) rest dec key) as [x|]
             eqn:Erec; simpl in H; [|discriminate].
           inversion H; subst v.
           assert (Hwl : (w + vr <= length l)%nat).
           { destruct (Nat.lt_ge_cases (w + vr) (length l)) as [|Hge];
               [lia|].
             rewrite skipn_all2 in Erec by lia.
             rewrite jkey_f_nil in Erec. discriminate. }
           apply IH in Erec. apply key_at_shift; assumption.
        -- destruct (find_c 44 (skipn w2 after2)) as [co|] eqn:Fco;
             [|discriminate].
           destruct (jkey_f f (skipn (w + co) l) rest dec key) as [x|]
             eqn:Erec; simpl in H; [|discriminate].
           inversion H; subst v.
           assert (Hwl : (w + co <= length l)%nat).
           { destruct (Nat.lt_ge_cases (w + co) (length l)) as [|Hge];
               [lia|].
             rewrite skipn_all2 in Erec by lia.
             rewrite jkey_f_nil in Erec. discriminate. }
           apply IH in Erec. apply key_at_shift; assumption.
  - destruct (jkey_f f after rest dec key) as [x|] eqn:Erec;
      simpl in H; [|discriminate].
    inversion H; subst v.
    apply IH in Erec.
    apply key_at_shift.
    + subst e. exact Fl.
    + exact Erec.
Qed.

(* ---------------- concrete decoder: jstr ---------------- *)
(* Faithful list model of caiw.c's jstr: escape letters, \uXXXX via a
   4-hex reader, high-surrogate + \uDC00-DFFF pairing, the bs cap checks
   (i+1 >= bs on plain/short escapes, i+4 >= bs on every \u), and the
   embedded-NUL rejection.  Returns (decoded bytes, consumed count). *)

Definition hexv (c : Z) : option Z :=
  if (48 <=? c) && (c <=? 57) then Some (c - 48)
  else if (97 <=? c) && (c <=? 102) then Some (c - 87)
  else if (65 <=? c) && (c <=? 70) then Some (c - 55)
  else None.

Definition escv (c : Z) : option Z :=
  if c =? 34 then Some 34
  else if c =? 92 then Some 92
  else if c =? 47 then Some 47
  else if c =? 98 then Some 8
  else if c =? 102 then Some 12
  else if c =? 110 then Some 10
  else if c =? 114 then Some 13
  else if c =? 116 then Some 9
  else None.

Definition utf8_enc (cp : Z) : list Z :=
  if cp <? 128 then [cp]
  else if cp <? 2048 then [192 + Z.shiftr cp 6; 128 + cp mod 64]
  else if cp <? 65536 then
    [224 + Z.shiftr cp 12; 128 + (Z.shiftr cp 6) mod 64; 128 + cp mod 64]
  else
    [240 + Z.shiftr cp 18; 128 + (Z.shiftr cp 12) mod 64;
     128 + (Z.shiftr cp 6) mod 64; 128 + cp mod 64].

(* emit_cp: C's shared "\u" tail — cap check i+4 >= bs, NUL-cp
   rejection, UTF-8 emit; rec = already-computed recursive result;
   u = extra surrogate bytes consumed (0 or 6). *)
Definition emit_cp (cp : Z) (cap i u : nat) (rec : option (list Z * nat))
    : option (list Z * nat) :=
  if (i + 4 <? cap)%nat then
    if cp =? 0 then None
    else option_map (fun '(d,k) => (utf8_enc cp ++ d, (6 + u + k)%nat)) rec
  else None.

(* jstr_body: l = bytes AFTER the open quote; i = decoded count so far.
   Returns (decoded suffix, bytes consumed from l) — Some (_,1) at the
   close quote. *)
Fixpoint jstr_body (l : list Z) (cap i : nat) : option (list Z * nat) :=
  match l with
  | [] => None
  | c :: t =>
    if c =? 0 then None
    else if c =? 34 then Some ([], 1)%nat
    else if c =? 92 then
      match t with
      | [] => None
      | e :: t2 =>
        if e =? 117 then                       (* backslash-u *)
          match t2 with
          | h1 :: h2 :: h3 :: h4 :: t3 =>
            match hexv h1, hexv h2, hexv h3, hexv h4 with
            | Some a, Some b, Some cv, Some d =>
              let cp0 := a * 4096 + b * 256 + cv * 16 + d in
              let resume :=
                emit_cp cp0 cap i 0
                  (jstr_body t3 cap (i + length (utf8_enc cp0))%nat) in
              if (55296 <=? cp0) && (cp0 <=? 56319) then
                match t3 with
                | x1 :: x2 :: t3b =>
                  if (x1 =? 92) && (x2 =? 117) then
                    match t3b with
                    | l1 :: l2 :: l3 :: l4 :: t4 =>
                      match hexv l1, hexv l2, hexv l3, hexv l4 with
                      | Some la, Some lb, Some lc, Some ld =>
                        let lo := la*4096 + lb*256 + lc*16 + ld in
                        if (56320 <=? lo) && (lo <=? 57343) then
                          emit_cp
                            (65536 + (cp0 - 55296) * 1024 + (lo - 56320))
                            cap i 6
                            (jstr_body t4 cap
                              (i + length (utf8_enc
                                 (65536 + (cp0 - 55296) * 1024
                                  + (lo - 56320))))%nat)
                        else resume
                      | _, _, _, _ => resume
                      end
                    | _ => resume
                    end
                  else resume
                | _ => resume
                end
              else resume
            | _, _, _, _ => None
            end
          | _ => None
          end
        else
          match escv e with
          | None => None
          | Some cv =>
            if (i + 1 <? cap)%nat then
              option_map (fun '(d,k) => (cv :: d, S (S k)))
                         (jstr_body t2 cap (i + 1)%nat)
            else None
          end
      end
    else
      if (i + 1 <? cap)%nat then
        option_map (fun '(d,k) => (c :: d, S k)) (jstr_body t cap (i + 1)%nat)
      else None
  end.

Definition jstr_dec (cap : nat) (l : list Z) : option (list Z * nat) :=
  match l with
  | c :: t => if c =? 34 then
                option_map (fun '(d,k) => (d, S k)) (jstr_body t cap 0)
              else None
  | [] => None
  end.

(* the concrete decoder meets the abstract contract: a successful
   decode always consumes at least the open quote *)
Lemma jstr_dec_contract : forall cap, dec_contract (jstr_dec cap).
Proof.
  intros cap b d k H. unfold jstr_dec in H.
  destruct b as [|c t]; [discriminate|].
  destruct (Z.eqb c 34); [|discriminate].
  apply option_map_inv in H. destruct H as [y [F Hk]].
  destruct y as [d2 k2]. inversion Hk. subst. lia.
Qed.

(* jkey with the real decoder wired in — fully concrete, extractable *)
Definition jkey_c (l rest key : list Z) : option nat :=
  jkey l rest (jstr_dec (S (length l))) key.

(* the abstract theorems discharge on the concrete decoder *)
Theorem jkey_c_bound : forall l rest key v,
  jkey_c l rest key = Some v -> (v < length l)%nat.
Proof. intros l rest key v H. eapply jkey_bound. exact H. Qed.

Theorem jkey_c_fuel : forall l rest fuel key,
  (length l < fuel)%nat ->
  jkey_f fuel l rest (jstr_dec (S (length l))) key
  = jkey_c l rest key.
Proof.
  intros l rest fuel key Hf. unfold jkey_c, jkey.
  apply jkey_fuel with (n := length l);
    [apply jstr_dec_contract | apply Nat.le_refl | exact Hf].
Qed.

Theorem jkey_c_sem : forall l rest key v,
  jkey_c l rest key = Some v ->
  key_at l rest (jstr_dec (S (length l))) key v.
Proof. intros l rest key v H. eapply jkey_sem. exact H. Qed.

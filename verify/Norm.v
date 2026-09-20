(* Norm.v — Coq proof of the norm_ctx output contract.
 *
 * Algorithm-level proof, same tier as Rans.v / Utf8.v.  norm_ctx is the
 * histogram normalizer feeding every rANS channel: it must emit a
 * table summing to exactly TOT while preserving support (zero stays
 * zero, positive stays positive).  A violation desyncs encoder and
 * decoder catastrophically.  CBMC proves this only at AW=2 — the
 * symbolic division h[i]*bud/tot defeats bit-blasting for larger
 * alphabets.  Here we prove the contract for ALL histograms and ALL
 * aw <= TOT (the implicit C precondition: every call site passes
 * aw <= 1024).
 *
 * C model (caiw.c):
 *   tot = Σh; nz = #{h_i > 0}
 *   if (!nz) { f_i = TOT/aw; f[aw-1] += TOT - (TOT/aw)*aw; return; }
 *   bud = TOT - nz
 *   f_i = h_i>0 ? h_i*bud/tot + 1 : 0        (u64 path; u128 same q)
 *   f[bi] += TOT - Σf   where bi = first index of max h
 *)

From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Open Scope Z_scope.

Definition TOT : Z := 32768.

Definition zsum (l : list Z) : Z := fold_right Z.add 0 l.

Definition zcnt_pos (h : list Z) : nat :=
  length (filter (fun x => x >? 0) h).

(* update: l[i] += v *)
Fixpoint upd (i : nat) (l : list Z) (v : Z) : list Z :=
  match i, l with
  | _, [] => []
  | O, x :: tl => (x + v) :: tl
  | S i', x :: tl => x :: upd i' tl v
  end.

Definition zmax (h : list Z) : Z := fold_right Z.max 0 h.

Fixpoint first_idx (h : list Z) (v : Z) : nat :=
  match h with
  | [] => 0
  | x :: tl => if x =? v then 0 else S (first_idx tl v)
  end.

Definition argmax (h : list Z) : nat := first_idx h (zmax h).

Definition base_f (h : list Z) : list Z :=
  map (fun hi =>
         if hi >? 0
         then hi * (TOT - Z.of_nat (zcnt_pos h)) / zsum h + 1
         else 0) h.

Definition norm_nz (h : list Z) (bi : nat) : list Z :=
  upd bi (base_f h) (TOT - zsum (base_f h)).

Definition norm_z (aw : nat) : list Z :=
  match aw with
  | O => []
  | S k => upd k (repeat (TOT / Z.of_nat (S k)) (S k))
               (TOT - (TOT / Z.of_nat (S k)) * Z.of_nat (S k))
  end.

Definition norm_ctx (h : list Z) : list Z :=
  if (zcnt_pos h =? 0)%nat
  then norm_z (length h)
  else norm_nz h (argmax h).

(* ================= generic list lemmas ================= *)

Lemma upd_length : forall i l v, (length (upd i l v) = length l)%nat.
Proof. induction i; intros [|x tl] v; simpl; auto. Qed.

Lemma nth_upd_eq : forall i l v d,
  (i < length l)%nat -> nth i (upd i l v) d = nth i l d + v.
Proof.
  induction i; intros [|x tl] v d H; simpl in *; try lia.
  rewrite (IHi tl v d) by lia. reflexivity.
Qed.

Lemma nth_upd_neq : forall i j l v d,
  i <> j -> nth j (upd i l v) d = nth j l d.
Proof.
  induction i; intros [|j'] [|x tl] v d H; simpl in *;
    try lia; auto.
Qed.

Lemma zsum_upd : forall i l v,
  (i < length l)%nat -> zsum (upd i l v) = zsum l + v.
Proof.
  induction i; intros [|x tl] v H; simpl in *; try lia.
  rewrite IHi by lia. lia.
Qed.

Lemma zsum_nonneg : forall l,
  Forall (fun x => 0 <= x) l -> 0 <= zsum l.
Proof. intros l H. induction H; simpl; lia. Qed.

Lemma zsum_nth_le : forall i l d,
  Forall (fun x => 0 <= x) l -> (i < length l)%nat ->
  nth i l d <= zsum l.
Proof.
  induction i; intros [|x tl] d Hf H; simpl in *; try lia.
  - inversion Hf; subst. pose proof (zsum_nonneg _ H3). lia.
  - inversion Hf; subst. pose proof (IHi tl d H3 ltac:(lia)). lia.
Qed.

Lemma zsum_map_add : forall (f g : Z -> Z) l,
  zsum (map (fun x => f x + g x) l) = zsum (map f l) + zsum (map g l).
Proof.
  intros f g l. induction l as [|x tl IH]; simpl; lia.
Qed.

Lemma zsum_cons : forall x l, zsum (x :: l) = x + zsum l.
Proof. reflexivity. Qed.

Lemma zsum_indicator : forall h,
  zsum (map (fun x => if x >? 0 then 1 else 0) h)
  = Z.of_nat (zcnt_pos h).
Proof.
  unfold zcnt_pos. induction h as [|x tl IH]; [reflexivity|].
  cbn [map filter length]. destruct (x >? 0).
  - cbn [length]. rewrite Nat2Z.inj_succ.
    rewrite zsum_cons. rewrite IH. lia.
  - rewrite zsum_cons. rewrite IH. lia.
Qed.

Lemma zsum_map_mul : forall h b,
  zsum (map (fun x => x * b) h) = b * zsum h.
Proof.
  induction h as [|x tl IH]; intros b; simpl; rewrite ?IH; ring.
Qed.

Lemma zcnt_pos_le_length : forall h, (zcnt_pos h <= length h)%nat.
Proof.
  unfold zcnt_pos. induction h; simpl; auto.
  destruct (a >? 0); simpl; lia.
Qed.

Lemma zcnt_pos_zero : forall h,
  Forall (fun x => 0 <= x) h -> zcnt_pos h = 0%nat -> zsum h = 0.
Proof.
  induction h as [|x tl IH]; intros Hf Hz; [reflexivity|].
  inversion Hf as [|x0 tl0 Hx Htl]; subst.
  unfold zcnt_pos in *. simpl in Hz.
  destruct (x >? 0) eqn:E.
  - simpl in Hz. discriminate.
  - simpl. assert (x <= 0) by (destruct (Z.gtb_spec x 0);
                              congruence).
    assert (x = 0) by lia. subst x.
    pose proof (IH Htl Hz). lia.
Qed.

Lemma zcnt_pos_sum_pos : forall h,
  Forall (fun x => 0 <= x) h -> (0 < zcnt_pos h)%nat -> 0 < zsum h.
Proof.
  intros h Hf Hc. unfold zcnt_pos in Hc.
  destruct (filter (fun x => x >? 0) h) as [|x0 fl] eqn:Ef;
    [simpl in Hc; lia|].
  assert (Hin : In x0 h).
  { assert (In x0 (x0 :: fl)) by (left; reflexivity).
    rewrite <- Ef in H. apply filter_In in H. tauto. }
  assert (0 < x0).
  { assert (In x0 (x0 :: fl)) by (left; reflexivity).
    rewrite <- Ef in H. apply filter_In in H.
    destruct H as [_ Hb]. apply Z.gtb_lt in Hb. exact Hb. }
  apply In_nth with (d := 0) in Hin.
  destruct Hin as (j & Hj & <-).
  pose proof (zsum_nth_le j h 0 Hf Hj). lia.
Qed.

Lemma upd_nonneg : forall i l v,
  Forall (fun x => 0 <= x) l -> 0 <= v ->
  Forall (fun x => 0 <= x) (upd i l v).
Proof.
  induction i; intros [|x tl] v Hf Hv; simpl; auto.
  - inversion Hf as [|? ? Hx Htl]; subst.
    constructor; [lia|auto].
  - inversion Hf as [|? ? Hx Htl]; subst.
    constructor; [exact Hx|]. apply IHi; auto.
Qed.

Lemma nth_map' : forall (A B : Type) (f : A -> B) (l : list A)
                      (d : A) (d' : B) (n : nat),
  (n < length l)%nat -> nth n (map f l) d' = f (nth n l d).
Proof.
  intros A B f l d d' n Hn.
  replace (nth n (map f l) d') with (nth n (map f l) (f d)).
  - apply map_nth.
  - apply nth_indep. rewrite length_map. exact Hn.
Qed.

Lemma nth_repeat' : forall (A : Type) (v : A) n i d,
  (i < n)%nat -> nth i (repeat v n) d = v.
Proof.
  intros A v n i d Hi.
  assert (E : nth i (repeat v n) d = nth i (repeat v n) v).
  { apply nth_indep. rewrite repeat_length. exact Hi. }
  rewrite E. apply nth_repeat.
Qed.

Lemma nth_base_f : forall h i,
  (i < length h)%nat ->
  nth i (base_f h) 0
  = (if nth i h 0 >? 0
     then nth i h 0 * (TOT - Z.of_nat (zcnt_pos h)) / zsum h + 1
     else 0).
Proof.
  intros h i Hi. unfold base_f.
  rewrite (nth_map' _ _ _ _ 0 _ _ Hi). reflexivity.
Qed.

Lemma zsum_repeat : forall n v,
  zsum (repeat v n) = Z.of_nat n * v.
Proof.
  induction n; intros v.
  - reflexivity.
  - cbn [repeat]. rewrite zsum_cons. rewrite IHn.
    rewrite Nat2Z.inj_succ. unfold Z.succ. ring.
Qed.

(* ================= division lemma ============================= *)
(* Σ floor(a_i/d) <= floor(Σ a_i / d) for nonneg a_i — the key fact
   bounding the quotient sum. *)

Lemma div_add_floor : forall a b d,
  0 <= a -> 0 <= b -> 0 < d -> a / d + b / d <= (a + b) / d.
Proof.
  intros a b d Ha Hb Hd.
  apply Z.div_le_lower_bound; [exact Hd|].
  pose proof (Z.div_mod a d ltac:(lia)) as Ha'.
  pose proof (Z.div_mod b d ltac:(lia)) as Hb'.
  pose proof (Z.mod_pos_bound a d ltac:(lia)) as Ham.
  pose proof (Z.mod_pos_bound b d ltac:(lia)) as Hbm.
  lia.
Qed.

Lemma zsum_div_le : forall l d,
  Forall (fun x => 0 <= x) l -> 0 < d ->
  zsum (map (fun x => x / d) l) <= zsum l / d.
Proof.
  induction l as [|x tl IH]; intros d Hf Hd; simpl.
  - rewrite Z.div_0_l by lia. lia.
  - inversion Hf as [|? ? Hx Htl]; subst.
    pose proof (div_add_floor x (zsum tl) d Hx
                  (zsum_nonneg _ Htl) Hd).
    specialize (IH d Htl Hd). lia.
Qed.

(* ================= argmax ===================================== *)

Lemma in_zmax : forall h,
  Forall (fun x => 0 <= x) h -> h <> [] -> In (zmax h) h.
Proof.
  induction h as [|x tl IH]; intros Hf Hne; [congruence|].
  inversion Hf as [|? ? Hx Htl]; subst.
  unfold zmax in *; simpl.
  destruct (Z.max_dec x (fold_right Z.max 0 tl)) as [E|E].
  - left. symmetry. exact E.
  - destruct tl as [|y tl'].
    + simpl in *. assert (Z.max x 0 = x)
        by (apply Z.max_l; lia).
      assert (x = 0) by lia. subst. left. reflexivity.
    + right. rewrite E. apply IH; auto. discriminate.
Qed.

Lemma zmax_ge : forall i h d,
  Forall (fun x => 0 <= x) h -> (i < length h)%nat ->
  nth i h d <= zmax h.
Proof.
  induction i; intros [|x tl] d Hf H; simpl in *; try lia.
  inversion Hf as [|? ? Hx Htl]; subst.
  pose proof (IHi tl d Htl ltac:(lia)).
  assert (Hmx : fold_right Z.max 0 tl <= zmax (x :: tl)).
  { unfold zmax; simpl. apply Z.le_max_r. }
  lia.
Qed.

Lemma first_idx_spec : forall h v,
  In v h ->
  (first_idx h v < length h)%nat /\ nth (first_idx h v) h 0 = v.
Proof.
  induction h as [|x tl IH]; intros v Hin; simpl in *;
    [contradiction|].
  destruct (x =? v) eqn:Exv.
  - apply Z.eqb_eq in Exv. subst x. simpl.
    split; [lia|reflexivity].
  - apply Z.eqb_neq in Exv.
    destruct Hin as [Hx|Hin]; [congruence|].
    destruct (IH v Hin) as [Hlt Heq]. simpl.
    split; [lia|exact Heq].
Qed.

Lemma argmax_spec : forall h,
  Forall (fun x => 0 <= x) h -> h <> [] ->
  (argmax h < length h)%nat /\ nth (argmax h) h 0 = zmax h.
Proof.
  intros h Hf Hne. unfold argmax.
  apply first_idx_spec. apply in_zmax; auto.
Qed.

(* ================= the contract ============================== *)

Lemma base_f_nonneg : forall h,
  0 < zsum h -> Forall (fun x => 0 <= x) h ->
  0 <= TOT - Z.of_nat (zcnt_pos h) ->
  Forall (fun x => 0 <= x) (base_f h).
Proof.
  intros h Ht Hf Hb.
  unfold base_f. rewrite Forall_map.
  eapply Forall_impl; [|exact Hf].
  intros x Hx. destruct (Z.gtb_spec x 0) as [Hg|Hg].
  - assert (0 <= x * (TOT - Z.of_nat (zcnt_pos h)) / zsum h)
      by (apply Z.div_pos;
          [apply Z.mul_nonneg_nonneg; lia | lia]).
    lia.
  - lia.
Qed.

Lemma base_f_sum : forall h,
  Forall (fun x => 0 <= x) h ->
  (length h <= Z.to_nat TOT)%nat -> 0 < zsum h ->
  zsum (base_f h) <= TOT.
Proof.
  intros h Hf HawT Ht.
  unfold base_f.
  assert (Hbud : 0 <= TOT - Z.of_nat (zcnt_pos h)).
  { pose proof (zcnt_pos_le_length h).
    unfold TOT in *. lia. }
  (* rewrite each summand as division + indicator *)
  assert (Hstep :
    map (fun hi => if hi >? 0
                   then hi * (TOT - Z.of_nat (zcnt_pos h)) / zsum h + 1
                   else 0) h
    = map (fun hi => hi * (TOT - Z.of_nat (zcnt_pos h)) / zsum h
                     + (if hi >? 0 then 1 else 0)) h).
  { apply map_ext_in. intros a Ha.
    assert (0 <= a)
      by (rewrite Forall_forall in Hf; apply Hf; exact Ha).
    destruct (Z.gtb_spec a 0) as [Hg|Hg].
    - lia.
    - assert (a = 0) by lia. subst a. reflexivity. }
  rewrite Hstep, zsum_map_add, zsum_indicator.
  (* Σ h_i*bud/tot <= (Σ h_i*bud)/tot = bud *)
  assert (Hme :
    map (fun hi => hi * (TOT - Z.of_nat (zcnt_pos h)) / zsum h) h
    = map (fun x => x / zsum h)
          (map (fun hi => hi * (TOT - Z.of_nat (zcnt_pos h))) h)).
  { rewrite map_map. reflexivity. }
  assert (FTL : Forall (fun x => 0 <= x)
    (map (fun hi => hi * (TOT - Z.of_nat (zcnt_pos h))) h)).
  { rewrite Forall_map. eapply Forall_impl; [|exact Hf].
    intros x Hx. apply Z.mul_nonneg_nonneg.
    - exact Hx.
    - exact Hbud. }
  rewrite Hme.
  pose proof (zsum_div_le _ _ FTL Ht) as Hle.
  rewrite zsum_map_mul in Hle.
  rewrite Z.div_mul in Hle by lia.
  lia.
Qed.

(* the nonzero branch: contract holds for ANY bi that attains the max
   (C picks the first one — a special case) *)
Theorem norm_nz_correct : forall h bi,
  Forall (fun x => 0 <= x) h ->
  (length h <= Z.to_nat TOT)%nat ->
  0 < zsum h ->
  (bi < length h)%nat -> nth bi h 0 = zmax h ->
  zsum (norm_nz h bi) = TOT
  /\ (length (norm_nz h bi) = length h)%nat
  /\ (forall i, (i < length h)%nat ->
        (nth i h 0 = 0 -> nth i (norm_nz h bi) 0 = 0)
        /\ (0 < nth i h 0 -> 0 < nth i (norm_nz h bi) 0 <= TOT)).
Proof.
  intros h bi Hf HawT Ht Hbi Hmax.
  assert (Hbud : 0 <= TOT - Z.of_nat (zcnt_pos h)).
  { pose proof (zcnt_pos_le_length h). unfold TOT in *. lia. }
  assert (Hbf : Forall (fun x => 0 <= x) (base_f h))
    by (apply base_f_nonneg; auto).
  assert (Hbsum : zsum (base_f h) <= TOT)
    by (apply base_f_sum; auto).
  assert (Hlen : (length (base_f h) = length h)%nat)
    by (unfold base_f; apply length_map).
  assert (Hmaxpos : 0 < zmax h).
  { assert (Hc : (0 < zcnt_pos h)%nat).
    { destruct (zcnt_pos h) eqn:Ez; [|lia].
      pose proof (zcnt_pos_zero h Hf Ez). lia. }
    unfold zcnt_pos in Hc.
    destruct (filter (fun x => x >? 0) h) as [|x0 fl] eqn:Ef;
      [simpl in Hc; lia|].
    assert (Hb : In x0 h /\ (x0 >? 0) = true).
    { assert (Hin0 : In x0 (filter (fun x => x >? 0) h))
        by (rewrite Ef; left; reflexivity).
      apply filter_In in Hin0. exact Hin0. }
    destruct Hb as [Hin Hb]. apply Z.gtb_lt in Hb.
    apply In_nth with (d := 0) in Hin.
    destruct Hin as (j & Hj & <-).
    pose proof (zmax_ge j h 0 Hf Hj). lia. }
  assert (Hfsum : zsum (norm_nz h bi) = TOT).
  { unfold norm_nz.
    rewrite (zsum_upd bi (base_f h) (TOT - zsum (base_f h))).
    - lia.
    - rewrite Hlen. exact Hbi. }
  assert (Hfnn : Forall (fun x => 0 <= x) (norm_nz h bi)).
  { unfold norm_nz. apply upd_nonneg; auto. lia. }
  assert (Hblen : (length (norm_nz h bi) = length h)%nat)
    by (unfold norm_nz; rewrite upd_length; exact Hlen).
  split; [exact Hfsum|].
  split; [exact Hblen|].
  intros i Hi. split; intros Hcond.
  - (* h_i = 0 -> f_i = 0 *)
    destruct (Nat.eq_dec i bi) as [->|Hneq].
    + (* impossible: bi attains the max, and max > 0 *)
      exfalso.
      assert (0 < nth bi h 0) by (rewrite Hmax; exact Hmaxpos).
      lia.
    + unfold norm_nz. rewrite nth_upd_neq by congruence.
      rewrite nth_base_f by exact Hi.
      rewrite Hcond. reflexivity.
  - (* h_i > 0 -> 0 < f_i <= TOT *)
    assert (Hbound : nth i (norm_nz h bi) 0 <= TOT).
    { pose proof (zsum_nth_le i (norm_nz h bi) 0 Hfnn
                    ltac:(rewrite Hblen; exact Hi)) as Hle.
      lia. }
    split; [|exact Hbound].
    destruct (Nat.eq_dec i bi) as [->|Hneq].
    + unfold norm_nz.
      rewrite (nth_upd_eq bi (base_f h)
                 (TOT - zsum (base_f h)) 0)
        by (rewrite Hlen; exact Hbi).
      rewrite nth_base_f by exact Hbi.
      destruct (Z.gtb_spec (nth bi h 0) 0) as [Hg|Hg].
      * assert (0 <= nth bi h 0 * (TOT - Z.of_nat (zcnt_pos h))
                     / zsum h)
          by (apply Z.div_pos;
              [apply Z.mul_nonneg_nonneg; lia | lia]).
        lia.
      * assert (0 < nth bi h 0) by (rewrite Hmax; exact Hmaxpos).
        lia.
    + unfold norm_nz. rewrite nth_upd_neq by congruence.
      rewrite nth_base_f by exact Hi.
      destruct (Z.gtb_spec (nth i h 0) 0) as [Hg|Hg].
      * assert (0 <= nth i h 0 * (TOT - Z.of_nat (zcnt_pos h))
                     / zsum h)
          by (apply Z.div_pos;
              [apply Z.mul_nonneg_nonneg; lia | lia]).
        lia.
      * lia.
Qed.

(* zero branch: uniform table, sum TOT, all cells positive *)
Theorem norm_z_correct : forall aw,
  (0 < aw)%nat -> (aw <= Z.to_nat TOT)%nat ->
  zsum (norm_z aw) = TOT /\ (length (norm_z aw) = aw)%nat /\
  Forall (fun x => 0 < x <= TOT) (norm_z aw).
Proof.
  intros aw Haw Htot.
  unfold norm_z.
  destruct aw as [|k]; [lia|].
  remember (TOT / Z.of_nat (S k)) as v eqn:Ev.
  remember (TOT - v * Z.of_nat (S k)) as r eqn:Er.
  assert (Hsk : 1 <= Z.of_nat (S k) <= TOT)
    by (unfold TOT in *; lia).
  assert (Hv : 0 < v).
  { rewrite Ev. apply Z.div_str_pos. lia. }
  assert (Hvle : v <= TOT).
  { rewrite Ev. apply Z.div_le_upper_bound; [lia|nia]. }
  assert (Hr : 0 <= r).
  { rewrite Er, Ev.
    pose proof (Z.div_mod TOT (Z.of_nat (S k)) ltac:(lia)) as Hdm.
    pose proof (Z.mod_pos_bound TOT (Z.of_nat (S k))
                   ltac:(lia)) as Hmb.
    lia. }
  split; [|split].
  - rewrite (zsum_upd k (repeat v (S k)) r).
    + rewrite zsum_repeat. lia.
    + rewrite repeat_length. lia.
  - rewrite upd_length, repeat_length. reflexivity.
  - rewrite Forall_nth. intros i d Hi.
    rewrite upd_length, repeat_length in Hi.
    destruct (Nat.eq_dec i k) as [->|Hneq].
    + rewrite (nth_upd_eq k (repeat v (S k)) r d)
        by (rewrite repeat_length; lia).
      assert (E1 : nth k (repeat v (S k)) d = v)
        by (apply nth_repeat'; lia).
      rewrite E1. split; [lia|]. rewrite Er. nia.
    + rewrite nth_upd_neq by congruence.
      assert (E1 : nth i (repeat v (S k)) d = v)
        by (apply nth_repeat'; exact Hi).
      rewrite E1. split; [exact Hv|exact Hvle].
Qed.

(* glue: norm_ctx itself, with the concrete first-max argmax *)
Theorem norm_ctx_contract : forall h,
  Forall (fun x => 0 <= x) h ->
  (0 < length h)%nat -> (length h <= Z.to_nat TOT)%nat ->
  zsum (norm_ctx h) = TOT
  /\ (length (norm_ctx h) = length h)%nat
  /\ Forall (fun x => 0 <= x <= TOT) (norm_ctx h)
  /\ (0 < zsum h ->
        forall i, (i < length h)%nat ->
          (nth i h 0 = 0 -> nth i (norm_ctx h) 0 = 0)
          /\ (0 < nth i h 0 -> 0 < nth i (norm_ctx h) 0))
  /\ (zsum h = 0 -> Forall (fun x => 0 < x) (norm_ctx h)).
Proof.
  intros h Hf Haw0 HawT.
  unfold norm_ctx.
  destruct (zcnt_pos h =? 0)%nat eqn:Ez.
  - (* zero branch *)
    apply Nat.eqb_eq in Ez.
    pose proof (zcnt_pos_zero h Hf Ez) as Hz.
    destruct (norm_z_correct (length h) Haw0 HawT)
      as (Hs & Hl & Hpos).
    split; [|split; [|split; [|split]]].
    + exact Hs.
    + exact Hl.
    + eapply Forall_impl; [|exact Hpos].
      intros x Hx. destruct Hx. lia.
    + intros Hp. exfalso. lia.
    + intros _. eapply Forall_impl; [|exact Hpos].
      intros x Hx. destruct Hx. lia.
  - (* nonzero branch *)
    apply Nat.eqb_neq in Ez.
    assert (Hs : 0 < zsum h).
    { apply zcnt_pos_sum_pos; [exact Hf|].
      destruct (zcnt_pos h); [congruence|lia]. }
    destruct (argmax_spec h Hf) as [Hlt Heq].
    { destruct h as [|x0 tl0]; [simpl in Haw0; lia|discriminate]. }
    destruct (norm_nz_correct h (argmax h) Hf HawT Hs Hlt Heq)
      as (Hsum & Hlen & Hsup).
    assert (Hfnn : Forall (fun x => 0 <= x)
                      (norm_nz h (argmax h))).
    { unfold norm_nz. apply upd_nonneg.
      - apply base_f_nonneg; auto.
        pose proof (zcnt_pos_le_length h). unfold TOT in *. lia.
      - pose proof (base_f_sum h Hf HawT Hs). lia. }
    split; [|split; [|split; [|split]]].
    + exact Hsum.
    + exact Hlen.
    + rewrite Forall_nth. intros i d Hi. split.
      * rewrite Forall_forall in Hfnn.
        apply Hfnn. apply nth_In. exact Hi.
      * pose proof (zsum_nth_le i (norm_nz h (argmax h)) d Hfnn Hi)
          as Hle.
        rewrite Hsum in Hle. exact Hle.
    + intros _ i Hi. destruct (Hsup i Hi) as [Hz Hp]. split.
      * exact Hz.
      * intros Hg. destruct (Hp Hg) as [Hp0 _]. exact Hp0.
    + intros Hz0. exfalso. lia.
Qed.

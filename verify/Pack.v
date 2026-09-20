(* Pack.v — Coq proof of the pack_enc/pack_dec bitstream round trip.
 *
 * Algorithm-level proof, same tier as Rans.v / Utf8.v / Norm.v.  PACK is
 * caiw's dictionary+bitpacking method: it stores the k distinct atoms
 * (k <= 256) as an ascending little-endian dictionary, then packs each
 * element's dictionary index into ceil(log2 k)-bit MSB-first fields.
 * pack_dec reads the fields back and expands through the dictionary.
 * CBMC/ESBMC prove the decoder's bit-index bounds only for bounded
 * streams (BL<=48, n<=36); here we prove the full semantic round trip
 *     dec(enc(data)) = data
 * for ARBITRARY element counts and both atom widths (bsz in {1,2}).
 *
 * C model (caiw.c 1263-1328):
 *   enc: k = |distinct atoms|; dict = present atoms in ascending order;
 *        idx_i = position of atom_i in dict (rmap); emit le4 k, le2 dict
 *        entries, then stream bit (i*ib + t) = bit (ib-1-t) of idx_i
 *        (ib = ceil(log2 k), MSB-first, |= into a zeroed buffer).
 *   dec: read k (le4) and the k dict byte-pairs; per element accumulate
 *        ib stream bits MSB-first into idx; emit dict[2*idx] and, for
 *        bsz=2, dict[2*idx+1].
 *
 * Faithfulness notes:
 *  - The |= accumulation on a zeroed buffer is modeled functionally:
 *    `bits` IS the list of emitted bit values.  Each stream position is
 *    written exactly once (element groups are disjoint), so the OR-sum
 *    equals the functional assignment.
 *  - dec's bounds/idx<k/exact-consumption checks are absent from the
 *    model — on honest encoder output they always pass (idx = position
 *    < k); the BMC layers prove them for hostile inputs.
 *  - The length precondition bsz | len is real: PACK is only selected
 *    when len is a multiple of the atom size (bsz=2 is gated on the
 *    f16 dtypes whose len is always even).
 *)

From Stdlib Require Import ZArith List Bool Lia.
Import ListNotations.
Open Scope Z_scope.

(* ================= generic helpers ================= *)

Definition byteb (b : Z) : Prop := 0 <= b < 256.
Definition bit01 (b : Z) : Prop := 0 <= b < 2.

Lemma zofnat_nonneg : forall n : nat, 0 <= Z.of_nat n.
Proof. intros n. replace 0 with (Z.of_nat 0) by reflexivity.
  rewrite <- Nat2Z.inj_le. lia. Qed.

Lemma zofnat_lt : forall n m : nat, (n < m)%nat -> Z.of_nat n < Z.of_nat m.
Proof. intros n m H. rewrite <- Nat2Z.inj_lt. exact H. Qed.

Lemma zpow2_pos : forall n : nat, 0 < 2 ^ Z.of_nat n.
Proof.
  induction n as [|n IH].
  - simpl. lia.
  - rewrite Nat2Z.inj_succ. rewrite Z.pow_succ_r by lia. lia.
Qed.

Lemma map_nth : forall (A B : Type) (f : A -> B) l i d da,
  (i < length l)%nat -> nth i (map f l) d = f (nth i l da).
Proof.
  induction l as [|x l IH]; intros i d da Hi; simpl in *; [lia|].
  destruct i as [|i']; simpl.
  - reflexivity.
  - apply IH. lia.
Qed.

Lemma seq_Sn : forall s n, seq s (S n) = seq s n ++ [(s + n)%nat].
Proof.
  intros s n; revert s; induction n as [|n IH]; intros s.
  - replace (s + 0)%nat with s by lia. reflexivity.
  - change (seq s (S (S n))) with (s :: seq (S s) (S n)).
    rewrite (IH (S s)).
    change (seq s (S n)) with (s :: seq (S s) n).
    rewrite <- app_comm_cons.
    assert (E2 : (S s + n = s + S n)%nat) by lia.
    rewrite E2. reflexivity.
Qed.

Lemma firstn_app_add : forall (a b : nat) (l : list Z),
  firstn (a + b) l = firstn a l ++ firstn b (skipn a l).
Proof.
  induction a as [|a IH]; intros b l.
  - reflexivity.
  - destruct l as [|x l]; simpl.
    + destruct b; reflexivity.
    + f_equal. apply IH.
Qed.

Lemma flat_map_ext_in : forall (A B : Type) (f g : A -> list B) l,
  (forall a, In a l -> f a = g a) -> flat_map f l = flat_map g l.
Proof.
  induction l as [|x l IH]; simpl; intros H; auto.
  rewrite H by auto. rewrite IH by auto. reflexivity.
Qed.

Lemma Forall_firstn : forall n (l : list Z) P,
  Forall P l -> Forall P (firstn n l).
Proof.
  induction n as [|n IH]; intros l P Hf; simpl.
  - constructor.
  - destruct l; simpl; auto. inversion Hf; subst. constructor; auto.
Qed.

Lemma Forall_skipn : forall n (l : list Z) P,
  Forall P l -> Forall P (skipn n l).
Proof.
  induction n as [|n IH]; intros l P Hf; simpl; auto.
  destruct l; simpl; auto. inversion Hf; subst. auto.
Qed.

Lemma Forall_flat_map : forall (A : Type) (f : A -> list Z) l P,
  (forall x, In x l -> Forall P (f x)) -> Forall P (flat_map f l).
Proof.
  intros A f l P; induction l as [|x l IH]; simpl; intros H.
  - constructor.
  - rewrite Forall_forall. intros y Hy.
    apply in_app_iff in Hy. destruct Hy as [Hy|Hy].
    + pose proof (H x ltac:(left; reflexivity)) as Hx.
      rewrite Forall_forall in Hx. apply (Hx y Hy).
    + pose proof (IH (fun a Ha => H a (or_intror Ha))) as Hfl.
      rewrite Forall_forall in Hfl. apply (Hfl y Hy).
Qed.

Lemma Forall_nth_Z : forall (P : Z -> Prop) l i,
  Forall P l -> (i < length l)%nat -> P (nth i l 0).
Proof.
  intros P l i Hf Hi. rewrite Forall_forall in Hf.
  apply Hf. apply nth_In. exact Hi.
Qed.

Lemma length_flat_map : forall (A : Type) (f : A -> list Z) (l : list A) m,
  (forall x, In x l -> (length (f x) = m)%nat) ->
  (length (flat_map f l) = length l * m)%nat.
Proof.
  intros A f l m; induction l as [|x l IH]; simpl; intros H; auto.
  rewrite length_app. rewrite (H x) by (left; reflexivity).
  rewrite IH by (intros y Hy; apply H; right; exact Hy). lia.
Qed.

Lemma nth_flat_map : forall (A : Type) (f : A -> list Z) (l : list A) m i j da,
  (forall x, In x l -> (length (f x) = m)%nat) ->
  (i < length l)%nat -> (j < m)%nat ->
  nth (i * m + j)%nat (flat_map f l) 0 = nth j (f (nth i l da)) 0.
Proof.
  induction l as [|x l IH]; intros m i j da Hlen Hi Hj; simpl in *.
  - lia.
  - destruct i as [|i'].
    + simpl. rewrite app_nth1; auto. rewrite (Hlen x) by auto. exact Hj.
    + replace (S i' * m + j)%nat with (length (f x) + (i' * m + j))%nat
        by (rewrite (Hlen x) by auto; lia).
      rewrite app_nth2 by lia.
      replace (length (f x) + (i' * m + j) - length (f x))%nat
        with (i' * m + j)%nat by lia.
      rewrite (IH m i' j da
                 ltac:(intros y Hy; apply Hlen; auto)
                 ltac:(lia)
                 ltac:(exact Hj)).
      reflexivity.
Qed.

(* ================= bit groups ================= *)

(* bitsv: MSB-first bit group -> value; the decoder's (idx<<1)|bit fold. *)
Definition bitsv (g : list Z) : Z := fold_left (fun a b => 2 * a + b) g 0.

Lemma fold_bitsv : forall g z,
  fold_left (fun a b => 2 * a + b) g z
  = z * 2 ^ Z.of_nat (length g) + fold_left (fun a b => 2 * a + b) g 0.
Proof.
  induction g as [|b g IH]; intros z.
  - simpl. ring.
  - cbn [fold_left List.length].
    rewrite (IH (2 * z + b)). rewrite (IH (2 * 0 + b)).
    rewrite Nat2Z.inj_succ. rewrite Z.pow_succ_r by lia. ring.
Qed.

Lemma bitsv_cons : forall b g,
  bitsv (b :: g) = b * 2 ^ Z.of_nat (length g) + bitsv g.
Proof.
  intros b g. unfold bitsv. cbn [fold_left]. rewrite fold_bitsv. ring.
Qed.

Lemma bitsv_bound : forall g,
  Forall bit01 g -> 0 <= bitsv g < 2 ^ Z.of_nat (length g).
Proof.
  induction g as [|b g IH]; intros Hf.
  - unfold bitsv; simpl. split; lia.
  - inversion Hf as [|? ? Hb Htl]; subst.
    destruct (IH Htl) as [Hv0 Hv1]. destruct Hb as [Hb0 Hb1].
    rewrite bitsv_cons. simpl (length _).
    rewrite Nat2Z.inj_succ. rewrite Z.pow_succ_r by lia. nia.
Qed.

(* element t (MSB-first) of g sits at bit position length g - 1 - t *)
Lemma nth_bitsv : forall g t,
  Forall bit01 g -> (t < length g)%nat ->
  nth t g 0 = (bitsv g / 2 ^ Z.of_nat (length g - 1 - t)%nat) mod 2.
Proof.
  induction g as [|b g IH]; intros t Hf Ht.
  - simpl in Ht; lia.
  - inversion Hf as [|? ? Hb Htl]; subst.
    destruct (bitsv_bound g Htl) as [Hv0 Hv1].
    destruct Hb as [Hb0 Hb1].
    destruct t as [|t'].
    + cbn [nth]. rewrite bitsv_cons.
      replace (length (b :: g) - 1 - 0)%nat with (length g)
        by (simpl; lia).
      rewrite Z.div_add_l by (pose proof (zpow2_pos (length g)); lia).
      rewrite (Z.div_small (bitsv g)) by lia.
      rewrite Z.add_0_r. rewrite Z.mod_small by lia. reflexivity.
    + cbn [nth].
      rewrite (IH t' Htl ltac:(simpl in Ht; lia)).
      simpl in Ht.
      rewrite bitsv_cons.
      replace (length (b :: g) - 1 - S t')%nat
        with (length g - 1 - t')%nat by (simpl; lia).
      set (p := Z.of_nat (length g - 1 - t')%nat).
      assert (Hp : (length g - 1 - t' < length g)%nat) by lia.
      replace (b * 2 ^ Z.of_nat (length g))
        with ((b * 2 ^ Z.of_nat (S t')) * 2 ^ p).
      2: { rewrite <- Z.mul_assoc.
           rewrite <- Z.pow_add_r by (try unfold p; lia).
           unfold p. rewrite <- Nat2Z.inj_add. f_equal. f_equal. f_equal. lia. }
      rewrite Z.div_add_l
        by (unfold p; pose proof (zpow2_pos (length g - 1 - t')); lia).
      replace (b * 2 ^ Z.of_nat (S t') + bitsv g / 2 ^ p)
        with (bitsv g / 2 ^ p + (b * 2 ^ Z.of_nat t') * 2).
      2: { rewrite Nat2Z.inj_succ. rewrite Z.pow_succ_r by lia. ring. }
      rewrite Z.mod_add by lia. reflexivity.
Qed.

(* ================= encoder bit stream ================= *)

(* bits_of ib x: the ib bits of x, MSB first — element t is bit (ib-1-t). *)
Definition bits_of (ib : nat) (x : Z) : list Z :=
  map (fun t => x / 2 ^ Z.of_nat t mod 2) (rev (seq 0 ib)).

Lemma bits_of_len : forall ib x, (length (bits_of ib x) = ib)%nat.
Proof.
  intros. unfold bits_of. rewrite length_map, length_rev, length_seq. auto.
Qed.

Lemma bits_of_bit01 : forall ib x, Forall bit01 (bits_of ib x).
Proof.
  intros ib x. apply Forall_forall. intros b Hb.
  unfold bits_of in Hb. apply in_map_iff in Hb.
  destruct Hb as [t [<- _]]. unfold bit01.
  pose proof (Z.mod_pos_bound (x / 2 ^ Z.of_nat t) 2 ltac:(lia)). lia.
Qed.

Lemma bitsv_bits_of : forall ib x,
  0 <= x -> bitsv (bits_of ib x) = x mod 2 ^ Z.of_nat ib.
Proof.
  induction ib as [|ib IH]; intros x Hx.
  - unfold bits_of, bitsv; simpl. rewrite Z.mod_1_r. reflexivity.
  - assert (E : bits_of (S ib) x = (x / 2 ^ Z.of_nat ib mod 2) :: bits_of ib x).
    { unfold bits_of. rewrite seq_Sn. rewrite rev_app_distr.
      cbn [rev]. rewrite map_app. cbn [map]. reflexivity. }
    rewrite E. rewrite bitsv_cons. rewrite bits_of_len. rewrite IH by lia.
    pose proof (zpow2_pos ib) as Hm.
    pose proof (Z.div_mod x (2 ^ Z.of_nat ib) ltac:(lia)) as H1.
    pose proof (Z.div_mod (x / 2 ^ Z.of_nat ib) 2 ltac:(lia)) as H2.
    rewrite Z.div_div in H2 by lia.
    assert (HS : 2 ^ Z.of_nat ib * 2 = 2 ^ Z.of_nat (S ib)).
    { rewrite Nat2Z.inj_succ. rewrite Z.pow_succ_r by lia. ring. }
    pose proof (Z.div_mod x (2 ^ Z.of_nat (S ib))
                  ltac:(rewrite <- HS; lia)) as H3.
    pose proof (Z.mod_pos_bound x (2 ^ Z.of_nat (S ib))
                  ltac:(rewrite <- HS; lia)) as Hr1.
    pose proof (Z.mod_pos_bound (x / 2 ^ Z.of_nat ib) 2 ltac:(lia)) as Hbb.
    pose proof (Z.mod_pos_bound x (2 ^ Z.of_nat ib) ltac:(lia)) as Hr.
    destruct (Z.div_mod_unique (2 ^ Z.of_nat (S ib))
                (x / 2 ^ Z.of_nat (S ib))
                (x / 2 ^ Z.of_nat (S ib))
                (x mod 2 ^ Z.of_nat (S ib))
                ((x / 2 ^ Z.of_nat ib mod 2) * 2 ^ Z.of_nat ib
                 + x mod 2 ^ Z.of_nat ib)) as [_ Hres].
    + left. exact Hr1.
    + left. rewrite <- HS. split; nia.
    + rewrite <- H3. rewrite <- HS. nia.
    + symmetry. exact Hres.
Qed.

(* ================= byte packing ================= *)

(* stream bit j lives in byte j/8 at in-byte position 7 - (j mod 8) *)
Definition bit_of (bs : list Z) (j : nat) : Z :=
  (nth (j / 8)%nat bs 0 / 2 ^ Z.of_nat (7 - j mod 8)%nat) mod 2.

Definition chunk (l : list Z) (q : nat) : list Z := firstn 8 (skipn (8 * q) l).
Definition nchunk (l : list Z) : nat := (length l + 7) / 8.

(* byte q = the (up to) 8 stream bits MSB-first, low bits zero-padded *)
Definition bytes_of (bits : list Z) : list Z :=
  map (fun q => bitsv (chunk bits q)
                  * 2 ^ Z.of_nat (8 - length (chunk bits q))%nat)
      (seq 0 (nchunk bits)).

Lemma bit_of_bytes_of : forall bits j,
  Forall bit01 bits -> (j < length bits)%nat ->
  bit_of (bytes_of bits) j = nth j bits 0.
Proof.
  intros bits j Hf Hj.
  pose proof (Nat.div_mod j 8 ltac:(lia)) as Hjd.
  pose proof (Nat.mod_upper_bound j 8 ltac:(lia)) as Hr8.
  set (q := (j / 8)%nat). set (r := (j mod 8)%nat).
  assert (Hq : (q < nchunk bits)%nat).
  { unfold nchunk, q.
    pose proof (Nat.div_mod (length bits + 7) 8 ltac:(lia)) as Hcd.
    pose proof (Nat.mod_upper_bound (length bits + 7) 8 ltac:(lia)) as Hcm.
    lia. }
  unfold bit_of, bytes_of. fold q r.
  rewrite map_nth with (da := 0%nat) by (rewrite length_seq; exact Hq).
  rewrite seq_nth by exact Hq.
  set (cq := chunk bits q).
  assert (Hl : (length cq = Nat.min 8 (length bits - 8 * q))%nat).
  { unfold cq, chunk. rewrite length_firstn, length_skipn. reflexivity. }
  assert (Hrq : (r < length cq)%nat).
  { rewrite Hl.
    destruct (Nat.min_dec 8 (length bits - 8 * q)) as [Hm|Hm]; rewrite Hm;
    unfold q, r in *; lia. }
  assert (Hcq : Forall bit01 cq).
  { unfold cq, chunk. apply Forall_firstn. apply Forall_skipn. exact Hf. }
  pose proof (bitsv_bound cq Hcq) as Hvb.
  replace (7 - r)%nat
    with ((length cq - 1 - r) + (8 - length cq))%nat by lia.
  rewrite Nat2Z.inj_add. rewrite Z.pow_add_r by lia.
  rewrite (Z.mul_comm (2 ^ Z.of_nat (length cq - 1 - r)%nat)
                      (2 ^ Z.of_nat (8 - length cq)%nat)).
  rewrite <- Z.div_div
    by (pose proof (zpow2_pos (8 - length cq)); lia).
  rewrite Z.div_mul
    by (pose proof (zpow2_pos (8 - length cq)); lia).
  rewrite <- nth_bitsv by (first [exact Hcq | exact Hrq]).
  unfold cq, chunk.
  rewrite nth_firstn.
  destruct (r <? 8)%nat eqn:Er.
  - rewrite nth_skipn. f_equal. unfold q, r. lia.
  - apply Nat.ltb_ge in Er. lia.
Qed.

(* ================= dictionary ================= *)

Definition aw_of (bsz : Z) : nat := if Z.eqb bsz 2 then 65536%nat else 256%nat.

Definition atom_val (bsz : Z) (data : list Z) (i : nat) : Z :=
  if Z.eqb bsz 2
  then nth (2 * i)%nat data 0 + 256 * nth (2 * i + 1)%nat data 0
  else nth i data 0.

Definition dict_of (atoms : list Z) (bsz : Z) : list Z :=
  filter (fun a => if in_dec Z.eq_dec a atoms then true else false)
         (map Z.of_nat (seq 0 (aw_of bsz))).

Lemma in_map_seq : forall a aw,
  In a (map Z.of_nat (seq 0 aw)) <-> (0 <= a < Z.of_nat aw).
Proof.
  intros a aw. rewrite in_map_iff. split.
  - intros [x [Hx Hin]]. apply in_seq in Hin. subst a.
    split; [apply zofnat_nonneg|]. apply zofnat_lt. lia.
  - intros Ha. exists (Z.to_nat a). split.
    + apply Z2Nat.id. lia.
    + apply in_seq. split; [lia|].
      rewrite <- (Nat2Z.id aw). apply Z2Nat.inj_lt; lia.
Qed.

Lemma in_dict : forall bsz atoms a,
  0 <= a < Z.of_nat (aw_of bsz) ->
  (In a (dict_of atoms bsz) <-> In a atoms).
Proof.
  intros bsz atoms a Ha. unfold dict_of. rewrite filter_In.
  rewrite in_map_seq. split.
  - intros [_ Hc]. destruct (in_dec Z.eq_dec a atoms); [exact i|discriminate].
  - intros Hin. split; [exact Ha|].
    destruct (in_dec Z.eq_dec a atoms); [reflexivity|contradiction].
Qed.

Lemma atom_bound : forall bsz data i,
  Forall byteb data -> (bsz = 1 \/ bsz = 2) ->
  ((if Z.eqb bsz 2 then 2 * i + 1 else i) < length data)%nat ->
  0 <= atom_val bsz data i < Z.of_nat (aw_of bsz).
Proof.
  intros bsz data i Hf Hbsz Hi.
  destruct Hbsz as [->| ->]; unfold atom_val, aw_of; simpl in Hi.
  - replace (Z.eqb 1 2) with false by reflexivity. cbv iota.
    change (Z.of_nat 256) with 256.
    pose proof (Forall_nth_Z _ _ _ Hf Hi) as [H0 H1]. split; lia.
  - replace (Z.eqb 2 2) with true by reflexivity. cbv iota.
    change (Z.of_nat 65536) with 65536.
    pose proof (Forall_nth_Z _ _ (2 * i)%nat Hf ltac:(lia)) as [H0a H0b].
    pose proof (Forall_nth_Z _ _ (2 * i + 1)%nat Hf ltac:(lia)) as [H1a H1b].
    split; lia.
Qed.

Lemma in_length_pos : forall (a : Z) l, In a l -> (0 < length l)%nat.
Proof.
  intros a l H. apply in_split in H. destruct H as [l1 [l2 ->]].
  rewrite length_app. simpl. lia.
Qed.

Fixpoint idx_of (a : Z) (l : list Z) : nat :=
  match l with
  | [] => 0%nat
  | x :: t => if Z.eq_dec x a then 0%nat else S (idx_of a t)
  end.

Lemma idx_of_spec : forall a l, In a l ->
  (idx_of a l < length l)%nat /\ nth (idx_of a l) l 0 = a.
Proof.
  induction l as [|x t IH]; intros Hin.
  - contradiction.
  - simpl. destruct (Z.eq_dec x a) as [->|Hne].
    + split; [lia|reflexivity].
    + destruct Hin as [Heq|Hin']; [contradiction|].
      destruct (IH Hin') as [H1 H2]. split; [lia|exact H2].
Qed.

Definition ib_of (k : nat) : nat := Nat.log2_up k.

Lemma ib_spec : forall k, (1 <= k)%nat -> (k <= 2 ^ ib_of k)%nat.
Proof.
  intros k Hk. unfold ib_of.
  destruct k as [|[|k']].
  - lia.
  - vm_compute. lia.
  - destruct (Nat.log2_up_spec (S (S k')) ltac:(lia)) as [_ H]. exact H.
Qed.

Lemma ib_bound : forall k, (1 <= k)%nat ->
  Z.of_nat k <= 2 ^ Z.of_nat (ib_of k).
Proof.
  intros k Hk.
  replace (2 : Z) with (Z.of_nat 2) by reflexivity.
  rewrite <- Nat2Z.inj_pow.
  rewrite <- Nat2Z.inj_le. apply ib_spec. exact Hk.
Qed.

(* ================= little-endian fields ================= *)

Definition le2 (a : Z) : list Z := [a mod 256; a / 256 mod 256].

Definition le4 (a : Z) : list Z :=
  [a mod 256; a / 256 mod 256; a / 65536 mod 256; a / 16777216 mod 256].

Definition le4v (b : list Z) : Z :=
  nth 0 b 0 + 256 * nth 1 b 0 + 65536 * nth 2 b 0 + 16777216 * nth 3 b 0.

Lemma le4_inv : forall a, 0 <= a < 4294967296 -> le4v (le4 a) = a.
Proof.
  intros a Ha. unfold le4v, le4. cbn [nth].
  pose proof (Z.div_mod a 256 ltac:(lia)) as H1.
  pose proof (Z.div_mod (a / 256) 256 ltac:(lia)) as H2.
  pose proof (Z.div_mod (a / 65536) 256 ltac:(lia)) as H3.
  rewrite Z.div_div in H2 by lia.
  replace (256 * 256) with 65536 in H2 by ring.
  rewrite Z.div_div in H3 by lia.
  replace (65536 * 256) with 16777216 in H3 by ring.
  assert (H4 : a / 16777216 < 256).
  { assert (Hq : 0 <= a / 16777216) by (apply Z.div_pos; lia).
    pose proof (Z.div_mod a 16777216 ltac:(lia)) as Hd.
    pose proof (Z.mod_pos_bound a 16777216 ltac:(lia)) as Hrb. lia. }
  rewrite (Z.mod_small (a / 16777216)) by (split; [apply Z.div_pos|]; lia).
  lia.
Qed.

Lemma nth_app_add : forall (l1 l2 : list Z) k d,
  nth (length l1 + k)%nat (l1 ++ l2) d = nth k l2 d.
Proof.
  intros. rewrite app_nth2 by lia. f_equal. lia.
Qed.

Lemma nth_le2_lo : forall dict i,
  nth (2 * i)%nat (flat_map le2 dict) 0 = nth i dict 0 mod 256.
Proof.
  induction dict as [|a t IH]; intros i.
  - destruct i; reflexivity.
  - cbn [flat_map].
    destruct i as [|i'].
    + reflexivity.
    + replace (2 * S i')%nat with (length (le2 a) + 2 * i')%nat
        by (unfold le2; simpl; lia).
      rewrite nth_app_add. rewrite IH.
      cbn [nth]. reflexivity.
Qed.

Lemma nth_le2_hi : forall dict i,
  nth (2 * i + 1)%nat (flat_map le2 dict) 0 = nth i dict 0 / 256 mod 256.
Proof.
  induction dict as [|a t IH]; intros i.
  - destruct i; reflexivity.
  - cbn [flat_map].
    destruct i as [|i'].
    + reflexivity.
    + replace (2 * S i' + 1)%nat with (length (le2 a) + (2 * i' + 1))%nat
        by (unfold le2; simpl; lia).
      rewrite nth_app_add. rewrite IH.
      cbn [nth]. reflexivity.
Qed.

Lemma length_le2 : forall dict,
  (length (flat_map le2 dict) = 2 * length dict)%nat.
Proof.
  intros. rewrite length_flat_map with (m := 2%nat); auto; lia.
Qed.

(* ================= record models ================= *)

Definition enc_record (bsz : Z) (data : list Z) : list Z :=
  let bszn := Z.to_nat bsz in
  let ne := (length data / bszn)%nat in
  let atoms := map (atom_val bsz data) (seq 0 ne) in
  let dict := dict_of atoms bsz in
  let k := length dict in
  let ib := ib_of k in
  let idxs := map (fun a => Z.of_nat (idx_of a dict)) atoms in
  let bits := flat_map (fun x => bits_of ib x) idxs in
  le4 (Z.of_nat k) ++ flat_map le2 dict ++ bytes_of bits.

Definition read_idx (stream : list Z) (ib i : nat) : Z :=
  bitsv (map (fun t => bit_of stream (i * ib + t)%nat) (seq 0 ib)).

Definition dec_model (rec : list Z) (bsz : Z) (n : nat) : list Z :=
  let k := Z.to_nat (le4v (firstn 4 rec)) in
  let dictb := firstn (2 * k)%nat (skipn 4 rec) in
  let ib := ib_of k in
  let ne := (n / Z.to_nat bsz)%nat in
  let stream := skipn (4 + 2 * k)%nat rec in
  flat_map (fun i =>
    let idx := Z.to_nat (read_idx stream ib i) in
    if Z.eqb bsz 2
    then [nth (2 * idx)%nat dictb 0; nth (2 * idx + 1)%nat dictb 0]
    else [nth (2 * idx)%nat dictb 0]) (seq 0 ne).

Lemma read_idx_ok : forall stream bits idxs ib i,
  Forall bit01 bits ->
  Forall (fun x => 0 <= x) idxs ->
  bits = flat_map (fun x => bits_of ib x) idxs ->
  stream = bytes_of bits ->
  (i < length idxs)%nat ->
  read_idx stream ib i = nth i idxs 0 mod 2 ^ Z.of_nat ib.
Proof.
  intros stream bits idxs ib i Hb Hnn Hbits Hst Hi.
  unfold read_idx. subst stream bits.
  assert (El : (length (flat_map (fun x => bits_of ib x) idxs)
                = length idxs * ib)%nat).
  { apply length_flat_map. intros x _. apply bits_of_len. }
  assert (H01 : Forall bit01 (flat_map (fun x => bits_of ib x) idxs)).
  { apply Forall_flat_map. intros x _. apply bits_of_bit01. }
  assert (Em : map (fun t => bit_of (bytes_of
                     (flat_map (fun x => bits_of ib x) idxs))
                     (i * ib + t)%nat) (seq 0 ib)
             = bits_of ib (nth i idxs 0)).
  { apply nth_ext with (d := 0) (d' := 0).
    - rewrite length_map, length_seq, bits_of_len. reflexivity.
    - intros n Hn. rewrite length_map, length_seq in Hn.
      rewrite map_nth with (da := 0%nat)
        by (rewrite length_seq; exact Hn).
      rewrite seq_nth by exact Hn.
      change (0 + n)%nat with n.
      assert (Hlt : (i * ib + n < length (flat_map (fun x => bits_of ib x) idxs))%nat)
        by (rewrite El; nia).
      rewrite bit_of_bytes_of by auto.
      rewrite (nth_flat_map _ _ idxs ib i n 0)
        by (first [intros x _; apply bits_of_len | exact Hi | exact Hn]).
      reflexivity. }
  rewrite Em. rewrite bitsv_bits_of; auto.
  apply (Forall_nth_Z _ _ _ Hnn Hi).
Qed.

Lemma flat_map_slices : forall n m (l : list Z),
  flat_map (fun i => firstn m (skipn (m * i) l)) (seq 0 n)
  = firstn (m * n) l.
Proof.
  induction n as [|n IH]; intros m l.
  - replace (m * 0)%nat with 0%nat by lia. reflexivity.
  - rewrite seq_Sn. rewrite flat_map_app. rewrite IH.
    cbn [flat_map]. rewrite app_nil_r.
    replace (m * S n)%nat with (m * n + m)%nat by lia.
    rewrite firstn_app_add.
    replace (m * (0 + n))%nat with (m * n)%nat by lia.
    reflexivity.
Qed.

Lemma firstn1_nth : forall l, (1 <= length l)%nat ->
  firstn 1 l = [nth 0 l 0].
Proof.
  intros [|x t] Hl; simpl in *; [lia|reflexivity].
Qed.

Lemma firstn2_nth : forall l, (2 <= length l)%nat ->
  firstn 2 l = [nth 0 l 0; nth 1 l 0].
Proof.
  intros [|x [|y t]] Hl; simpl in *; try lia. reflexivity.
Qed.

Lemma firstn4_le4 : forall a l, firstn 4 (le4 a ++ l) = le4 a.
Proof.
  intros a l. rewrite firstn_app.
  assert (Hl : (length (le4 a) = 4)%nat) by reflexivity.
  rewrite Hl.
  rewrite firstn_all2 by (rewrite Hl; lia).
  replace (4 - 4)%nat with 0%nat by lia. cbn [firstn].
  rewrite app_nil_r. reflexivity.
Qed.

Lemma skipn4_le4 : forall a l, skipn 4 (le4 a ++ l) = l.
Proof.
  intros a l. rewrite skipn_app.
  assert (Hl : (length (le4 a) = 4)%nat) by reflexivity.
  rewrite Hl.
  rewrite skipn_all2 by (rewrite Hl; lia).
  replace (4 - 4)%nat with 0%nat by lia. cbn.
  reflexivity.
Qed.

Lemma dictb_eq : forall dict l,
  firstn (2 * length dict) (flat_map le2 dict ++ l) = flat_map le2 dict.
Proof.
  intros dict l. rewrite firstn_app. rewrite length_le2.
  replace (2 * length dict - 2 * length dict)%nat with 0%nat by lia.
  rewrite firstn_all2 by (rewrite length_le2; lia).
  cbn. rewrite app_nil_r. reflexivity.
Qed.

Lemma stream_eq : forall a dict s,
  skipn (4 + 2 * length dict)%nat (le4 a ++ (flat_map le2 dict ++ s)) = s.
Proof.
  intros a dict s. rewrite skipn_app.
  assert (Hl : (length (le4 a) = 4)%nat) by reflexivity.
  rewrite Hl. rewrite skipn_all2 by lia.
  replace (4 + 2 * length dict - 4)%nat with (2 * length dict)%nat by lia.
  rewrite skipn_app. rewrite length_le2.
  rewrite skipn_all2 by (rewrite length_le2; lia).
  replace (2 * length dict - 2 * length dict)%nat with 0%nat by lia.
  cbn. reflexivity.
Qed.

(* per-element decode correctness *)
Lemma emit_eq : forall bsz data atoms dict bits ib ne i,
  (bsz = 1 \/ bsz = 2) ->
  Forall byteb data ->
  (length data = ne * Z.to_nat bsz)%nat ->
  atoms = map (atom_val bsz data) (seq 0 ne) ->
  dict = dict_of atoms bsz ->
  (1 <= length dict)%nat ->
  bits = flat_map (fun x => bits_of ib x)
                  (map (fun a => Z.of_nat (idx_of a dict)) atoms) ->
  ib = ib_of (length dict) ->
  (i < ne)%nat ->
  (let idx := Z.to_nat (read_idx (bytes_of bits) ib i) in
   if Z.eqb bsz 2
   then [nth (2 * idx)%nat (flat_map le2 dict) 0;
         nth (2 * idx + 1)%nat (flat_map le2 dict) 0]
   else [nth (2 * idx)%nat (flat_map le2 dict) 0])
  = firstn (Z.to_nat bsz) (skipn (Z.to_nat bsz * i) data).
Proof.
  intros bsz data atoms dict bits ib ne i Hbsz Hb Hlen Hat Hdict Hk1 Hbits Hib Hi.
  subst atoms dict ib. subst bits.
  remember (map (fun a => Z.of_nat (idx_of a (dict_of
                     (map (atom_val bsz data) (seq 0 ne)) bsz)))
                (map (atom_val bsz data) (seq 0 ne))) as idxs eqn:Hidxs.
  assert (Ha_nth : nth i (map (atom_val bsz data) (seq 0 ne)) 0
                   = atom_val bsz data i).
  { rewrite map_nth with (da := 0%nat) by (rewrite length_seq; exact Hi).
    rewrite seq_nth by exact Hi. reflexivity. }
  assert (Hiat : In (atom_val bsz data i)
                    (map (atom_val bsz data) (seq 0 ne))).
  { apply in_map. apply in_seq. lia. }
  assert (Hab : 0 <= atom_val bsz data i
                < Z.of_nat (aw_of bsz)).
  { apply atom_bound; auto.
    destruct Hbsz as [->| ->];
      [ replace (Z.eqb 1 2) with false by reflexivity
      | replace (Z.eqb 2 2) with true by reflexivity ];
      cbv iota; lia. }
  assert (Hid : In (atom_val bsz data i)
                   (dict_of (map (atom_val bsz data) (seq 0 ne)) bsz)).
  { rewrite in_dict by exact Hab. exact Hiat. }
  destruct (idx_of_spec _ _ Hid) as [Hix_lt Hix_nth].
  set (ix := idx_of (atom_val bsz data i)
                    (dict_of (map (atom_val bsz data) (seq 0 ne)) bsz)).
  assert (Hr : read_idx (bytes_of
    (flat_map (fun x => bits_of (ib_of (length (dict_of
        (map (atom_val bsz data) (seq 0 ne)) bsz))) x) idxs))
    (ib_of (length (dict_of (map (atom_val bsz data) (seq 0 ne)) bsz))) i
    = Z.of_nat ix).
  { rewrite (read_idx_ok _ _ idxs
             (ib_of (length (dict_of (map (atom_val bsz data) (seq 0 ne)) bsz))) i
             ltac:(apply Forall_flat_map; intros x _; apply bits_of_bit01)
             ltac:(rewrite Hidxs; apply Forall_forall; intros x Hx;
                   apply in_map_iff in Hx; destruct Hx as [a [<- _]];
                   apply zofnat_nonneg)
             ltac:(reflexivity)
             ltac:(reflexivity)
             ltac:(rewrite Hidxs; rewrite length_map, length_map, length_seq; exact Hi)).
    rewrite Hidxs.
    rewrite map_nth with (da := 0)
      by (rewrite length_map, length_seq; exact Hi).
    rewrite Ha_nth. fold ix.
    rewrite Z.mod_small; auto.
    split; [apply zofnat_nonneg|].
    eapply Z.lt_le_trans; [apply zofnat_lt; exact Hix_lt|].
    apply ib_bound. exact Hk1. }
  rewrite Hr. rewrite Nat2Z.id. cbv zeta. subst ix.
  destruct (Z.eqb bsz 2) eqn:Eb2.
  - (* bsz = 2 *)
    apply Z.eqb_eq in Eb2. subst bsz.
    replace (Z.to_nat 2) with 2%nat in * by reflexivity.
    rewrite nth_le2_lo, nth_le2_hi. rewrite Hix_nth.
    unfold atom_val. replace (Z.eqb 2 2) with true by reflexivity.
    cbv iota.
    rewrite firstn2_nth
      by (rewrite length_skipn; lia).
    rewrite nth_skipn. rewrite nth_skipn.
    pose proof (Forall_nth_Z _ _ (2 * i)%nat Hb ltac:(lia)) as [Hb0 _].
    pose proof (Forall_nth_Z _ _ (2 * i + 1)%nat Hb ltac:(lia)) as [Hb1 _].
    pose proof (Forall_nth_Z _ _ (2 * i)%nat Hb ltac:(lia)) as [_ Hb0h].
    pose proof (Forall_nth_Z _ _ (2 * i + 1)%nat Hb ltac:(lia)) as [_ Hb1h].
    f_equal.
    + replace (nth (2 * i)%nat data 0 + 256 * nth (2 * i + 1)%nat data 0)
        with (nth (2 * i)%nat data 0 + nth (2 * i + 1)%nat data 0 * 256) by ring.
      rewrite Z.mod_add by lia.
      rewrite Z.mod_small by lia.
      replace (2 * i + 0)%nat with (2 * i)%nat by lia.
      reflexivity.
    + replace (nth (2 * i)%nat data 0 + 256 * nth (2 * i + 1)%nat data 0)
        with (nth (2 * i + 1)%nat data 0 * 256 + nth (2 * i)%nat data 0) by ring.
      rewrite Z.div_add_l by lia.
      rewrite (Z.div_small (nth (2 * i)%nat data 0)) by lia.
      rewrite Z.add_0_r. rewrite Z.mod_small by lia.
      reflexivity.
  - (* bsz = 1 *)
    apply Z.eqb_neq in Eb2.
    assert (bsz = 1) by (destruct Hbsz; congruence). subst bsz.
    replace (Z.to_nat 1) with 1%nat in * by reflexivity.
    rewrite nth_le2_lo. rewrite Hix_nth.
    unfold atom_val. replace (Z.eqb 1 2) with false by reflexivity.
    cbv iota.
    rewrite firstn1_nth
      by (rewrite length_skipn; lia).
    rewrite nth_skipn.
    pose proof (Forall_nth_Z _ _ i Hb ltac:(lia)) as [Hbil Hbi].
    rewrite Z.mod_small by lia.
    f_equal. f_equal. lia.
Qed.

(* ================= main theorem ================= *)

Theorem pack_round_trip : forall (data : list Z) (bsz : Z),
  Forall byteb data ->
  (bsz = 1 \/ bsz = 2) ->
  (length data = length data / Z.to_nat bsz * Z.to_nat bsz)%nat ->
  (1 <= length data / Z.to_nat bsz)%nat ->
  (length (dict_of (map (atom_val bsz data)
                        (seq 0 (length data / Z.to_nat bsz))) bsz)
   <= 256)%nat ->
  dec_model (enc_record bsz data) bsz (length data) = data.
Proof.
  intros data bsz Hb Hbsz Hlen Hne Hk.
  unfold dec_model, enc_record. cbv beta zeta.
  set (bszn := Z.to_nat bsz).
  set (ne := (length data / bszn)%nat).
  set (atoms := map (atom_val bsz data) (seq 0 ne)).
  set (dict := dict_of atoms bsz).
  set (ib := ib_of (length dict)).
  set (idxs := map (fun a => Z.of_nat (idx_of a dict)) atoms).
  set (bits := flat_map (fun x => bits_of ib x) idxs).
  assert (Hlen2 : (length data = ne * bszn)%nat) by exact Hlen.
  assert (Hne2 : (1 <= ne)%nat) by exact Hne.
  assert (Hk2 : (length dict <= 256)%nat) by exact Hk.
  assert (Hin0 : In (atom_val bsz data 0) dict).
  { subst dict. rewrite in_dict.
    - apply in_map. apply in_seq. unfold ne. lia.
    - apply atom_bound; auto.
      destruct Hbsz as [->| ->];
        [ replace (Z.eqb 1 2) with false by reflexivity
        | replace (Z.eqb 2 2) with true by reflexivity ];
        cbv iota; unfold bszn; simpl; rewrite Hlen2; lia. }
  assert (Hk1 : (1 <= length dict)%nat).
  { apply in_length_pos with (a := atom_val bsz data 0). exact Hin0. }
  rewrite firstn4_le4.
  rewrite le4_inv.
  2: { split; [apply zofnat_nonneg|].
       eapply Z.le_lt_trans.
       - apply Nat2Z.inj_le. exact Hk2.
       - reflexivity. }
  rewrite Nat2Z.id.
  rewrite skipn4_le4. rewrite dictb_eq. rewrite stream_eq.
  (* decoder ne = encoder ne *)
  assert (Hext : forall i, In i (seq 0 ne) ->
    (fun i0 => let idx := Z.to_nat (read_idx (bytes_of bits) ib i0) in
               if Z.eqb bsz 2
               then [nth (2 * idx)%nat (flat_map le2 dict) 0;
                     nth (2 * idx + 1)%nat (flat_map le2 dict) 0]
               else [nth (2 * idx)%nat (flat_map le2 dict) 0]) i
    = firstn bszn (skipn (bszn * i) data)).
  { intros i Hin. apply in_seq in Hin. cbv beta.
    apply (emit_eq bsz data atoms dict bits ib ne i); auto; try reflexivity; lia. }
  change (flat_map
    (fun i0 => let idx := Z.to_nat (read_idx (bytes_of bits) ib i0) in
               if Z.eqb bsz 2
               then [nth (2 * idx)%nat (flat_map le2 dict) 0;
                     nth (2 * idx + 1)%nat (flat_map le2 dict) 0]
               else [nth (2 * idx)%nat (flat_map le2 dict) 0])
    (seq 0 ne) = data).
  rewrite (flat_map_ext_in _ _ _ _ _ Hext).
  rewrite flat_map_slices.
  rewrite firstn_all2 by lia.
  reflexivity.
Qed.

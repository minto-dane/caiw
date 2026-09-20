(* Utf8.v — Coq proof that caiw's utf8_ok accepts exactly the
 * well-formed UTF-8 strings (RFC 3629).
 *
 * Algorithm-level proof, same tier as Rans.v / TLA+ / Alloy: it models
 * the C byte-scan as a list function and proves it equivalent to an
 * independent specification of well-formedness — over ARBITRARY input
 * length.  CBMC's bounded scan tops out at n<=16; WP proved memory
 * safety for n<=2^30 but not the accept condition.  This closes the
 * semantic gap: no overlongs, no lone surrogates, no >U+10FFFF, no
 * truncation, no stray continuation lead bytes.
 *
 * C model (caiw.c):
 *   for (i = 0; i < n;) {
 *     c = s[i]; if (c < 0x80) { i++; continue; }
 *     if (c < 0xC0) return 0;
 *     else if (c < 0xE0) { cp = c&0x1F; w=2; if (cp<2) return 0; }
 *     else if (c < 0xF0) { cp = c&0x0F; w=3; }
 *     else if (c < 0xF5) { cp = c&7;    w=4; }
 *     else return 0;
 *     if (i + w > n) return 0;
 *     for (k = 1; k < w; k++)
 *       { t = s[i+k]; if ((t&0xC0)!=0x80) return 0; cp = (cp<<6)|(t&0x3F); }
 *     if (cp>=0xD800 && cp<=0xDFFF) return 0;
 *     if (w==3 && cp<0x800) return 0;
 *     if (w==4 && (cp<0x10000 || cp>0x10FFFF)) return 0;
 *     i += w; }
 *   return 1;
 *)

From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Open Scope Z_scope.
Open Scope bool_scope.

(* ------------------- implementation model ------------------- *)
(* Byte-domain hypothesis: every byte of the input lies in [0,256).   *)

Definition contb (b : Z) : bool := (128 <=? b) && (b <=? 191).

(* The scan, one sequence at a time; consumes 1..4 bytes per step. *)
Fixpoint uok (s : list Z) : bool :=
  match s with
  | [] => true
  | c :: r =>
      if c <? 128 then uok r
      else if c <? 194 then false              (* stray cont / C0,C1 *)
      else if c <? 224 then                    (* w = 2 *)
        match r with
        | t :: r' => if contb t then uok r' else false
        | [] => false
        end
      else if c <? 240 then                    (* w = 3 *)
        match r with
        | t1 :: t2 :: r' =>
            if contb t1 && contb t2 then
              let cp := (c mod 16) * 4096
                        + (t1 mod 64) * 64 + (t2 mod 64) in
              if (2048 <=? cp) && negb ((55296 <=? cp) && (cp <=? 57343))
              then uok r' else false
            else false
        | _ => false
        end
      else if c <? 245 then                    (* w = 4 *)
        match r with
        | t1 :: t2 :: t3 :: r' =>
            if contb t1 && contb t2 && contb t3 then
              let cp := (c mod 8) * 262144
                        + (t1 mod 64) * 4096
                        + (t2 mod 64) * 64 + (t3 mod 64) in
              if (65536 <=? cp) && (cp <=? 1114111)
              then uok r' else false
            else false
        | _ => false
        end
      else false
  end.

(* ------------------- specification ---------------------------- *)
(* enc_ok bs cp: bs is the canonical UTF-8 encoding of scalar cp.
 * Scalar = [0,0x10FFFF] minus surrogates; canonical = shortest form. *)

Definition enc_ok (bs : list Z) (cp : Z) : Prop :=
  match bs with
  | [b] => cp = b /\ 0 <= b < 128
  | [b0; b1] =>
      194 <= b0 <= 223 /\ 128 <= b1 <= 191 /\
      cp = (b0 - 192) * 64 + (b1 - 128)
  | [b0; b1; b2] =>
      224 <= b0 <= 239 /\ 128 <= b1 <= 191 /\ 128 <= b2 <= 191 /\
      cp = (b0 - 224) * 4096 + (b1 - 128) * 64 + (b2 - 128) /\
      2048 <= cp /\ ~ (55296 <= cp <= 57343)
  | [b0; b1; b2; b3] =>
      240 <= b0 <= 244 /\ 128 <= b1 <= 191 /\
      128 <= b2 <= 191 /\ 128 <= b3 <= 191 /\
      cp = (b0 - 240) * 262144 + (b1 - 128) * 4096
           + (b2 - 128) * 64 + (b3 - 128) /\
      65536 <= cp <= 1114111
  | _ => False
  end.

Inductive wutf8 : list Z -> Prop :=
| Wnil  : wutf8 []
| Wcons : forall bs rest cp,
            enc_ok bs cp -> wutf8 rest -> wutf8 (bs ++ rest).

Definition byte_dom (s : list Z) : Prop :=
  Forall (fun b => 0 <= b < 256) s.

(* ------------------- arithmetic helpers ------------------------ *)
(* cp in the model is built from (x mod w) factors; on the accepted
 * byte ranges these are plain offsets — discharge them once. *)

Lemma mod_lead3 : forall c, 224 <= c <= 239 -> c mod 16 = c - 224.
Proof. intros c Hc. replace c with (c - 224 + 14 * 16) at 1 by lia.
       rewrite Z.mod_add by lia. apply Z.mod_small. lia. Qed.

Lemma mod_lead4 : forall c, 240 <= c <= 244 -> c mod 8 = c - 240.
Proof. intros c Hc. replace c with (c - 240 + 30 * 8) at 1 by lia.
       rewrite Z.mod_add by lia. apply Z.mod_small. lia. Qed.

Lemma mod_cont : forall t, 128 <= t <= 191 -> t mod 64 = t - 128.
Proof. intros t Ht. replace t with (t - 128 + 2 * 64) at 1 by lia.
       rewrite Z.mod_add by lia. apply Z.mod_small. lia. Qed.

Lemma contb_spec : forall b, contb b = true <-> 128 <= b <= 191.
Proof.
  intro b. unfold contb. rewrite Bool.andb_true_iff.
  destruct (Z.leb_spec0 128 b); destruct (Z.leb_spec0 b 191); lia.
Qed.

(* ------------------- soundness: uok s -> wutf8 s --------------- *)

Lemma uok_sound : forall s,
  byte_dom s -> uok s = true -> wutf8 s.
Proof.
  (* strong induction on length — uok consumes 1..4 bytes per step *)
  intro s.
  remember (length s) as n eqn:Hn.
  revert s Hn.
  induction n as [n IH] using lt_wf_ind.
  intros s Hn Hb Hu.
  destruct s as [|c r]; simpl in *.
  - constructor.
  - (* peel the byte-domain invariant *)
    inversion Hb as [|b0 l Hb0 Hbl]; subst.
    destruct (Z.ltb_spec0 c 128) as [Hc|Hc].
    + (* ASCII step *)
      apply (Wcons [c] r c).
      * simpl. lia.
      * apply (IH (length r));
          [ simpl in *; lia | reflexivity | exact Hbl | exact Hu ].
    + destruct (Z.ltb_spec0 c 194) as [Hc2|Hc2]; [discriminate|].
      destruct (Z.ltb_spec0 c 224) as [Hc3|Hc3].
      * (* w = 2 *)
        destruct r as [|t r']; simpl in *; [discriminate|].
        inversion Hbl as [|b1 l Hb1 Hbl']; subst.
        destruct (contb t) eqn:Et; [|discriminate].
        apply contb_spec in Et.
        apply (Wcons [c; t] r' ((c - 192) * 64 + (t - 128))).
        -- simpl. lia.
        -- apply (IH (length r'));
             [ simpl in *; lia | reflexivity | exact Hbl' | exact Hu ].
      * destruct (Z.ltb_spec0 c 240) as [Hc4|Hc4].
        -- (* w = 3 *)
           destruct r as [|t1 [|t2 r']]; simpl in *; try discriminate.
           inversion Hbl as [|b1 l Hb1 Hbl1]; subst.
           inversion Hbl1 as [|b2 l' Hb2 Hbl2]; subst.
           destruct (contb t1) eqn:E1; [|discriminate].
           destruct (contb t2) eqn:E2; [|discriminate].
           apply contb_spec in E1; apply contb_spec in E2.
           rewrite mod_lead3, mod_cont, mod_cont in Hu by lia.
           set (cp := (c - 224) * 4096 + (t1 - 128) * 64 + (t2 - 128))
             in *.
           destruct (Z.leb_spec0 2048 cp) as [HL|HL]; [|discriminate].
           destruct (Z.leb_spec0 55296 cp) as [HLo|HLo].
           ++ destruct (Z.leb_spec0 cp 57343) as [HHi|HHi];
              simpl in Hu; try discriminate.
              apply (Wcons [c; t1; t2] r' cp).
              ** simpl. repeat split; lia.
              ** apply (IH (length r'));
                   [ simpl in *; lia | reflexivity
                   | exact Hbl2 | exact Hu ].
           ++ apply (Wcons [c; t1; t2] r' cp).
              ** simpl. repeat split; lia.
              ** apply (IH (length r'));
                   [ simpl in *; lia | reflexivity
                   | exact Hbl2 | exact Hu ].
        -- destruct (Z.ltb_spec0 c 245) as [Hc5|Hc5]; [|discriminate].
           (* w = 4 *)
           destruct r as [|t1 [|t2 [|t3 r']]]; simpl in *;
             try discriminate.
           inversion Hbl as [|b1 l Hb1 Hbl1]; subst.
           inversion Hbl1 as [|b2 l' Hb2 Hbl2]; subst.
           inversion Hbl2 as [|b3 l'' Hb3 Hbl3]; subst.
           destruct (contb t1) eqn:E1; [|discriminate].
           destruct (contb t2) eqn:E2; [|discriminate].
           destruct (contb t3) eqn:E3; [|discriminate].
           apply contb_spec in E1, E2, E3.
           rewrite mod_lead4, !mod_cont in Hu by lia.
           set (cp := (c - 240) * 262144 + (t1 - 128) * 4096
                      + (t2 - 128) * 64 + (t3 - 128)) in *.
           destruct (Z.leb_spec0 65536 cp) as [HL|HL]; [|
             discriminate].
           destruct (Z.leb_spec0 cp 1114111) as [HH|HH]; [|
             discriminate].
           apply (Wcons [c; t1; t2; t3] r' cp).
           ++ simpl. repeat split; lia.
           ++ apply (IH (length r'));
                [ simpl in *; lia | reflexivity
                | exact Hbl3 | exact Hu ].
Qed.

(* ------------------- completeness: wutf8 s -> uok s ------------ *)

Lemma wutf8_bytes : forall bs cp,
  enc_ok bs cp -> Forall (fun b => 0 <= b < 256) bs.
Proof.
  intros bs cp H. destruct bs as [|b0 [|b1 [|b2 [|b3 [|b4 rest]]]]];
    simpl in H; try contradiction.
  - destruct H as [-> Hb]. repeat constructor; lia.
  - destruct H as (H0 & H1 & _). repeat constructor; lia.
  - destruct H as (H0 & H1 & H2 & _). repeat constructor; lia.
  - destruct H as (H0 & H1 & H2 & H3 & _ & _). repeat constructor; lia.
Qed.

Lemma uok_complete : forall s,
  byte_dom s -> wutf8 s -> uok s = true.
Proof.
  intro s.
  remember (length s) as n eqn:Hn.
  revert s Hn.
  induction n as [n IH] using lt_wf_ind.
  intros s Hn Hb Hw.
  destruct Hw as [|bs rest cp Henc Hwrest].
  - reflexivity.
  - (* s = bs ++ rest — case on the sequence length *)
    destruct bs as [|b0 [|b1 [|b2 [|b3 [|b4 rest']]]]];
      simpl in Henc; try contradiction;
      simpl in *.
    + (* 1-byte *)
      destruct Henc as [-> Hb0]. simpl.
      destruct (Z.ltb_spec0 b0 128); [|lia].
      inversion Hb as [|x1 l1 Hx0 Hb']; subst.
      apply (IH (length rest));
        [ simpl in *; lia | reflexivity | exact Hb' | exact Hwrest ].
    + (* 2-byte *)
      destruct Henc as (H0 & H1 & ->). simpl.
      inversion Hb as [|x1 l1 Hx0 Hb']; subst.
      inversion Hb' as [|x2 l2 Hx1 Hb'']; subst.
      destruct (Z.ltb_spec0 b0 128); [lia|].
      destruct (Z.ltb_spec0 b0 194); [lia|].
      destruct (Z.ltb_spec0 b0 224); [|lia].
      assert (Et : contb b1 = true) by (apply contb_spec; lia).
      rewrite Et. simpl.
      apply (IH (length rest));
        [ simpl in *; lia | reflexivity | exact Hb'' | exact Hwrest ].
    + (* 3-byte *)
      destruct Henc as (H0 & H1 & H2 & -> & Hlo & Hns). simpl.
      inversion Hb as [|x1 l1 Hx0 Hb']; subst.
      inversion Hb' as [|x2 l2 Hx1 Hb'']; subst.
      inversion Hb'' as [|x3 l3 Hx2 Hb''']; subst.
      destruct (Z.ltb_spec0 b0 128); [lia|].
      destruct (Z.ltb_spec0 b0 194); [lia|].
      destruct (Z.ltb_spec0 b0 224); [lia|].
      destruct (Z.ltb_spec0 b0 240); [|lia].
      assert (E1 : contb b1 = true) by (apply contb_spec; lia).
      assert (E2 : contb b2 = true) by (apply contb_spec; lia).
      rewrite E1, E2. simpl.
      rewrite mod_lead3, !mod_cont by lia.
      destruct (Z.leb_spec0 2048
        ((b0 - 224) * 4096 + (b1 - 128) * 64 + (b2 - 128)))
        as [HL|HL]; [|lia].
      destruct (Z.leb_spec0 55296
        ((b0 - 224) * 4096 + (b1 - 128) * 64 + (b2 - 128)))
        as [HLo|HLo].
      * destruct (Z.leb_spec0
          ((b0 - 224) * 4096 + (b1 - 128) * 64 + (b2 - 128)) 57343)
          as [HHi|HHi]; simpl.
        -- exfalso. apply Hns. lia.
        -- apply (IH (length rest));
             [ simpl in *; lia | reflexivity
             | exact Hb''' | exact Hwrest ].
      * simpl. apply (IH (length rest));
          [ simpl in *; lia | reflexivity | exact Hb''' | exact Hwrest ].
    + (* 4-byte *)
      destruct Henc as (H0 & H1 & H2 & H3 & -> & Hlohi). simpl.
      inversion Hb as [|x1 l1 Hx0 Hb']; subst.
      inversion Hb' as [|x2 l2 Hx1 Hb'']; subst.
      inversion Hb'' as [|x3 l3 Hx2 Hb''']; subst.
      inversion Hb''' as [|x4 l4 Hx3 Hb'''']; subst.
      destruct (Z.ltb_spec0 b0 128); [lia|].
      destruct (Z.ltb_spec0 b0 194); [lia|].
      destruct (Z.ltb_spec0 b0 224); [lia|].
      destruct (Z.ltb_spec0 b0 240); [lia|].
      destruct (Z.ltb_spec0 b0 245); [|lia].
      assert (E1 : contb b1 = true) by (apply contb_spec; lia).
      assert (E2 : contb b2 = true) by (apply contb_spec; lia).
      assert (E3 : contb b3 = true) by (apply contb_spec; lia).
      rewrite E1, E2, E3. simpl.
      rewrite mod_lead4, !mod_cont by lia.
      destruct (Z.leb_spec0 65536
        ((b0 - 240) * 262144 + (b1 - 128) * 4096
         + (b2 - 128) * 64 + (b3 - 128))) as [HL|HL]; [|lia].
      destruct (Z.leb_spec0
        ((b0 - 240) * 262144 + (b1 - 128) * 4096
         + (b2 - 128) * 64 + (b3 - 128)) 1114111) as [HH|HH]; [|lia].
      apply (IH (length rest));
        [ simpl in *; lia | reflexivity | exact Hb'''' | exact Hwrest ].
Qed.

(* ------------------- the equivalence --------------------------- *)

Theorem utf8_ok_spec : forall s,
  byte_dom s -> (uok s = true <-> wutf8 s).
Proof.
  intros s Hb. split.
  - apply uok_sound; exact Hb.
  - apply uok_complete; exact Hb.
Qed.

(* Rans.v — Coq proof of the general-domain rANS single-symbol round trip.
 *
 * This is an ALGORITHM-level proof (same tier as the TLA+/Alloy models):
 * it proves the arithmetic inverse relation that every BMC engine timed
 * out on (CBMC minisat+Z3; ESBMC Bitwuzla/Z3/CVC5/Boolector — the
 * nondeterministic-divisor problem).  It does NOT verify the C code;
 * correspondence is by inspection of the 3-line enc/dec in caiw.c, and
 * the C level is covered by the bounded cbmc_rans_f sweep.
 *
 * C model (caiw.c):
 *   enc(x,f,c): while x >= XMAX(f) { *--pp = x & 0xff; x >>= 8 }
 *               return ((x/f) << SB) + (x % f) + c
 *   dec(x,f,c): x = f*(x >> SB) + (x & (TOT-1)) - c
 *               while (x < LOWER && rp < end) x = (x<<8) | *rp++
 *   SB=15  TOT=2^15  LOWER=2^31  XMAX(f)=((LOWER<<8)/TOT)*f = 2^24*f
 *
 * Emitted bytes are prepended (--pp), so the stream stores them
 * LAST-emitted-first; dec reads forward, shifting them back in the same
 * order.  Proved: for LOWER <= x < LOWER*256, 1<=f<=TOT and
 * 0<=c<=TOT-f, dec(enc(x,f,c)) = x — exactly the general domain that
 * defeats SAT/SMT bit-blasting.  At most 2 bytes are ever emitted on
 * this domain; the proof case-splits on the emit count.
 *)

From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Open Scope Z_scope.

Definition TOT : Z := 32768.            (* 1 << 15 *)
Definition LOWER : Z := 2147483648.     (* 1 << 31 *)
Definition XMAX (f : Z) : Z := 16777216 * f.   (* (LOWER<<8)/TOT = 2^24 *)

(* Encoder renormalization: returns (final state, stream bytes).
 * Stream order is last-emitted-first to match dec's forward reads.
 * Fuel 8 far exceeds the true emit count (<= 2) on our domain. *)
Fixpoint enc_renorm (x f : Z) (fuel : nat) : Z * list Z :=
  match fuel with
  | O => (x, [])
  | S n =>
      if x >=? XMAX f then
        let '(xk, bs) := enc_renorm (x / 256) f n in
        (xk, bs ++ [x mod 256])
      else (x, [])
  end.

Definition enc (x f c : Z) : Z * list Z :=
  let '(xk, bs) := enc_renorm x f 8 in
  ((xk / f) * TOT + xk mod f + c, bs).

(* Decoder: un-mix the slot, then shift bytes while x < LOWER
 * (stops at stream end too — mirrors *rp < end). *)
Fixpoint dec_renorm (x : Z) (bs : list Z) : Z :=
  match bs with
  | [] => x
  | b :: tl => if x <? LOWER then dec_renorm (x * 256 + b) tl else x
  end.

Definition dec (x f c : Z) (bs : list Z) : Z :=
  dec_renorm (f * (x / TOT) + x mod TOT - c) bs.

(* ---------- core lemma: the slot carries xk exactly ------------------ *)
(* slot = (x/f)*TOT + r where r = x mod f + c satisfies 0 <= r < TOT,
   so slot/TOT = x/f and slot mod TOT = r; then f*(x/f)+x mod f = x. *)
Lemma slot_unmix : forall x f c,
  1 <= f ->
  0 <= c <= TOT - f ->
  f * (((x / f) * TOT + x mod f + c) / TOT)
    + (((x / f) * TOT + x mod f + c) mod TOT) - c = x.
Proof.
  intros x f c Hf Hc.
  assert (Hmf : 0 <= x mod f < f) by (apply Z.mod_pos_bound; lia).
  assert (Hr : 0 <= x mod f + c < TOT) by (unfold TOT in *; lia).
  (* normalise the goal's left-associative sum to A + (x mod f + c) *)
  replace ((x / f) * TOT + x mod f + c)
    with ((x / f) * TOT + (x mod f + c)) by ring.
  set (s := (x / f) * TOT + (x mod f + c)).
  (* quotient/remainder uniqueness discharges slot/TOT and slot mod TOT
     in one step *)
  assert (Hqr : s / TOT = x / f /\ s mod TOT = x mod f + c).
  { destruct (Z.div_mod_unique TOT (s / TOT) (x / f) (s mod TOT)
               (x mod f + c)) as [Hq Hm].
    - left. apply Z.mod_pos_bound. unfold TOT; lia.
    - left. exact Hr.
    - assert (s = TOT * (s / TOT) + s mod TOT)
        by (apply Z.div_mod; unfold TOT; lia).
      unfold s in *. lia.
    - split; assumption. }
  destruct Hqr as [Hq Hm].
  rewrite Hq, Hm.
  assert (Hdm : x = f * (x / f) + x mod f)
    by (apply Z.div_mod; lia).
  lia.
Qed.

(* ---------- byte reconstruction -------------------------------------- *)
Lemma byte_rejoin : forall x, x / 256 * 256 + x mod 256 = x.
Proof.
  intro x.
  assert (x = 256 * (x / 256) + x mod 256)
    by (apply Z.div_mod; lia).
  lia.
Qed.

(* ---------- domain facts --------------------------------------------- *)
Lemma sh1_lt_lower : forall x, x < LOWER * 256 -> x / 256 < LOWER.
Proof.
  intros x Hx. unfold LOWER in *.
  apply Z.div_lt_upper_bound; lia.
Qed.

Lemma sh2_lt_xmax : forall x f,
  1 <= f -> x < LOWER * 256 -> x / 256 / 256 < XMAX f.
Proof.
  intros x f Hf Hx. unfold XMAX, LOWER in *.
  rewrite Z.div_div by lia.
  assert (x / (256 * 256) < 8388608).
  { apply Z.div_lt_upper_bound; lia. }
  set (y := x / (256 * 256)) in *. lia.
Qed.

(* ---------- main theorem --------------------------------------------- *)
Theorem rans_round_trip : forall x f c,
  1 <= f <= TOT ->
  0 <= c <= TOT - f ->
  LOWER <= x < LOWER * 256 ->
  let '(s, bs) := enc x f c in dec s f c bs = x.
Proof.
  intros x f c Hf Hc Hx. unfold enc, dec.
  cbn [enc_renorm].
  destruct (Z.geb_spec x (XMAX f)) as [E1|E1].
  - (* k >= 1 *)
    destruct (Z.geb_spec (x / 256) (XMAX f)) as [E2|E2].
    + destruct (Z.geb_spec (x / 256 / 256) (XMAX f)) as [E3|E3].
      * exfalso. pose proof (sh2_lt_xmax x f (proj1 Hf) (proj2 Hx)). lia.
      * (* k = 2: bs = [x/256 mod 256 ; x mod 256], xk = x/256/256 *)
        rewrite slot_unmix by lia.
        cbn [dec_renorm app].
        destruct (Z.ltb_spec0 (x / 256 / 256) LOWER) as [L1|L1].
        -- rewrite byte_rejoin.
           destruct (Z.ltb_spec0 (x / 256) LOWER) as [L2|L2].
           ++ rewrite byte_rejoin. reflexivity.
           ++ exfalso. pose proof (sh1_lt_lower x (proj2 Hx)). lia.
        -- exfalso.
           assert (x / 256 / 256 < LOWER).
           { rewrite Z.div_div by lia. unfold LOWER in *.
             apply Z.div_lt_upper_bound; lia. }
           lia.
    + (* k = 1: bs = [x mod 256], xk = x/256 *)
      rewrite slot_unmix by lia.
      cbn [dec_renorm app].
      destruct (Z.ltb_spec0 (x / 256) LOWER) as [L1|L1].
      * rewrite byte_rejoin. reflexivity.
      * exfalso. pose proof (sh1_lt_lower x (proj2 Hx)). lia.
  - (* k = 0: no bytes emitted; the slot alone carries x *)
    rewrite slot_unmix by lia. reflexivity.
Qed.

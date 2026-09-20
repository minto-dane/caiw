(* Extract the proved models to OCaml for differential testing
   against the real C implementation (diff_jkey.sh / diff_models.sh).
   ZBigInt maps Z to Zarith: the default unary of_nat makes
   seq-0-65536 dict scans and u64 histograms infeasible. *)
From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt
     ExtrOcamlZBigInt.
Require Import Jkey Utf8 Norm Pack Rans.

(* ZBigInt leaves Z.of_nat as the unary Pos.of_succ_nat — remap it to the
   O(1) bigint constructor (nat extracts to int; of_nat is exact on it). *)
Extract Constant BinInt.Z.of_nat => "Big_int_Z.big_int_of_int".
Extraction "jkeym.ml" jkey_bytes.
Extraction "utf8m.ml" uok_bytes.
Extraction "normm.ml" norm_ctx.
Extraction "packm.ml" enc_record dec_model.
Extraction "ransm.ml" enc dec.

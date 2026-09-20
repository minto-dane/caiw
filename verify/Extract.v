(* Extract the proved jkey model to OCaml for differential testing
   against the real C implementation (diff_jkey.sh). *)
From Stdlib Require Import Extraction ExtrOcamlBasic ExtrOcamlNatInt.
Require Import Jkey.
Extraction "jkeym.ml" jkey_bytes.

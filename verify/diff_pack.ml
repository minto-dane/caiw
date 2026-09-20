(* OCaml driver for extracted enc_record/dec_model (PACK round trip).
   Input: [u8 bsz][u64 n][n bytes]. Prints record hex + decoded hex. *)
let phex l =
  List.iter (fun z -> Printf.printf "%02x" (Z.to_int z)) l;
  print_newline ()

let run1 path =
  let ic = open_in_bin path in
  let u64 () =
    let v = ref 0L in
    for j = 0 to 7 do
      v := Int64.logor !v
             (Int64.shift_left (Int64.of_int (input_byte ic)) (8 * j))
    done; !v in
  let bsz = input_byte ic in
  let n = Int64.to_int (u64 ()) in
  let s = really_input_string ic n in
  close_in ic;
  let data =
    List.init n (fun i -> Z.of_int (Char.code s.[i])) in
  let bz = Z.of_int bsz in
  let rec_ = Packm.enc_record bz data in
  phex rec_;
  phex (Packm.dec_model rec_ bz n)

let () =
  for i = 1 to Array.length Sys.argv - 1 do
    run1 Sys.argv.(i)
  done

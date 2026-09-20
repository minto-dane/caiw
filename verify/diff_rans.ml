(* OCaml driver for extracted rANS enc/dec (Zarith-backed z).
   Input: [u64 x][u64 f][u64 c] (LE). Prints x', stream hex, x''.
   Note: C wraps at u64; the proved domain is x in [LOWER,256*LOWER)
   where the slot fits — the generator stays inside it. *)
let run1 path =
  let ic = open_in_bin path in
  let u64 () =
    let v = ref 0L in
    for j = 0 to 7 do
      v := Int64.logor !v
             (Int64.shift_left (Int64.of_int (input_byte ic)) (8 * j))
    done; !v in
  let x = u64 () and f = u64 () and c = u64 () in
  close_in ic;
  if f = 0L then print_string "skip\n"
  else begin
    let (x2, bs) = Ransm.enc (Z.of_int64 x) (Z.of_int64 f)
        (Z.of_int64 c) in
    Printf.printf "%s\n" (Z.to_string x2);
    List.iter (fun z -> Printf.printf "%02x" (Z.to_int z)) bs;
    print_newline ();
    Printf.printf "%s\n"
      (Z.to_string (Ransm.dec x2 (Z.of_int64 f) (Z.of_int64 c) bs))
  end

let () =
  for i = 1 to Array.length Sys.argv - 1 do
    run1 Sys.argv.(i)
  done

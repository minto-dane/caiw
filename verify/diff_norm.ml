(* OCaml driver for the extracted norm_ctx model (Zarith-backed z).
   Input: [u32 aw][aw x u64 LE]. Prints B, f[i] per line, E. *)
let run1 path =
  let ic = open_in_bin path in
  let u32 () =
    let b0 = input_byte ic and b1 = input_byte ic
    and b2 = input_byte ic and b3 = input_byte ic in
    b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (b3 lsl 24) in
  let u64 () =
    let v = ref 0L in
    for j = 0 to 7 do
      v := Int64.logor !v
             (Int64.shift_left (Int64.of_int (input_byte ic)) (8 * j))
    done; !v in
  let aw = u32 () in
  let h = List.init aw (fun _ -> Z.of_int64 (u64 ())) in
  close_in ic;
  let f = Normm.norm_ctx h in
  print_string "B\n";
  List.iter (fun z -> Printf.printf "%s\n" (Z.to_string z)) f;
  print_string "E\n"

let () =
  for i = 1 to Array.length Sys.argv - 1 do
    run1 Sys.argv.(i)
  done

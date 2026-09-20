(* OCaml driver for the extracted jkey model.
   Input file: [u32 olen][u32 rlen][u32 klen][obj][rest][key] (LE u32).
   Prints the returned object-relative offset, or -1 on failure.
   Accepts multiple case files; one output line per file. *)
let run1 path =
  let ic = open_in_bin path in
  let u32 () =
    let b0 = input_byte ic and b1 = input_byte ic
    and b2 = input_byte ic and b3 = input_byte ic in
    b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (b3 lsl 24) in
  let olen = u32 () and rlen = u32 () and klen = u32 () in
  let bytes n =
    let s = really_input_string ic n in
    List.init n (fun i -> Char.code s.[i]) in
  let obj = bytes olen and rest = bytes rlen and key = bytes klen in
  close_in ic;
  match Jkeym.jkey_bytes obj rest key with
  | Some v -> Printf.printf "%d\n" v
  | None -> print_string "-1\n"

let () =
  for i = 1 to Array.length Sys.argv - 1 do
    run1 Sys.argv.(i)
  done

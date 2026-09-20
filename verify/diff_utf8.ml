(* OCaml driver for the extracted uok (proved == RFC-3629 well-formed).
   Input: [u32 n][bytes]. Prints 0/1. Multiple files, one line each. *)
let run1 path =
  let ic = open_in_bin path in
  let u32 () =
    let b0 = input_byte ic and b1 = input_byte ic
    and b2 = input_byte ic and b3 = input_byte ic in
    b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (b3 lsl 24) in
  let n = u32 () in
  let s = really_input_string ic n in
  close_in ic;
  let bytes = List.init n (fun i -> Char.code s.[i]) in
  print_string (if Utf8m.uok_bytes bytes then "1\n" else "0\n")

let () =
  for i = 1 to Array.length Sys.argv - 1 do
    run1 Sys.argv.(i)
  done
